"""Link enrichment — fetch URL, parse title/description/og tags, cache in link_previews.

Zero-dep HTML parsing via stdlib html.parser. Best-effort: never raises; on failure
records status='error' so we don't retry a dead URL every call.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

_URL_RE = re.compile(
    r"https?://[^\s<>\"'\])]+",
    flags=re.IGNORECASE,
)

_TRAILING_PUNCT = ".,;:!?)]}>\"'"

_MAX_HTML_BYTES = 1_500_000  # 1.5 MB is plenty for <head>
_FETCH_TIMEOUT = 10.0
_USER_AGENT = "Mozilla/5.0 (compatible; FrenLinkEnricher/1.0)"


def extract_urls(text: str) -> list[str]:
    """Extract unique URLs from text, stripped of trailing punctuation."""
    if not text:
        return []
    found = _URL_RE.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for raw in found:
        url = raw.rstrip(_TRAILING_PUNCT)
        # Balance parens: if URL has more closing than opening, strip one
        while url.endswith(")") and url.count("(") < url.count(")"):
            url = url[:-1]
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


class _HeadParser(HTMLParser):
    """Extract <title>, <meta name=description>, and og:* tags from <head>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: str = ""
        self.description: str = ""
        self.site_name: str = ""
        self.og_title: str = ""
        self.og_description: str = ""
        self._in_title = False
        self._in_head = False
        self._stop = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._stop:
            return
        t = tag.lower()
        if t == "head":
            self._in_head = True
            return
        if t == "body":
            # Parse head only — once body starts we're done.
            self._stop = True
            return
        if t == "title":
            self._in_title = True
            return
        if t != "meta":
            return
        amap = {k.lower(): (v or "") for k, v in attrs}
        name = amap.get("name", "").lower()
        prop = amap.get("property", "").lower()
        content = amap.get("content", "").strip()
        if not content:
            return
        if name == "description" and not self.description:
            self.description = content
        elif prop == "og:title" and not self.og_title:
            self.og_title = content
        elif prop == "og:description" and not self.og_description:
            self.og_description = content
        elif prop == "og:site_name" and not self.site_name:
            self.site_name = content

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False
        elif tag.lower() == "head":
            self._stop = True

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            chunk = data.strip()
            if chunk:
                self.title = chunk


def parse_head(html: str) -> dict[str, str]:
    """Return title/description/og fields from HTML head. Empty strings when missing."""
    parser = _HeadParser()
    with contextlib.suppress(Exception):  # malformed HTML — partial data still useful
        parser.feed(html)
    return {
        "title": parser.title,
        "description": parser.description,
        "site_name": parser.site_name,
        "og_title": parser.og_title,
        "og_description": parser.og_description,
    }


def build_preview_text(preview: dict[str, str | None]) -> str:
    """Concatenate preview fields into one embedding-friendly blob."""
    parts: list[str] = []
    for k in ("title", "og_title"):
        v = preview.get(k)
        if v and v not in parts:
            parts.append(v)
    for k in ("description", "og_description"):
        v = preview.get(k)
        if v and v not in parts:
            parts.append(v)
    site = preview.get("site_name")
    if site and site not in parts:
        parts.append(site)
    return "\n".join(parts).strip()


async def fetch_preview(url: str, *, client: httpx.AsyncClient | None = None) -> dict[str, object]:
    """Fetch a URL and return a preview dict.

    Returns:
        {
            "url": str,
            "status": "ok" | "error" | "skip",
            "http_status": int | None,
            "title", "description", "site_name", "og_title", "og_description": str,
            "error": str,  # empty on success
        }
    """
    result: dict[str, object] = {
        "url": url,
        "status": "error",
        "http_status": None,
        "title": "",
        "description": "",
        "site_name": "",
        "og_title": "",
        "og_description": "",
        "error": "",
    }

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        result["error"] = "invalid scheme or host"
        return result

    # Skip binary-ish paths cheaply
    low = parsed.path.lower()
    if low.endswith((".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".mp4", ".mp3", ".webp", ".svg")):
        result["status"] = "skip"
        result["error"] = f"skipped non-html path {low}"
        return result

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,*/*"},
        )
    try:
        assert client is not None
        resp = await client.get(url)
        result["http_status"] = resp.status_code
        if resp.status_code >= 400:
            result["error"] = f"http {resp.status_code}"
            return result
        ctype = resp.headers.get("content-type", "").lower()
        if "html" not in ctype and "xml" not in ctype and ctype:
            result["status"] = "skip"
            result["error"] = f"non-html content-type: {ctype}"
            return result
        raw = resp.content[:_MAX_HTML_BYTES]
        # Decode carefully — fall back to utf-8 with replacement
        encoding = resp.encoding or "utf-8"
        try:
            html = raw.decode(encoding, errors="replace")
        except (LookupError, TypeError):
            html = raw.decode("utf-8", errors="replace")
        parsed_head = parse_head(html)
        result.update(parsed_head)
        result["status"] = "ok" if (parsed_head["title"] or parsed_head["og_title"]) else "error"
        if result["status"] == "error" and not result["error"]:
            result["error"] = "no title or og:title found"
    except httpx.HTTPError as e:
        result["error"] = f"{type(e).__name__}: {e}"
    except Exception as e:
        result["error"] = f"unexpected: {type(e).__name__}: {e}"
    finally:
        if owns_client and client is not None:
            await client.aclose()
    return result


async def fetch_previews(urls: list[str], *, concurrency: int = 4) -> list[dict[str, object]]:
    """Fetch multiple URLs concurrently with a semaphore."""
    if not urls:
        return []
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT, "Accept": "text/html,*/*"},
    ) as client:

        async def one(u: str) -> dict[str, object]:
            async with sem:
                return await fetch_preview(u, client=client)

        return await asyncio.gather(*(one(u) for u in urls))
