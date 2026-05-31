"""YouTube Fetcher — SearchAPI.io integration for channel videos and transcripts."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

_SEARCHAPI_BASE = "https://www.searchapi.io/api/v1/search"
_CHUNK_WORDS = 4000
_MAX_PAGES = 50

_YT_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]{11})"
)


def _chunk_text(text: str, max_words: int = _CHUNK_WORDS) -> list[str]:
    """Split text into chunks of approximately max_words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks or [""]


class Input(BaseModel):
    command: str = Field(
        description="fetch-channel-videos|fetch-transcript|fetch-all-channels|fetch-channel-full|ingest-url|search-videos"
    )
    channel_id: str = Field(default="", description="Internal channel_id")
    video_id: str = Field(default="", description="Internal video_id")
    url: str = Field(default="", description="YouTube URL (for ingest-url command)")
    query: str = Field(default="", description="Search query (for search-videos command)")
    max_results: int = Field(default=10, description="Max results for search-videos (default 10)")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    new_count: int = 0
    error: str = ""


class YouTubeFetcherTool(ScriptTool[Input, Output]):
    name = "youtube_fetcher"
    description = "Fetch YouTube channel videos and transcripts via SearchAPI.io"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.settings import get_settings
        from app.db.repos.research import TopicChannelLinkRepo

        api_key = get_settings().searchapi_key
        if not api_key:
            return Output(success=False, error="SEARCHAPI_KEY not configured")

        cmd = inp.command

        if cmd == "fetch-channel-videos":
            return await self._fetch_channel_videos(inp.channel_id, api_key)

        if cmd == "fetch-transcript":
            return await self._fetch_transcript(inp.video_id, api_key)

        if cmd == "fetch-all-channels":
            link_repo = TopicChannelLinkRepo()
            channel_ids = await link_repo.get_all_active_channel_ids()
            results: list[dict[str, Any]] = []
            total_new = 0
            for cid in channel_ids:
                r = await self._fetch_channel_videos(cid, api_key)
                results.append({"channel_id": cid, "new_count": r.new_count, "success": r.success})
                total_new += r.new_count
            return Output(success=True, items=results, count=len(results), new_count=total_new)

        if cmd == "fetch-channel-full":
            return await self._fetch_channel_full(inp.channel_id, api_key)

        if cmd == "ingest-url":
            return await self._ingest_url(inp.url, api_key)

        if cmd == "search-videos":
            return await self._search_videos(inp.query, api_key, max_results=inp.max_results)

        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _ingest_url(self, url: str, api_key: str) -> Output:
        """Parse a YouTube URL, create a standalone video record, and fetch its transcript."""
        from app.db.repos.research import YouTubeVideoRepo

        if not url:
            return Output(success=False, error="No URL provided")

        m = _YT_URL_RE.search(url)
        if not m:
            return Output(success=False, error=f"Could not extract YouTube video ID from URL: {url}")

        yt_video_id = m.group(1)
        vid_repo = YouTubeVideoRepo()

        # Check if already exists
        existing = await vid_repo.get_by_yt_id(yt_video_id)
        if existing:
            already_done = existing.get("transcript_status") == "done"
            if already_done:
                return Output(
                    success=True,
                    item={
                        "video_id": existing["video_id"],
                        "yt_video_id": yt_video_id,
                        "transcript_length": len(existing.get("transcript") or ""),
                        "already_existed": True,
                    },
                )
            # Exists but transcript not done — fetch transcript
            result = await self._fetch_transcript(existing["video_id"], api_key)
            result.item["already_existed"] = True
            result.item["yt_video_id"] = yt_video_id
            return result

        # Create new standalone video (no channel)
        vid_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(yt_video_id)) % 0xFFFFFFFF:08x}"
        await vid_repo.create_standalone(vid_id, yt_video_id)

        # Fetch transcript
        result = await self._fetch_transcript(vid_id, api_key)
        result.item["yt_video_id"] = yt_video_id
        result.item["already_existed"] = False
        return result

    async def _search_videos(self, query: str, api_key: str, *, max_results: int = 10) -> Output:
        """Search YouTube for videos matching a query and save new ones to DB."""
        from app.db.repos.research import YouTubeVideoRepo

        if not query:
            return Output(success=False, error="No query provided")

        vid_repo = YouTubeVideoRepo()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                _SEARCHAPI_BASE,
                params={
                    "engine": "youtube",
                    "q": query,
                    "api_key": api_key,
                },
            )
            if resp.status_code != 200:
                return Output(success=False, error=f"SearchAPI {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        video_results = data.get("video_results", data.get("videos", []))[:max_results]
        items: list[dict[str, Any]] = []
        new_count = 0

        for v in video_results:
            yt_vid_id = v.get("id", "")
            if not yt_vid_id:
                # Try extracting from link
                link = v.get("link", "")
                m = _YT_URL_RE.search(link)
                if m:
                    yt_vid_id = m.group(1)
            if not yt_vid_id:
                continue

            title = v.get("title", "")
            item = {
                "yt_video_id": yt_vid_id,
                "title": title,
                "description": v.get("description", v.get("snippet", "")),
                "channel": v.get("channel", {}).get("name", v.get("channel_name", "")),
                "views": v.get("views", v.get("view_count", "")),
                "published": v.get("published_date", v.get("date", "")),
                "length": v.get("length", v.get("duration", "")),
                "link": v.get("link", f"https://www.youtube.com/watch?v={yt_vid_id}"),
            }

            if not await vid_repo.exists_by_yt_id(yt_vid_id):
                vid_id = f"vid_{datetime.now().strftime('%Y%m%d')}_{id(v) % 0xFFFFFFFF:08x}"
                await vid_repo.create_standalone(vid_id, yt_vid_id, title=title)
                item["video_id"] = vid_id
                item["new"] = True
                new_count += 1
                # Cache the new video artifact
                try:
                    from app.db.repos.context_cache import add_to_cache

                    channel_name = item.get("channel", "")
                    desc = item.get("description", "")[:100]
                    await add_to_cache(
                        "youtube_video",
                        f"Found YouTube video: '{title}' by {channel_name} - {desc}",
                        entity_type="youtube_videos",
                        entity_id=vid_id,
                        tags=["youtube", "video"],
                        source_agent="youtube_fetcher",
                    )
                except Exception:
                    pass
            else:
                existing = await vid_repo.get_by_yt_id(yt_vid_id)
                item["video_id"] = existing["video_id"] if existing else ""
                item["new"] = False

            items.append(item)

        return Output(success=True, items=items, count=len(items), new_count=new_count)

    async def _fetch_channel_videos(self, channel_id: str, api_key: str) -> Output:
        from app.db.repos.research import YouTubeChannelRepo, YouTubeVideoRepo

        chan_repo = YouTubeChannelRepo()
        vid_repo = YouTubeVideoRepo()

        channel = await chan_repo.get(channel_id)
        if not channel:
            return Output(success=False, error=f"Channel not found: {channel_id}")

        yt_channel_id = channel["yt_channel_id"]

        async with httpx.AsyncClient(timeout=30) as client:
            data = None
            for attempt in range(3):
                try:
                    resp = await client.get(
                        _SEARCHAPI_BASE,
                        params={
                            "engine": "youtube_channel_videos",
                            "channel_id": yt_channel_id,
                            "api_key": api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt == 2:
                        return Output(success=False, error=f"Channel videos fetch failed after 3 attempts: {e}")
                    await asyncio.sleep(2**attempt)
            if data is None:
                return Output(success=False, error="Channel videos fetch returned no data")

        videos = data.get("videos", [])
        new_count = 0
        for v in videos:
            yt_vid_id = v.get("id", "")
            if not yt_vid_id:
                continue
            if await vid_repo.exists_by_yt_id(yt_vid_id):
                continue
            vid_id = f"vid_{datetime.now().strftime('%Y%m%d')}_{id(v) % 0xFFFFFFFF:08x}"
            await vid_repo.create(
                vid_id,
                yt_vid_id,
                channel_id,
                title=v.get("title", ""),
                raw_api_response=v,
            )
            new_count += 1

        await chan_repo.mark_fetched(channel_id)
        return Output(success=True, count=len(videos), new_count=new_count)

    async def _fetch_channel_full(self, channel_id: str, api_key: str) -> Output:
        """Paginate through ALL videos on a channel, then fetch transcripts for new ones."""
        from app.db.repos.research import YouTubeVideoRepo

        backfill = await self._fetch_channel_videos_paginated(channel_id, api_key)
        if not backfill.success:
            return backfill

        vid_repo = YouTubeVideoRepo()
        pending = await vid_repo.list_pending_for_channel(channel_id)
        transcripts_fetched = 0
        for v in pending:
            try:
                r = await self._fetch_transcript(v["video_id"], api_key)
                if r.success:
                    transcripts_fetched += 1
            except Exception:
                continue

        return Output(
            success=True,
            item={
                "total_videos_found": backfill.count,
                "new_videos_saved": backfill.new_count,
                "transcripts_fetched": transcripts_fetched,
                "pages_fetched": backfill.item.get("pages_fetched", 1),
            },
            count=backfill.count,
            new_count=backfill.new_count,
        )

    async def _fetch_channel_videos_paginated(self, channel_id: str, api_key: str) -> Output:
        """Fetch all videos for a channel, paginating through all pages."""
        from app.db.repos.research import YouTubeChannelRepo, YouTubeVideoRepo

        chan_repo = YouTubeChannelRepo()
        vid_repo = YouTubeVideoRepo()

        channel = await chan_repo.get(channel_id)
        if not channel:
            return Output(success=False, error=f"Channel not found: {channel_id}")

        yt_channel_id = channel["yt_channel_id"]
        total_count = 0
        new_count = 0
        page = 0
        next_page_token: str | None = None

        async with httpx.AsyncClient(timeout=30) as client:
            while page < _MAX_PAGES:
                params: dict[str, Any] = {
                    "engine": "youtube_channel_videos",
                    "channel_id": yt_channel_id,
                    "api_key": api_key,
                }
                if next_page_token:
                    params["next_page_token"] = next_page_token

                resp = await client.get(_SEARCHAPI_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()

                videos = data.get("videos", [])
                total_count += len(videos)
                for v in videos:
                    yt_vid_id = v.get("id", "")
                    if not yt_vid_id:
                        continue
                    if await vid_repo.exists_by_yt_id(yt_vid_id):
                        continue
                    vid_id = f"vid_{datetime.now().strftime('%Y%m%d')}_{id(v) % 0xFFFFFFFF:08x}"
                    await vid_repo.create(
                        vid_id,
                        yt_vid_id,
                        channel_id,
                        title=v.get("title", ""),
                        raw_api_response=v,
                    )
                    new_count += 1

                page += 1
                next_page_token = data.get("pagination", {}).get("next_page_token")
                if not next_page_token:
                    break

        await chan_repo.mark_fetched(channel_id)
        return Output(
            success=True,
            item={"pages_fetched": page},
            count=total_count,
            new_count=new_count,
        )

    @staticmethod
    async def _fetch_video_title_oembed(yt_video_id: str) -> str:
        """Fetch video title via YouTube oEmbed (free, no API key needed)."""
        try:
            url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={yt_video_id}&format=json"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.json().get("title", "")
        except Exception:
            pass
        return ""

    async def _fetch_transcript(self, video_id: str, api_key: str) -> Output:
        from app.db.repos.research import YouTubeVideoRepo

        vid_repo = YouTubeVideoRepo()
        video = await vid_repo.get(video_id)
        if not video:
            return Output(success=False, error=f"Video not found: {video_id}")

        yt_video_id = video["yt_video_id"]

        async with httpx.AsyncClient(timeout=60) as client:
            data = None
            for attempt in range(3):
                try:
                    resp = await client.get(
                        _SEARCHAPI_BASE,
                        params={
                            "engine": "youtube_transcripts",
                            "video_id": yt_video_id,
                            "api_key": api_key,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    if attempt == 2:
                        return Output(success=False, error=f"Transcript fetch failed after 3 attempts: {e}")
                    await asyncio.sleep(2**attempt)
            if data is None:
                return Output(success=False, error="Transcript fetch returned no data")

        transcripts = data.get("transcripts", [])
        full_text = " ".join(t.get("text", "") for t in transcripts)

        # Chunk long transcripts
        chunks = _chunk_text(full_text)
        transcript_raw = {
            "segments": transcripts,
            "chunks": chunks,
            "chunk_count": len(chunks),
        }

        await vid_repo.update_transcript(
            video_id,
            transcript=full_text,
            transcript_raw=transcript_raw,
            transcript_status="done",
        )

        # Cache the transcript artifact
        try:
            from app.db.repos.context_cache import add_to_cache

            # Prefer title from DB, then from API metadata, then oEmbed fallback
            title = video.get("title", "") or data.get("title", "") or data.get("search_metadata", {}).get("title", "")
            if not title:
                title = await self._fetch_video_title_oembed(yt_video_id)

            # Look up channel name if video has a channel_id
            channel_name = ""
            if video.get("channel_id"):
                try:
                    from app.db.repos.research import YouTubeChannelRepo

                    chan = await YouTubeChannelRepo().get(video["channel_id"])
                    if chan:
                        channel_name = chan.get("name", "")
                except Exception:
                    pass

            yt_url = f"https://youtube.com/watch?v={yt_video_id}"
            if title and channel_name:
                summary = f"'{title}' by {channel_name} — transcript ({len(full_text)} chars) — {yt_url}"
            elif title:
                summary = f"'{title}' — transcript ({len(full_text)} chars) — {yt_url}"
            else:
                summary = f"Video transcript ({len(full_text)} chars) — {yt_url}"

            # Also update the DB title if we found one from the API/oEmbed
            if title and not video.get("title"):
                try:
                    from sqlalchemy import text as sa_text

                    from app.db.session import execute_sql, get_async_session

                    async with get_async_session() as s:
                        await execute_sql(
                            s,
                            sa_text("UPDATE youtube_videos SET title = :title WHERE video_id = :vid"),
                            {"title": title, "vid": video_id},
                        )
                except Exception:
                    pass
            await add_to_cache(
                "youtube_transcript",
                summary,
                entity_type="youtube_videos",
                entity_id=video_id,
                tags=["youtube", "transcript"],
                source_agent="youtube_fetcher",
            )
        except Exception:
            pass  # Cache write is best-effort

        return Output(
            success=True,
            item={"video_id": video_id, "transcript_length": len(full_text), "chunk_count": len(chunks)},
        )


if __name__ == "__main__":
    YouTubeFetcherTool.run()
