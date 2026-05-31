"""Gmail API wrapper — sync methods, use asyncio.to_thread from tools."""

from __future__ import annotations

import base64
import re
from email.mime.text import MIMEText

from googleapiclient.discovery import build

from app.settings import get_settings
from app.services.google_auth import get_credentials


def _service(account: str = "primary"):
    return build("gmail", "v1", credentials=get_credentials(account))


def _assert_writable(account: str) -> None:
    """Raise PermissionError if account is read-only."""
    settings = get_settings()
    readonly_names = {name.strip().lower() for name in settings.gmail_readonly_accounts.split(",") if name.strip()}
    if account.lower() in readonly_names:
        raise PermissionError(f"Account '{account}' is read-only")


def _parse_headers(headers: list[dict]) -> dict[str, str]:
    """Extract common headers from Gmail message payload."""
    result = {}
    for h in headers:
        name = h.get("name", "").lower()
        if name in ("from", "to", "cc", "bcc", "subject", "date"):
            result[name] = h.get("value", "")
    return result


def _extract_body(payload: dict) -> str:
    """Extract plain text body from message payload (handles multipart)."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    parts = payload.get("parts", [])
    for part in parts:
        body = _extract_body(part)
        if body:
            return body

    return ""


def _format_message(msg: dict, full: bool = False) -> dict:
    """Format a Gmail message into a clean dict."""
    payload = msg.get("payload", {})
    headers = _parse_headers(payload.get("headers", []))

    result = {
        "id": msg.get("id", ""),
        "threadId": msg.get("threadId", ""),
        "snippet": msg.get("snippet", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "subject": headers.get("subject", ""),
        "date": headers.get("date", ""),
        "labelIds": msg.get("labelIds", []),
    }

    if full:
        result["body"] = _extract_body(payload)
        result["cc"] = headers.get("cc", "")
        result["bcc"] = headers.get("bcc", "")
        attachments = []
        for part in payload.get("parts", []):
            filename = part.get("filename")
            if filename:
                attachments.append(
                    {
                        "filename": filename,
                        "mimeType": part.get("mimeType", ""),
                        "size": part.get("body", {}).get("size", 0),
                    }
                )
        result["attachments"] = attachments

    return result


# ── Public API ──


def list_messages(
    query: str = "", max_results: int = 10, label_ids: list[str] | None = None, account: str = "primary"
) -> list[dict]:
    svc = _service(account)
    kwargs: dict = {"userId": "me", "maxResults": max_results}
    if query:
        kwargs["q"] = query
    if label_ids:
        kwargs["labelIds"] = label_ids

    resp = svc.users().messages().list(**kwargs).execute()
    messages = resp.get("messages", [])

    results = []
    for msg_ref in messages:
        msg = (
            svc.users()
            .messages()
            .get(userId="me", id=msg_ref["id"], format="metadata", metadataHeaders=["From", "To", "Subject", "Date"])
            .execute()
        )
        results.append(_format_message(msg))

    return results


def get_message(message_id: str, account: str = "primary") -> dict:
    svc = _service(account)
    msg = svc.users().messages().get(userId="me", id=message_id, format="full").execute()
    return _format_message(msg, full=True)


def get_thread(thread_id: str, account: str = "primary") -> dict:
    svc = _service(account)
    thread = svc.users().threads().get(userId="me", id=thread_id, format="full").execute()
    return {
        "id": thread.get("id", ""),
        "messages": [_format_message(m, full=True) for m in thread.get("messages", [])],
    }


def search(query: str, max_results: int = 10, account: str = "primary") -> list[dict]:
    return list_messages(query=query, max_results=max_results, account=account)


def get_labels(account: str = "primary") -> list[dict]:
    svc = _service(account)
    resp = svc.users().labels().list(userId="me").execute()
    return [
        {"id": label["id"], "name": label["name"], "type": label.get("type", "")} for label in resp.get("labels", [])
    ]


def mark_read(message_id: str, account: str = "primary") -> bool:
    _assert_writable(account)
    svc = _service(account)
    svc.users().messages().modify(userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}).execute()
    return True


def _validate_whitelist(recipients: list[str]) -> list[str]:
    """Return list of recipients NOT in the whitelist."""
    settings = get_settings()
    if not settings.gmail_whitelist:
        return []  # No whitelist configured = all allowed

    allowed = {addr.strip().lower() for addr in settings.gmail_whitelist.split(",") if addr.strip()}
    violations = []
    for addr in recipients:
        # Extract email from "Name <email>" format
        clean = addr.strip().lower()
        if "<" in clean:
            clean = clean.split("<")[1].rstrip(">")
        if clean not in allowed:
            violations.append(addr)
    return violations


def _collect_recipients(to: str, cc: str = "", bcc: str = "") -> list[str]:
    """Collect all recipient addresses."""
    recipients = []
    for field in (to, cc, bcc):
        if field:
            recipients.extend(addr.strip() for addr in field.split(",") if addr.strip())
    return recipients


def _looks_like_markdown(text: str) -> bool:
    """Check if text contains markdown formatting."""
    md_patterns = [
        r"^#{1,3}\s",  # headings
        r"\*\*[^*]+\*\*",  # bold
        r"\*[^*]+\*",  # italic
        r"^[-*]\s",  # unordered lists
        r"^\d+\.\s",  # ordered lists
        r"\[.+\]\(.+\)",  # links
        r"```",  # code blocks
    ]
    return any(re.search(p, text, re.MULTILINE) for p in md_patterns)


def _markdown_to_html(text: str) -> str:
    """Convert markdown to clean HTML email body. Lightweight, no dependencies."""
    # 1. Extract code blocks FIRST (before HTML escaping)
    code_blocks: list[str] = []

    def _save_code(m: re.Match) -> str:
        code_blocks.append(m.group(2))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text = re.sub(r"```\w*\n(.*?)```", _save_code, text, flags=re.DOTALL)

    # 2. Escape HTML entities on non-code content
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # 3. Headings
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)

    # 4. Bold and italic
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)

    # 5. Links
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)

    # 6. Unordered lists
    lines = text.split("\n")
    result: list[str] = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-*]\s", stripped):
            if not in_list:
                result.append("<ul>")
                in_list = True
            item = re.sub(r"^[-*]\s+", "", stripped)
            result.append(f"  <li>{item}</li>")
        else:
            if in_list:
                result.append("</ul>")
                in_list = False
            result.append(line)
    if in_list:
        result.append("</ul>")
    text = "\n".join(result)

    # 7. Paragraphs — double newlines become paragraph breaks
    text = re.sub(r"\n\n+", "</p>\n<p>", text)
    # Single newlines become <br> (but not inside block elements)
    text = text.replace("\n", "<br>\n")

    # 8. Restore code blocks (already HTML-safe, no <br> injection)
    for i, code in enumerate(code_blocks):
        escaped_code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(
            f"__CODE_BLOCK_{i}__",
            f'<pre style="background:#f5f5f5;padding:12px;border-radius:4px;overflow-x:auto">'
            f"<code>{escaped_code}</code></pre>",
        )

    return (
        "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        'font-size: 14px; line-height: 1.6; color: #333;">\n'
        f"<p>{text}</p>\n"
        "</div>"
    )


def create_draft(
    to: str,
    subject: str,
    body: str,
    cc: str = "",
    bcc: str = "",
    reply_to_message_id: str = "",
    html: bool = False,
    account: str = "primary",
) -> dict:
    """Create a Gmail draft. Validates recipients against whitelist."""
    _assert_writable(account)

    recipients = _collect_recipients(to, cc, bcc)
    violations = _validate_whitelist(recipients)
    if violations:
        return {"error": "whitelist_violation", "blocked_recipients": violations}

    # Auto-convert markdown to clean HTML if body contains markdown formatting
    if not html and _looks_like_markdown(body):
        body = _markdown_to_html(body)
        html = True
    mime_type = "html" if html else "plain"
    message = MIMEText(body, mime_type)
    message["to"] = to
    message["subject"] = subject
    if cc:
        message["cc"] = cc
    if bcc:
        message["bcc"] = bcc

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    draft_body: dict = {"message": {"raw": raw}}
    if reply_to_message_id:
        draft_body["message"]["threadId"] = reply_to_message_id

    svc = _service(account)
    draft = svc.users().drafts().create(userId="me", body=draft_body).execute()
    return {
        "draft_id": draft["id"],
        "message_id": draft.get("message", {}).get("id", ""),
    }


def send_draft(draft_id: str, account: str = "primary") -> dict:
    """Send an existing draft. Re-validates whitelist before sending."""
    _assert_writable(account)

    svc = _service(account)

    # Fetch draft to re-validate recipients
    draft = svc.users().drafts().get(userId="me", id=draft_id, format="full").execute()
    headers = _parse_headers(draft.get("message", {}).get("payload", {}).get("headers", []))
    recipients = _collect_recipients(headers.get("to", ""), headers.get("cc", ""), headers.get("bcc", ""))
    violations = _validate_whitelist(recipients)
    if violations:
        return {"error": "whitelist_violation", "blocked_recipients": violations}

    result = svc.users().drafts().send(userId="me", body={"id": draft_id}).execute()
    return {
        "message_id": result.get("id", ""),
        "threadId": result.get("threadId", ""),
    }


def list_drafts(max_results: int = 10, account: str = "primary") -> list[dict]:
    svc = _service(account)
    resp = svc.users().drafts().list(userId="me", maxResults=max_results).execute()
    drafts = resp.get("drafts", [])
    results = []
    for d in drafts:
        full = svc.users().drafts().get(userId="me", id=d["id"], format="metadata").execute()
        headers = _parse_headers(full.get("message", {}).get("payload", {}).get("headers", []))
        results.append(
            {
                "draft_id": d["id"],
                "to": headers.get("to", ""),
                "subject": headers.get("subject", ""),
                "date": headers.get("date", ""),
                "snippet": full.get("message", {}).get("snippet", ""),
            }
        )
    return results


def delete_draft(draft_id: str, account: str = "primary") -> bool:
    _assert_writable(account)
    svc = _service(account)
    svc.users().drafts().delete(userId="me", id=draft_id).execute()
    return True
