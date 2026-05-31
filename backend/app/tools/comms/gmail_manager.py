"""Gmail Manager — list, read, search, draft, and send emails."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="list|read|search|get-thread|get-labels|mark-read|create-draft|send-draft|list-drafts|delete-draft|check-auth|accounts"
    )
    account: str = Field(default="primary", description="Gmail account name (default: primary)")
    message_id: str = Field(default="", description="Gmail message ID")
    thread_id: str = Field(default="", description="Gmail thread ID")
    draft_id: str = Field(default="", description="Gmail draft ID")
    query: str = Field(default="", description="Search query (Gmail syntax)")
    max_results: int = Field(default=10, description="Max results to return")
    label_ids: str = Field(default="", description="Comma-separated label IDs to filter")
    to: str = Field(default="", description="Recipient email address(es)")
    cc: str = Field(default="", description="CC recipients")
    bcc: str = Field(default="", description="BCC recipients")
    subject: str = Field(default="", description="Email subject")
    body: str = Field(default="", description="Email body text")
    reply_to_message_id: str = Field(default="", description="Message ID to reply to")
    html: bool = Field(default=False, description="Send body as HTML")


class Output(BaseModel):
    success: bool = True
    message: dict = Field(default_factory=dict)
    messages: list[dict] = Field(default_factory=list)
    thread: dict = Field(default_factory=dict)
    draft: dict = Field(default_factory=dict)
    drafts: list[dict] = Field(default_factory=list)
    labels: list[dict] = Field(default_factory=list)
    accounts: list[dict] = Field(default_factory=list)
    count: int = 0
    whitelist_violation: list[str] = Field(default_factory=list)
    error: str = ""


class GmailManagerTool(ScriptTool[Input, Output]):
    name = "gmail_manager"
    description = "List, read, search, draft, and send emails via Gmail"
    stream_field = "body"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.services import gmail_client
        from app.services.google_auth import check_auth, list_accounts

        acct = inp.account

        if inp.command == "accounts":
            accts = list_accounts()
            return Output(success=True, accounts=accts, count=len(accts))

        if inp.command == "check-auth":
            result = check_auth(acct)
            return Output(success=result.get("authenticated", False), message=result)

        try:
            if inp.command == "list":
                label_ids = [lid.strip() for lid in inp.label_ids.split(",") if lid.strip()] if inp.label_ids else None
                msgs = gmail_client.list_messages(
                    query=inp.query, max_results=inp.max_results, label_ids=label_ids, account=acct
                )
                return Output(success=True, messages=msgs, count=len(msgs))

            if inp.command == "read":
                if not inp.message_id:
                    return Output(success=False, error="message_id required")
                msg = gmail_client.get_message(inp.message_id, account=acct)
                return Output(success=True, message=msg)

            if inp.command == "search":
                msgs = gmail_client.search(query=inp.query, max_results=inp.max_results, account=acct)
                return Output(success=True, messages=msgs, count=len(msgs))

            if inp.command == "get-thread":
                if not inp.thread_id:
                    return Output(success=False, error="thread_id required")
                thread = gmail_client.get_thread(inp.thread_id, account=acct)
                return Output(success=True, thread=thread)

            if inp.command == "get-labels":
                labels = gmail_client.get_labels(account=acct)
                return Output(success=True, labels=labels, count=len(labels))

            if inp.command == "mark-read":
                if not inp.message_id:
                    return Output(success=False, error="message_id required")
                gmail_client.mark_read(inp.message_id, account=acct)
                return Output(success=True)

            if inp.command == "create-draft":
                if not inp.to or not inp.subject:
                    return Output(success=False, error="to and subject required")
                result = gmail_client.create_draft(
                    to=inp.to,
                    subject=inp.subject,
                    body=inp.body,
                    cc=inp.cc,
                    bcc=inp.bcc,
                    reply_to_message_id=inp.reply_to_message_id,
                    html=inp.html,
                    account=acct,
                )
                if "error" in result and result["error"] == "whitelist_violation":
                    return Output(
                        success=False,
                        error="Recipients not in whitelist",
                        whitelist_violation=result["blocked_recipients"],
                    )
                return Output(success=True, draft=result)

            if inp.command == "send-draft":
                if not inp.draft_id:
                    return Output(success=False, error="draft_id required")
                result = gmail_client.send_draft(inp.draft_id, account=acct)
                if "error" in result and result["error"] == "whitelist_violation":
                    return Output(
                        success=False,
                        error="Recipients not in whitelist",
                        whitelist_violation=result["blocked_recipients"],
                    )
                return Output(success=True, message=result)

            if inp.command == "list-drafts":
                drafts = gmail_client.list_drafts(max_results=inp.max_results, account=acct)
                return Output(success=True, drafts=drafts, count=len(drafts))

            if inp.command == "delete-draft":
                if not inp.draft_id:
                    return Output(success=False, error="draft_id required")
                gmail_client.delete_draft(inp.draft_id, account=acct)
                return Output(success=True)

        except PermissionError as e:
            return Output(success=False, error=str(e))

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    GmailManagerTool.run()
