"""Context pinning tool — manage discussion topics and pinned context."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description=(
            "create-topic|end-topic|list-topics|update-topic|"
            "get-current-context|pin|unpin|get-pins|"
            "add-doc-ref|get-doc-refs|cleanup"
        )
    )
    topic_name: str = Field(default="", description="Topic name for create-topic")
    topic_summary: str = Field(default="", description="Topic summary for create/update-topic")
    content: str = Field(default="", description="Content text for pin command")
    content_type: str = Field(default="message", description="message|document_ref|summary|note")
    pin_id: int = Field(default=0, description="Pin ID for unpin command")
    topic_id: int = Field(default=0, description="Topic ID (0 = current active topic)")
    document_id: str = Field(default="", description="Document ID for add-doc-ref")
    reference_reason: str = Field(default="", description="Reason for document reference")
    relevance_score: float = Field(default=1.0, description="Relevance score for pin (0.0-1.0)")
    expires_hours: int = Field(default=24, description="Hours until pin expires (0 = never)")
    limit: int = Field(default=20, description="Max items to return")


class Output(BaseModel):
    success: bool = True
    topic: dict | None = None
    topics: list[dict] = Field(default_factory=list)
    pins: list[dict] = Field(default_factory=list)
    document_refs: list[dict] = Field(default_factory=list)
    context: dict | None = None
    count: int = 0
    error: str = ""


class ContextPinTool(ScriptTool[Input, Output]):
    name = "context_pin"
    description = "Manage discussion topics, pinned context, and document references"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _resolve_topic_id(self, inp: Input) -> int | None:
        """Resolve topic_id: use explicit value or fall back to active topic."""
        if inp.topic_id > 0:
            return inp.topic_id
        from app.db.repos.context_pins import ContextPinsRepo

        topic = await ContextPinsRepo().get_active_topic()
        return topic["id"] if topic else None

    async def _run(self, inp: Input) -> Output:
        from app.db.repos.context_pins import ContextPinsRepo

        repo = ContextPinsRepo()

        if inp.command == "create-topic":
            if not inp.topic_name:
                return Output(success=False, error="topic_name is required")
            topic = await repo.create_topic(inp.topic_name, inp.topic_summary)
            return Output(success=True, topic=topic)

        if inp.command == "end-topic":
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=False, error="No active topic to end")
            topic = await repo.end_topic(tid)
            return Output(success=True, topic=topic)

        if inp.command == "list-topics":
            topics = await repo.list_topics(limit=inp.limit)
            return Output(success=True, topics=topics, count=len(topics))

        if inp.command == "update-topic":
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=False, error="No topic to update")
            topic = await repo.update_topic(tid, inp.topic_summary)
            return Output(success=True, topic=topic)

        if inp.command == "get-current-context":
            ctx = await repo.get_current_context(message_limit=inp.limit)
            if not ctx:
                return Output(success=True, context=None)
            return Output(
                success=True,
                context=ctx,
                topic=ctx.get("topic"),
                pins=ctx.get("pins", []),
                document_refs=ctx.get("document_refs", []),
            )

        if inp.command == "pin":
            if not inp.content:
                return Output(success=False, error="content is required for pin")
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=False, error="No active topic to pin to")
            pin = await repo.create_pin(
                inp.content,
                inp.content_type,
                tid,
                relevance_score=inp.relevance_score,
                expires_hours=inp.expires_hours,
            )
            return Output(success=True, pins=[pin])

        if inp.command == "unpin":
            if inp.pin_id <= 0:
                return Output(success=False, error="pin_id is required for unpin")
            deleted = await repo.delete_pin(inp.pin_id)
            return Output(success=deleted, error="" if deleted else f"Pin {inp.pin_id} not found")

        if inp.command == "get-pins":
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=True, pins=[], count=0)
            ct = inp.content_type if inp.content_type != "message" else ""
            pins = await repo.get_pins(tid, content_type=ct, limit=inp.limit)
            return Output(success=True, pins=pins, count=len(pins))

        if inp.command == "add-doc-ref":
            if not inp.document_id:
                return Output(success=False, error="document_id is required")
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=False, error="No active topic for document reference")
            ref = await repo.add_document_ref(inp.document_id, tid, reference_reason=inp.reference_reason)
            return Output(success=True, document_refs=[ref])

        if inp.command == "get-doc-refs":
            tid = await self._resolve_topic_id(inp)
            if not tid:
                return Output(success=True, document_refs=[], count=0)
            refs = await repo.get_document_refs(tid, limit=inp.limit)
            return Output(success=True, document_refs=refs, count=len(refs))

        if inp.command == "cleanup":
            count = await repo.cleanup_expired()
            return Output(success=True, count=count)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    ContextPinTool.run()
