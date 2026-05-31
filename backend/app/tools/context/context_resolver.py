"""Context Resolver — resolve pronoun/reference in user messages."""

from __future__ import annotations

import asyncio
import re

from src import ScriptTool
from pydantic import BaseModel, Field

REFERENCE_PATTERNS = [
    (r"\bthis\b", "this"),
    (r"\bthat\b", "that"),
    (r"\bit\b", "it"),
    (r"\bthese\b", "these"),
    (r"\bthose\b", "those"),
    (r"\bthe task\b", "the_task"),
    (r"\bthe todo\b", "the_task"),
    (r"\bthe goal\b", "the_goal"),
    (r"\bthe habit\b", "the_habit"),
    (r"\balready done\b", "completion_ref"),
    (r"\bfinished\b", "completion_ref"),
    (r"\bcompleted\b", "completion_ref"),
    (r"\bremove\b", "removal_ref"),
    (r"\bdelete\b", "removal_ref"),
]

TOPIC_PATTERNS = {
    "todo": [r"task", r"todo", r"remind", r"done", r"finish", r"complet"],
    "goal": [r"goal", r"objective", r"aim", r"target"],
    "habit": [r"habit", r"streak", r"daily", r"routine"],
    "server": [r"server", r"disk", r"cpu", r"memory", r"ram"],
}


class Input(BaseModel):
    command: str = Field(default="resolve", description="resolve")
    message: str = Field(description="Message to resolve references in")


class Output(BaseModel):
    success: bool = True
    enriched_context: str = ""
    likely_topic: str = ""
    resolved_items: list[dict] = Field(default_factory=list)
    confidence: float = 0.0
    error: str = ""


class ContextResolverTool(ScriptTool[Input, Output]):
    name = "context_resolver"
    description = "Resolve references like 'this', 'that', 'it' from conversation history"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._resolve(inp.message))

    async def _resolve(self, message: str) -> Output:
        from app.db.repos.chat import ChatMessagesRepo
        from app.db.repos.todos import TodosRepo

        references = self._detect_references(message)
        topic = self._detect_topic(message)

        try:
            chat_repo = ChatMessagesRepo()
            todo_repo = TodosRepo()
            history = await chat_repo.get_recent(limit=20)
            todos = await todo_repo.list(status="pending", limit=20)
        except Exception as e:
            return Output(
                success=True,
                enriched_context=message,
                confidence=0.5,
                error=f"DB unavailable: {e}",
            )

        if not references:
            return Output(success=True, enriched_context=message, confidence=1.0, likely_topic=topic)

        # Match referenced items against todos
        resolved = []
        for msg in history[:10]:
            msg_text = str(msg.get("message", ""))
            for todo in todos:
                title = todo.get("title", "")
                todo_id = todo.get("todo_id", "")
                if title and title.lower() in msg_text.lower():
                    resolved.append({"type": "task", "id": todo_id, "title": title})
                # Check for ID mentions
                if todo_id and todo_id in msg_text:
                    resolved.append({"type": "task", "id": todo_id, "title": title})

        # Deduplicate
        seen = set()
        unique = []
        for item in resolved:
            if item["id"] not in seen:
                seen.add(item["id"])
                unique.append(item)

        # Build enriched context
        parts = [f"Original message: {message}"]
        if unique:
            parts.append("\nRecent context (likely what user is referring to):")
            for item in unique[:5]:
                parts.append(f"  - Task: '{item['title']}' (ID: {item['id']})")
        parts.append("\nRecent conversation:")
        for msg in history[:5]:
            sender = msg.get("sender", "?")
            text = str(msg.get("message", ""))[:200]
            parts.append(f"  [{sender}]: {text}")

        return Output(
            success=True,
            enriched_context="\n".join(parts),
            likely_topic=topic,
            resolved_items=unique,
            confidence=0.8 if unique else 0.5,
        )

    def _detect_references(self, message: str) -> list[str]:
        found = []
        lower = message.lower()
        for pattern, ref_type in REFERENCE_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                found.append(ref_type)
        return found

    def _detect_topic(self, message: str) -> str:
        lower = message.lower()
        for topic, patterns in TOPIC_PATTERNS.items():
            for p in patterns:
                if re.search(p, lower, re.IGNORECASE):
                    return topic
        return ""


if __name__ == "__main__":
    ContextResolverTool.run()
