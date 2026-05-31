"""Response processor tool — detect task completions and acknowledgments."""

from __future__ import annotations

import asyncio
import hashlib
import re

from src import ScriptTool
from pydantic import BaseModel, Field

# ── Detection patterns ──

COMPLETION_PATTERNS = [
    r"\b(i|ive|i've|just)\s+(did|done|finished|completed|ate|eaten|had)\b",
    r"\b(done|finished|completed)\s+(with|it|that)\b",
    r"\b(already|just)\s+(ate|eaten|had|did|done)\b",
    r"\bfinished\s+\w+",
    r"\bjust\s+finished\b",
    r"\ball\s+done\b",
    r"\bchecked\s+off\b",
    r"\bmarked?\s+(as\s+)?done\b",
    r"\bgot\s+it\s+done\b",
    r"\btook\s+care\s+of\b",
    r"\bhandled\s+(it|that)\b",
]

TOPIC_PATTERNS: dict[str, list[str]] = {
    "lunch": [r"\b(lunch|ate|eaten|had\s+food|eating)\b"],
    "breakfast": [r"\b(breakfast)\b"],
    "dinner": [r"\b(dinner|supper)\b"],
    "exercise": [r"\b(workout|exercise|gym|run|running|walked|walk)\b"],
    "meeting": [r"\b(meeting|call|conference|zoom|standup)\b"],
    "email": [r"\b(email|emails|replied|responded)\b"],
    "report": [r"\b(report|document|doc|paper)\b"],
    "coding": [r"\b(code|coding|programming|pr|pull\s+request|commit)\b"],
    "reading": [r"\b(reading|read|book|article)\b"],
    "studying": [r"\b(study|studying|learned|learning)\b"],
}

ACKNOWLEDGMENT_PATTERNS = [
    r"\b(i\s+know|aware|got\s+it|understood|will\s+do|on\s+it|working\s+on)\b",
    r"\b(thanks|thank\s+you|ok|okay|alright|yes|yep|yeah|sure)\b",
    r"\b(noted|understood|roger|copy\s+that)\b",
]

STOPWORDS = {
    "i",
    "me",
    "my",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "for",
    "on",
    "at",
    "in",
    "of",
    "and",
    "or",
    "but",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "done",
    "just",
    "with",
    "from",
    "by",
    "it",
    "that",
    "this",
    "be",
    "been",
    "ive",
    "already",
    "finished",
    "completed",
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _detect_completion(message: str) -> bool:
    normalized = _normalize(message)
    return any(re.search(p, normalized) for p in COMPLETION_PATTERNS)


def _detect_topics(message: str) -> list[str]:
    normalized = _normalize(message)
    topics = []
    for topic, patterns in TOPIC_PATTERNS.items():
        if any(re.search(p, normalized) for p in patterns):
            topics.append(topic)
    return topics


def _detect_acknowledgment(message: str) -> bool:
    normalized = _normalize(message)
    return any(re.search(p, normalized) for p in ACKNOWLEDGMENT_PATTERNS)


_STEM_MAP = {
    "walked": "walk",
    "walking": "walk",
    "walks": "walk",
    "ran": "run",
    "running": "run",
    "runs": "run",
    "exercised": "exercise",
    "exercising": "exercise",
    "meditated": "meditate",
    "meditating": "meditate",
    "ate": "eat",
    "eaten": "eat",
    "eating": "eat",
    "drank": "drink",
    "drinking": "drink",
    "studied": "study",
    "studying": "study",
    "cooked": "cook",
    "cooking": "cook",
    "cleaned": "clean",
    "cleaning": "clean",
    "stretched": "stretch",
    "stretching": "stretch",
    "swam": "swim",
    "swimming": "swim",
    "cycled": "cycle",
    "cycling": "cycle",
    "lifted": "lift",
    "lifting": "lift",
    "returned": "return",
    "returning": "return",
    "finished": "finish",
    "finishing": "finish",
    "completed": "complete",
    "completing": "complete",
    "bought": "buy",
    "buying": "buy",
    "paid": "pay",
    "paying": "pay",
    "sent": "send",
    "sending": "send",
}


def _extract_keywords(text: str) -> set[str]:
    words = {w for w in _normalize(text).split() if len(w) > 2 and w not in STOPWORDS}
    return {_STEM_MAP.get(w, w) for w in words}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class Input(BaseModel):
    command: str = Field(description="process-message|analyze-message|get-question-hash")
    message: str = Field(default="", description="Message text to process")
    text: str = Field(default="", description="Text for question hash")
    timestamp: str = Field(default="", description="Optional timestamp")
    auto_complete: bool = Field(default=False, description="Auto-complete matching todos")


class Output(BaseModel):
    success: bool = True
    is_completion: bool = False
    is_acknowledgment: bool = False
    detected_topics: list[str] = Field(default_factory=list)
    matching_todos: list[dict] = Field(default_factory=list)
    matching_habits: list[dict] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    hash: str = ""
    error: str = ""


class ResponseProcessorTool(ScriptTool[Input, Output]):
    name = "response_processor"
    description = "Process user messages to detect task completions and acknowledgments"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        if inp.command == "process-message":
            return await self._process(inp.message, inp.auto_complete)
        if inp.command == "analyze-message":
            return await self._analyze(inp.message)
        if inp.command == "get-question-hash":
            normalized = _normalize(inp.text)
            normalized = re.sub(r"~\s*twily\s*$", "", normalized)[:50]
            h = hashlib.md5(normalized.encode()).hexdigest()[:8]
            return Output(success=True, hash=h)
        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _process(self, message: str, auto_complete: bool) -> Output:
        from app.db.repos.agent_notes import AgentNotesRepo
        from app.db.repos.habits import HabitsRepo
        from app.db.repos.todos import TodosRepo

        actions: list[str] = []
        topics: list[str] = []
        matching: list[dict] = []
        matching_habits: list[dict] = []

        if _detect_completion(message):
            topics = _detect_topics(message)
            repo = AgentNotesRepo()
            for topic in topics:
                await repo.set(f"user_said:{topic}", {"message": message}, expires_hours=4)
                actions.append(f"set note user_said:{topic}")

            msg_kw = _extract_keywords(message)
            if msg_kw:
                # Match todos
                todos = await TodosRepo().list(status="pending", limit=100)
                for todo in todos:
                    title = todo.get("title", "")
                    desc = todo.get("description", "")
                    sim = _jaccard(msg_kw, _extract_keywords(f"{title} {desc}"))
                    if sim >= 0.3:
                        matching.append(
                            {
                                "todo_id": todo.get("todo_id", ""),
                                "title": title,
                                "similarity": round(sim, 2),
                            }
                        )
                matching.sort(key=lambda x: x["similarity"], reverse=True)
                matching = matching[:3]

                if auto_complete:
                    for m in matching:
                        if m["similarity"] >= 0.5:
                            await TodosRepo().update(m["todo_id"], status="completed")
                            actions.append(f"completed todo {m['todo_id']}")

                # Match habits — check today's pending occurrences
                habits_repo = HabitsRepo()
                due_today = await habits_repo.get_due_today()
                for occ in due_today:
                    habit_title = occ.get("habit_title", "")
                    sim = _jaccard(msg_kw, _extract_keywords(habit_title))
                    if sim >= 0.3:
                        matching_habits.append(
                            {
                                "occurrence_id": occ.get("occurrence_id", ""),
                                "habit_id": occ.get("habit_id", ""),
                                "title": habit_title,
                                "similarity": round(sim, 2),
                            }
                        )
                matching_habits.sort(key=lambda x: x["similarity"], reverse=True)
                matching_habits = matching_habits[:3]

                if auto_complete:
                    for m in matching_habits:
                        if m["similarity"] >= 0.4:
                            await habits_repo.complete_occurrence(
                                m["occurrence_id"], notes=f"Auto-completed: {message}"
                            )
                            actions.append(f"completed habit occurrence {m['occurrence_id']} ({m['title']})")

        is_ack = _detect_acknowledgment(message)
        if is_ack:
            await AgentNotesRepo().set("last_acknowledgment", {"message": message}, expires_hours=4)
            actions.append("set note last_acknowledgment")

        return Output(
            success=True,
            is_completion=_detect_completion(message),
            is_acknowledgment=is_ack,
            detected_topics=topics,
            matching_todos=matching,
            matching_habits=matching_habits,
            actions_taken=actions,
        )

    async def _analyze(self, message: str) -> Output:
        from app.db.repos.todos import TodosRepo

        msg_kw = _extract_keywords(message)
        matching: list[dict] = []
        if msg_kw:
            todos = await TodosRepo().list(status="pending", limit=100)
            for todo in todos:
                title = todo.get("title", "")
                desc = todo.get("description", "")
                sim = _jaccard(msg_kw, _extract_keywords(f"{title} {desc}"))
                if sim >= 0.3:
                    matching.append(
                        {
                            "todo_id": todo.get("todo_id", ""),
                            "title": title,
                            "similarity": round(sim, 2),
                        }
                    )
            matching.sort(key=lambda x: x["similarity"], reverse=True)
            matching = matching[:5]

        return Output(
            success=True,
            is_completion=_detect_completion(message),
            is_acknowledgment=_detect_acknowledgment(message),
            detected_topics=_detect_topics(message),
            keywords=sorted(msg_kw),
            matching_todos=matching,
        )


if __name__ == "__main__":
    ResponseProcessorTool.run()
