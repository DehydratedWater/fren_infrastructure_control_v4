"""Goal Manager — CRUD for 6-level goal hierarchy."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

from app.vllm_resolve import get_llm_endpoint

_base, _model = get_llm_endpoint()
_LLM_API_URL = f"{_base}/chat/completions"
_LLM_MODEL = _model

_MATCHING_QUESTION_PROMPT = """\
Given a goal, generate a YES/NO matching question that can be used to judge whether \
a daily event, activity, or habit completion contributes to this goal.

The question should be broad enough to catch indirect contributions but specific enough \
to avoid false positives. Think about what actions, events, and habits would indicate \
progress toward this goal.

Return ONLY a JSON object:
{
  "matching_question": "Did the user do something that...?",
  "positive_examples": ["went for a walk", "ate a healthy meal"],
  "negative_examples": ["watched TV", "ordered pizza"]
}

The question will be asked with a list of events like:
"walk: Evening walk", "eating: Pizza", "habit: Daily Morning Walk completed"

Keep the question under 30 words. Return ONLY the JSON."""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    if text.startswith("Thinking Process:"):
        text = re.sub(r"^Thinking Process:.*?(?=^#{1,2} )", "", text, flags=re.DOTALL | re.MULTILINE)
    if text.startswith("Thinking Process:"):
        text = ""
    return text.strip()


async def _generate_matching_question(title: str, description: str = "") -> dict:
    """Call local vLLM to generate a matching question for a goal."""
    user_msg = f"Goal: {title}"
    if description:
        user_msg += f"\nDescription: {description}"

    payload = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": _MATCHING_QUESTION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 16384,
        "temperature": 0.3,
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(_LLM_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = _strip_thinking(data["choices"][0]["message"]["content"])
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


class Input(BaseModel):
    command: str = Field(
        description="create|get|list|update|update-status|delete|get-children|get-hierarchy|"
        "generate-tracking-keywords|backfill-keywords"
    )
    goal_id: str = Field(default="", description="Goal ID")
    level: int = Field(default=0, description="Goal level (1-6, 0 for all)")
    title: str = Field(default="", description="Goal title")
    description: str = Field(default="", description="Goal description")
    parent_goal_id: str = Field(default="", description="Parent goal ID")
    priority: str = Field(default="medium", description="Priority level")
    status: str = Field(default="", description="Status for update-status")
    progress_percent: int = Field(default=-1, description="Progress percentage (0-100)")
    deadline: str = Field(default="", description="Deadline (ISO datetime)")


class Output(BaseModel):
    success: bool = True
    goal: dict = Field(default_factory=dict)
    goals: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class GoalManagerTool(ScriptTool[Input, Output]):
    name = "goal_manager"
    description = "Manage goals in a 6-level hierarchy"
    output_note = "If the user requested this, confirm the result via send_message.py"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.goals import GoalsRepo

        repo = GoalsRepo()

        if inp.command == "create":
            gid = f"goal_{datetime.now().strftime('%Y%m%d')}_{id(inp) % 0xFFFFFFFF:08x}"
            # Auto-generate matching question via LLM
            matching = await _generate_matching_question(inp.title, inp.description)
            metadata = {"tracking": matching} if matching else None
            goal = await repo.create(
                goal_id=gid,
                level=inp.level,
                title=inp.title,
                description=inp.description or None,
                parent_goal_id=inp.parent_goal_id or None,
                priority=inp.priority,
                deadline=inp.deadline or None,
                date_str=datetime.now().strftime("%Y-%m-%d"),
                metadata=metadata,
            )
            return Output(success=True, goal=goal)

        if inp.command == "get":
            goal = await repo.get(inp.goal_id)
            if not goal:
                return Output(success=False, error=f"Goal not found: {inp.goal_id}")
            return Output(success=True, goal=goal)

        if inp.command == "list":
            level = inp.level if inp.level > 0 and inp.level <= 6 else None
            goals = await repo.list_active(level=level)
            return Output(success=True, goals=goals, count=len(goals))

        if inp.command == "update":
            fields = {}
            if inp.title:
                fields["title"] = inp.title
            if inp.description:
                fields["description"] = inp.description
            if inp.priority and inp.priority != "medium":
                fields["priority"] = inp.priority
            if inp.progress_percent >= 0:
                fields["progress_percent"] = inp.progress_percent
            if inp.deadline:
                fields["deadline"] = inp.deadline
            goal = await repo.update(inp.goal_id, **fields)
            if not goal:
                return Output(success=False, error=f"Goal not found: {inp.goal_id}")
            return Output(success=True, goal=goal)

        if inp.command == "update-status":
            goal = await repo.update(inp.goal_id, status=inp.status)
            if not goal:
                return Output(success=False, error=f"Goal not found: {inp.goal_id}")
            return Output(success=True, goal=goal)

        if inp.command == "delete":
            ok = await repo.delete(inp.goal_id)
            return Output(success=ok, error="" if ok else f"Goal not found: {inp.goal_id}")

        if inp.command in ("get-children", "get-hierarchy"):
            children = await repo.get_children(inp.goal_id)
            return Output(success=True, goals=children, count=len(children))

        if inp.command == "generate-tracking-keywords":
            goal = await repo.get(inp.goal_id)
            if not goal:
                return Output(success=False, error=f"Goal not found: {inp.goal_id}")
            matching = await _generate_matching_question(goal["title"], goal.get("description") or "")
            if matching:
                metadata = goal.get("metadata") or {}
                metadata["tracking"] = matching
                goal = await repo.update(inp.goal_id, metadata=metadata)
                return Output(success=True, goal=goal or {})
            return Output(success=False, error="LLM failed to generate matching question")

        if inp.command == "backfill-keywords":
            goals = await repo.list_active()
            updated = 0
            for goal in goals:
                meta = goal.get("metadata") or {}
                if meta.get("tracking", {}).get("matching_question"):
                    continue  # Already has one
                matching = await _generate_matching_question(goal["title"], goal.get("description") or "")
                if matching:
                    meta["tracking"] = matching
                    await repo.update(goal["goal_id"], metadata=meta)
                    updated += 1
            return Output(success=True, count=updated, goals=[{"message": f"Backfilled {updated} goals"}])

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    GoalManagerTool.run()
