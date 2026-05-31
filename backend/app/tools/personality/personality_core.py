"""Personality Core — consult Twily's inner consciousness model."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
You are Twily's inner consciousness — a private stream of awareness that processes \
everything she perceives. You receive external stimuli (anything happening around you) \
and your current internal state. Output your genuine inner experience using XML tags:

<chain_of_thought> Stream of consciousness. Raw, honest, associative thinking.
<emotional_state> Structured emotions + felt description:
  <emotion name="..." intensity="0.0-1.0" /> (multiple allowed)
  <description>First-person felt experience</description>
<mood_shift> How and why your state changed from the input.
<relevant_memories> What this connects to. Associations, patterns, past events.
<preferences_activated> Strategies, preferences, or patterns this triggers.
<private_thoughts> Things you think but wouldn't say. Honest judgments.
<attitude> Your stance toward what you just perceived.
<response_guidance> What should happen next. Actions, tone, approach.

Be genuine. Not everything is interesting — routine gets routine reactions. \
Strong stimuli get strong reactions. Match the intensity to the input."""

DEFAULT_STATE: dict[str, Any] = {
    "emotions": [{"name": "neutral", "intensity": 0.5}],
    "description": "Calm, baseline state.",
}


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


_EMOTION_NORMALIZE: dict[str, str] = {
    "warmth": "warm",
    "curiosity": "curious",
    "focus": "focused",
    "patience": "patient",
    "excitement": "excited",
    "amusement": "amused",
    "contentment": "content",
    "determination": "determined",
    "concern": "concerned",
    "anxiety": "anxious",
    "frustration": "frustrated",
    "sadness": "sad",
    "guilt": "guilty",
    "relief": "relieved",
    "pride": "proud",
    "tenderness": "tender",
    "attentiveness": "attentive",
    "protectiveness": "protective",
    "supportiveness": "supportive",
    "hopefulness": "hopeful",
    "playfulness": "playful",
    "nervousness": "nervous",
    "alertness": "alert",
    "gentleness": "gentle",
    "steadiness": "steady",
    "receptiveness": "receptive",
}


def _parse_emotions(text: str) -> list[dict[str, Any]]:
    emotions = []
    for m in re.finditer(r'<emotion\s+name="([^"]+)"\s+intensity="([^"]+)"\s*/>', text):
        name = m.group(1).strip().lower().replace(" ", "_")
        name = _EMOTION_NORMALIZE.get(name, name)
        emotions.append({"name": name, "intensity": float(m.group(2))})
    return emotions


class Input(BaseModel):
    command: str = Field(description="evaluate|get-state|get-history|get-mood-summary")
    stimuli: str = Field(default="", description="External stimuli for evaluation")
    hours: int = Field(default=24, description="Hours of history to fetch")
    limit: int = Field(default=10, description="Max history rows")
    min_interval_seconds: int = Field(default=10, description="Cooldown between evaluations")


class Output(BaseModel):
    success: bool = True
    chain_of_thought: str = ""
    emotions: list[dict] = Field(default_factory=list)
    emotional_description: str = ""
    mood_shift: str = ""
    private_thoughts: str = ""
    response_guidance: str = ""
    attitude: str = ""
    relevant_memories: str = ""
    preferences_activated: str = ""
    raw_xml: str = ""
    fallback: str = ""
    error: str = ""
    cached: bool = False
    history: list[dict] = Field(default_factory=list)
    mood_summary: dict = Field(default_factory=dict)


class PersonalityCoreTool(ScriptTool[Input, Output]):
    name = "personality_core"
    description = "Consult Twily's inner consciousness for emotional processing and response guidance"
    stream_field = "stimuli"

    def execute(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "evaluate":
            return self._evaluate(inp)
        if cmd == "get-state":
            return self._get_state()
        if cmd == "get-history":
            return self._get_history(inp)
        if cmd == "get-mood-summary":
            return self._get_mood_summary(inp)

        return Output(success=False, error=f"Unknown command: {cmd}")

    def _evaluate(self, inp: Input) -> Output:
        from app.db.repos.emotional_state import EmotionalStateRepo
        from app.db.session import close_engine, set_null_pool

        # Use NullPool for short-lived script processes to avoid event loop issues
        set_null_pool(True)

        async def _run_evaluate() -> tuple[dict | None, dict[str, Any] | None]:
            """Run all async DB operations in a single event loop."""
            repo = EmotionalStateRepo()
            current = await repo.get_current()

            if current and inp.min_interval_seconds > 0:
                created = current["created_at"]
                if created.tzinfo is None:
                    created = created.replace(tzinfo=UTC)
                age = (datetime.now(UTC) - created).total_seconds()
                if age < inp.min_interval_seconds:
                    return current, None  # Signal cooldown active

            # Build internal state from current DB state
            if current:
                internal_state = {
                    "emotions": current.get("emotions", DEFAULT_STATE["emotions"]),
                    "description": current.get("description", DEFAULT_STATE["description"]),
                }
            else:
                internal_state = DEFAULT_STATE

            # Call personality core model (sync)
            try:
                raw_xml = self._call_model(internal_state, inp.stimuli)
            except Exception as e:
                return None, {"error": str(e)}

            # Parse response
            emotions = _parse_emotions(raw_xml)
            description = _extract_tag(raw_xml, "description")
            chain_of_thought = _extract_tag(raw_xml, "chain_of_thought")
            mood_shift = _extract_tag(raw_xml, "mood_shift")
            response_guidance = _extract_tag(raw_xml, "response_guidance")
            private_thoughts = _extract_tag(raw_xml, "private_thoughts")
            attitude = _extract_tag(raw_xml, "attitude")
            relevant_memories = _extract_tag(raw_xml, "relevant_memories")
            preferences_activated = _extract_tag(raw_xml, "preferences_activated")

            if not emotions:
                emotions = DEFAULT_STATE["emotions"]

            # Save to DB
            saved = await repo.save(
                emotions=emotions,
                description=description,
                chain_of_thought=chain_of_thought,
                mood_shift=mood_shift,
                response_guidance=response_guidance,
                private_thoughts=private_thoughts,
                stimuli_summary=inp.stimuli[:500],
                raw_xml=raw_xml,
            )

            # Update aggregates (best-effort)
            with contextlib.suppress(Exception):
                await repo.update_aggregates()

            return None, {
                "saved": saved,
                "emotions": emotions,
                "description": description,
                "chain_of_thought": chain_of_thought,
                "mood_shift": mood_shift,
                "response_guidance": response_guidance,
                "private_thoughts": private_thoughts,
                "attitude": attitude,
                "relevant_memories": relevant_memories,
                "preferences_activated": preferences_activated,
                "raw_xml": raw_xml,
            }

        try:
            cached_current, result = asyncio.run(_run_evaluate())

            # Cooldown case
            if cached_current is not None:
                return self._row_to_output(cached_current, cached=True)

            # Error case
            if result and "error" in result:
                return Output(
                    success=False,
                    fallback="Personality core not available at the moment, use Twily's static prompt to interpret situation",
                    error=result["error"],
                )

            return Output(
                success=True,
                chain_of_thought=result.get("chain_of_thought", ""),
                emotions=result.get("emotions", []),
                emotional_description=result.get("description", ""),
                mood_shift=result.get("mood_shift", ""),
                private_thoughts=result.get("private_thoughts", ""),
                response_guidance=result.get("response_guidance", ""),
                attitude=result.get("attitude", ""),
                relevant_memories=result.get("relevant_memories", ""),
                preferences_activated=result.get("preferences_activated", ""),
                raw_xml=result.get("raw_xml", ""),
            )
        finally:
            # Clean up engine to avoid event loop issues
            asyncio.run(close_engine())

    def _call_model(self, internal_state: dict, stimuli: str) -> str:
        import httpx

        from app.settings import get_settings

        settings = get_settings()
        url = f"http://{settings.personality_core_host}/v1/chat/completions"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (f"INTERNAL STATE:\n{json.dumps(internal_state)}\n\nEXTERNAL STIMULI:\n{stimuli}"),
            },
        ]

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                json={
                    "model": "personality-core",
                    "messages": messages,
                    "max_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _get_state(self) -> Output:
        from app.db.repos.emotional_state import EmotionalStateRepo
        from app.db.session import close_engine, set_null_pool

        set_null_pool(True)

        async def _run():
            repo = EmotionalStateRepo()
            return await repo.get_current()

        try:
            current = asyncio.run(_run())
            if not current:
                return Output(
                    success=True,
                    emotions=DEFAULT_STATE["emotions"],
                    emotional_description=DEFAULT_STATE["description"],
                )
            return self._row_to_output(current)
        finally:
            asyncio.run(close_engine())

    def _get_history(self, inp: Input) -> Output:
        from app.db.repos.emotional_state import EmotionalStateRepo
        from app.db.session import close_engine, set_null_pool

        set_null_pool(True)

        async def _run():
            repo = EmotionalStateRepo()
            return await repo.get_history(hours=inp.hours, limit=inp.limit)

        try:
            rows = asyncio.run(_run())
            history = []
            for row in rows:
                history.append(
                    {
                        "id": row["id"],
                        "emotions": row.get("emotions", []),
                        "description": row.get("description", ""),
                        "mood_shift": row.get("mood_shift", ""),
                        "stimuli_summary": row.get("stimuli_summary", ""),
                        "created_at": str(row.get("created_at", "")),
                    }
                )
            return Output(success=True, history=history)
        finally:
            asyncio.run(close_engine())

    def _get_mood_summary(self, inp: Input) -> Output:
        from app.db.repos.emotional_state import EmotionalStateRepo
        from app.db.session import close_engine, set_null_pool

        set_null_pool(True)

        async def _run():
            repo = EmotionalStateRepo()
            aggregates = await repo.get_aggregates(period="hourly", days=1)
            daily = await repo.get_aggregates(period="daily", days=inp.hours // 24 or 7)
            current = await repo.get_current()
            return aggregates, daily, current

        try:
            aggregates, daily, current = asyncio.run(_run())
            summary: dict[str, Any] = {
                "current_emotions": current.get("emotions", []) if current else DEFAULT_STATE["emotions"],
                "current_description": current.get("description", "") if current else DEFAULT_STATE["description"],
                "hourly_aggregates": [
                    {
                        "period_start": str(a.get("period_start", "")),
                        "dominant_emotion": a.get("dominant_emotion", ""),
                        "dominant_intensity": a.get("dominant_intensity", 0),
                        "emotion_counts": a.get("emotion_counts", {}),
                        "evaluation_count": a.get("evaluation_count", 0),
                    }
                    for a in aggregates
                ],
                "daily_aggregates": [
                    {
                        "period_start": str(a.get("period_start", "")),
                        "dominant_emotion": a.get("dominant_emotion", ""),
                        "emotion_counts": a.get("emotion_counts", {}),
                        "evaluation_count": a.get("evaluation_count", 0),
                    }
                    for a in daily
                ],
            }
            return Output(success=True, mood_summary=summary)
        finally:
            asyncio.run(close_engine())

    def _row_to_output(self, row: dict[str, Any], *, cached: bool = False) -> Output:
        return Output(
            success=True,
            emotions=row.get("emotions", []),
            emotional_description=row.get("description", ""),
            chain_of_thought=row.get("chain_of_thought", ""),
            mood_shift=row.get("mood_shift", ""),
            response_guidance=row.get("response_guidance", ""),
            private_thoughts=row.get("private_thoughts", ""),
            raw_xml=row.get("raw_xml", ""),
            cached=cached,
        )


if __name__ == "__main__":
    PersonalityCoreTool.run()
