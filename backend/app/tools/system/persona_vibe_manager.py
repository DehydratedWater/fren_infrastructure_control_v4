"""Persona vibe manager — CRUD for vibe_state + style_events.

Exposes the palette blend and scorer audit log to agents via a single script tool.
Used by orchestrator (to inject blend into synthesizer), vibe_drift (to update),
and the dashboard (for read-only display).
"""

from __future__ import annotations

import asyncio
import json

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="get-vibe|set-weights|drift|reset-vibe|directives|list-events|count-violations|history"
    )
    chat_id: int = Field(default=0, description="Telegram chat_id (required for all commands)")

    # set-weights / drift
    w_warm_snarky: float = Field(default=-1.0, description="New weight (set-weights) or delta (drift)")
    w_dry_ironic: float = Field(default=-1.0, description="New weight or delta")
    w_caring_edge: float = Field(default=-1.0, description="New weight or delta")
    w_playful_flirt: float = Field(default=-1.0, description="New weight or delta")
    w_debate_socratic: float = Field(default=-1.0, description="New weight or delta")
    axis_delta: float = Field(default=0.0, description="drift: ironic_genuine_axis delta (-1..+1)")
    arousal_delta: float = Field(default=0.0, description="drift: arousal_axis delta (-1..+1)")
    ema: float = Field(default=0.75, description="drift: EMA coefficient (0..1)")
    trigger: str = Field(default="manual", description="drift: trigger label")
    user_tone: str = Field(default="", description="drift: classified user tone")

    # list-events / count-violations
    violation_type: str = Field(default="", description="list-events: filter by type")
    since_hours: int = Field(default=24, description="count-violations: time window")
    limit: int = Field(default=50, description="list-events: row limit")


class Output(BaseModel):
    success: bool = True
    state: dict | None = None
    directives: str = ""
    events: list[dict] = Field(default_factory=list)
    counts: list[dict] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    error: str = ""


class PersonaVibeManagerTool(ScriptTool[Input, Output]):
    name = "persona_vibe_manager"
    description = "Manage Twily's palette-blend vibe state and rule-scorer audit log"
    output_note = "Dominant palette and its markers are inside directives."

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.persona_vibe import StyleEventsRepo, VibeStateRepo
        from app.services.persona_palettes import blend_directives

        cmd = inp.command
        chat_id = inp.chat_id
        if not chat_id:
            # Fall back to settings.chat_id (Vis's default chat).
            try:
                from app.settings import get_settings

                raw = get_settings().chat_id
                chat_id = int(raw) if raw else 0
            except Exception:
                chat_id = 0
        if not chat_id and cmd != "directives":
            return Output(success=False, error="chat_id is required (no default configured)")
        # Reassign chat_id on the input so downstream calls use the resolved value.
        inp.chat_id = chat_id

        if cmd == "get-vibe":
            state = await VibeStateRepo().get(inp.chat_id)
            weights = {
                "w_warm_snarky": float(state["w_warm_snarky"]),
                "w_dry_ironic": float(state["w_dry_ironic"]),
                "w_caring_edge": float(state["w_caring_edge"]),
                "w_playful_flirt": float(state["w_playful_flirt"]),
                "w_debate_socratic": float(state["w_debate_socratic"]),
            }
            directives = blend_directives(
                weights,
                axis=float(state["ironic_genuine_axis"]),
                arousal=float(state.get("arousal_axis") or 0.0),
            )
            return Output(success=True, state=_serialize(state), directives=directives)

        if cmd == "directives":
            # Useful for cheap prompt assembly without another DB hit.
            state = await VibeStateRepo().get(inp.chat_id)
            weights = {
                "w_warm_snarky": float(state["w_warm_snarky"]),
                "w_dry_ironic": float(state["w_dry_ironic"]),
                "w_caring_edge": float(state["w_caring_edge"]),
                "w_playful_flirt": float(state["w_playful_flirt"]),
                "w_debate_socratic": float(state["w_debate_socratic"]),
            }
            return Output(
                success=True,
                directives=blend_directives(weights, axis=float(state["ironic_genuine_axis"])),
            )

        if cmd == "set-weights":
            weights: dict[str, float] = {}
            for k in ("w_warm_snarky", "w_dry_ironic", "w_caring_edge", "w_playful_flirt", "w_debate_socratic"):
                v = getattr(inp, k)
                if v >= 0:
                    weights[k] = v
            if not weights:
                return Output(success=False, error="No weights supplied")
            # Fill missing with current values.
            current = await VibeStateRepo().get(inp.chat_id)
            for k in ("w_warm_snarky", "w_dry_ironic", "w_caring_edge", "w_playful_flirt", "w_debate_socratic"):
                if k not in weights:
                    weights[k] = float(current[k])
            state = await VibeStateRepo().set_weights(inp.chat_id, weights)
            return Output(success=True, state=_serialize(state))

        if cmd == "drift":
            delta: dict[str, float] = {}
            for k in ("w_warm_snarky", "w_dry_ironic", "w_caring_edge", "w_playful_flirt", "w_debate_socratic"):
                v = getattr(inp, k)
                if v != -1.0:  # sentinel: -1 means "not supplied"
                    delta[k] = v
            if not delta and inp.axis_delta == 0.0 and inp.arousal_delta == 0.0:
                return Output(success=False, error="No drift delta supplied")
            state = await VibeStateRepo().drift(
                inp.chat_id,
                delta,
                trigger=inp.trigger,
                user_tone=inp.user_tone,
                ema=inp.ema,
                axis_delta=inp.axis_delta,
                arousal_delta=inp.arousal_delta,
            )
            return Output(success=True, state=_serialize(state))

        if cmd == "reset-vibe":
            state = await VibeStateRepo().reset(inp.chat_id)
            return Output(success=True, state=_serialize(state))

        if cmd == "list-events":
            events = await StyleEventsRepo().list_recent(
                inp.chat_id,
                limit=inp.limit,
                violation_type=inp.violation_type or None,
            )
            return Output(success=True, events=[_serialize(e) for e in events])

        if cmd == "history":
            rows = await VibeStateRepo().history(inp.chat_id, limit=inp.limit or 100)
            return Output(success=True, history=[_serialize(r) for r in rows])

        if cmd == "count-violations":
            counts = await StyleEventsRepo().count_by_type(inp.chat_id, since_hours=inp.since_hours)
            return Output(success=True, counts=[_serialize(c) for c in counts])

        return Output(success=False, error=f"Unknown command: {cmd}")


def _serialize(row: dict | None) -> dict:
    """Coerce datetime/decimal fields for JSON output."""
    if not row:
        return {}
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif isinstance(v, dict | list):
            out[k] = v
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


if __name__ == "__main__":
    PersonaVibeManagerTool.run()
