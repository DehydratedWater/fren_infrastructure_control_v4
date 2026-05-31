"""Emit PersonaGuidance — final delivery channel for in-scope planner agents.

Replaces send_message.py in the agent's bash allow-list. The agent emits a
PersonaGuidance JSON object and this tool delivers it INLINE inside the
agent's bash subprocess (Phase 4 architectural shift). Two paths:

1. message_kind == "ack":  bypass persona_prose, send key_points[0]
   verbatim via send_message.py for sub-second delivery. Writes a
   `persona_guidance_ack` audit artifact. Used by quick_ack and any
   agent that wants to land an immediate "I'm on it" before doing work.

2. message_kind != "ack":  call persona_prose.generate_persona_message
   synchronously (~5-10s LLM call), which renders Twily voice + delivers
   via send_message.py. Returns the delivered text on stdout so the
   agent can read what was actually said. Writes persona_response +
   persona_prose_trace audit artifacts.

Usage (from a compiled agent's bash):
    uv run scripts/emit_guidance.py --data '{"intent":"...","key_points":["..."],"message_kind":"reply"}'

Requires FREN_RUN_ID env var to be set by the caller (opencode_manager,
scheduler, etc.). If missing, the tool synthesizes a run_id and logs a
warning so the call still succeeds for debugging.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

from src import ScriptTool, StreamFormat
from pydantic import BaseModel, Field


class Input(BaseModel):
    # NOTE: ScriptTool builds argparse from field NAMES, not aliases.
    # We avoided `json_` with alias='json' because that produced BOTH
    # `--json` (broken no-value flag) and `--json_ JSON_` in the CLI,
    # so every agent calling `--json '{...}'` failed with "unrecognized
    # arguments". The fix is to use a non-conflicting field name.
    data: str = Field(description="PersonaGuidance JSON object (serialized)")


class Output(BaseModel):
    success: bool = True
    run_id: str = ""
    artifact_id: str = ""
    delivered_text: str = ""
    error: str = ""


class EmitGuidanceTool(ScriptTool[Input, Output]):
    name = "emit_guidance"
    description = (
        "Emit a PersonaGuidance JSON. For message_kind='ack' this delivers a Twily-voiced "
        "one-liner ack instantly (sub-second, no LLM). For all other kinds (reply, briefing, "
        "workflow_result, nudge, selfie_caption, video_caption) this calls persona_prose to "
        "render Twily voice and delivers via send_message.py. Returns the delivered text on "
        "stdout. REPLACES send_message.py — do NOT call send_message.py directly."
    )
    stream_format = StreamFormat.TEXT
    stream_field = "data"
    output_note = (
        "Guidance delivered. The 'delivered_text' field shows what the user actually saw. "
        "Your task is complete — do NOT send additional messages."
    )

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._emit(inp.data))

    async def _emit(self, raw_json: str) -> Output:
        from app.db.repos.execution_ledger import ExecutionLedgerRepo
        from app.telegram.persona_prose import (
            fetch_chat_context,
            generate_persona_message,
            parse_guidance_from_agent_output,
        )

        run_id = os.environ.get("FREN_RUN_ID", "").strip()
        if not run_id:
            run_id = f"orphan_{int(time.time())}"
            print(
                f"[emit_guidance] WARNING: FREN_RUN_ID not set — using {run_id}",
                file=sys.stderr,
            )

        guidance = parse_guidance_from_agent_output(raw_json)

        # ── Ack fast-path: deliver verbatim, no LLM ──
        if guidance.message_kind == "ack":
            return await self._emit_ack(guidance, run_id)

        # ── Full-render path: persona_prose inline ──
        return await self._emit_full(
            guidance, run_id, ExecutionLedgerRepo, fetch_chat_context, generate_persona_message
        )

    async def _emit_ack(self, guidance, run_id: str) -> Output:
        """Deliver an ack message immediately via send_message.py subprocess.

        The agent is responsible for putting a complete 1-sentence Twily-voiced
        ack into key_points[0]. We send it verbatim — no persona_prose call,
        no LLM, sub-second delivery.
        """
        from app.settings import get_settings
        from app.db.repos.execution_ledger import ExecutionLedgerRepo

        # Pick the ack text. Prefer key_points[0]; fall back to joining all
        # key_points with a sep; final fallback is the intent itself.
        if guidance.key_points:
            ack_text = guidance.key_points[0].strip()
            if not ack_text and len(guidance.key_points) > 1:
                ack_text = " — ".join(p.strip() for p in guidance.key_points if p.strip())
        else:
            ack_text = guidance.intent.strip() or "On it~"

        if not ack_text:
            return Output(success=False, run_id=run_id, error="empty ack text")

        # Fire send_message.py as a subprocess (same pattern persona_prose
        # uses internally) so the existing style_scorer + dedup + TTS +
        # chat_messages save pipeline runs.
        settings = get_settings()
        project_root = settings.project_root

        def _fire() -> tuple[int, str]:
            try:
                result = subprocess.run(
                    ["python", "scripts/send_message.py", "--message", ack_text],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                return result.returncode, (result.stderr or "")[:500]
            except Exception as e:
                return 1, str(e)[:500]

        code, err = await asyncio.to_thread(_fire)
        if code != 0:
            return Output(
                success=False,
                run_id=run_id,
                delivered_text=ack_text,
                error=f"send_message.py exit {code}: {err}",
            )

        # Audit write — best-effort, never blocks success return.
        try:
            repo = ExecutionLedgerRepo()
            await repo.ensure_run(run_id, interaction_mode="emit_guidance_ack")
            row = await repo.write_artifact(
                run_id=run_id,
                artifact_type="persona_guidance_ack",
                payload={
                    "guidance": guidance.to_dict(),
                    "delivered_text": ack_text,
                    "delivered": True,
                },
                producer="emit_guidance",
            )
            artifact_id = row.get("artifact_id", "")
        except Exception as e:
            print(f"[emit_guidance] ack audit write failed: {e}", file=sys.stderr)
            artifact_id = ""

        return Output(
            success=True,
            run_id=run_id,
            artifact_id=artifact_id,
            delivered_text=ack_text,
        )

    async def _emit_full(
        self,
        guidance,
        run_id: str,
        ExecutionLedgerRepo,
        fetch_chat_context,
        generate_persona_message,
    ) -> Output:
        """Render guidance through persona_prose inline and deliver."""
        from app.settings import get_settings

        settings = get_settings()
        try:
            chat_id_int = int(settings.chat_id) if settings.chat_id else 0
        except (ValueError, TypeError):
            chat_id_int = 0

        # Read any prior ack delivered for this run_id so persona_prose can
        # build on it without repeating (Fix 5 — continuation).
        prior_ack_text = ""
        try:
            repo = ExecutionLedgerRepo()
            ack_art = await repo.read_artifact(
                run_id=run_id,
                artifact_type="persona_guidance_ack",
                consumer="emit_guidance_full",
            )
            if ack_art:
                payload = ack_art.get("payload") or {}
                if isinstance(payload, dict):
                    prior_ack_text = str(payload.get("delivered_text") or "")
        except Exception as e:
            print(f"[emit_guidance] prior_ack lookup failed (non-fatal): {e}", file=sys.stderr)

        # Also write the source guidance to the ledger as persona_guidance
        # for retrospective inspection (separate from the trace artifact
        # that generate_persona_message writes).
        try:
            repo = ExecutionLedgerRepo()
            await repo.ensure_run(run_id, interaction_mode="emit_guidance")
            await repo.write_artifact(
                run_id=run_id,
                artifact_type="persona_guidance",
                payload=guidance.to_dict(),
                producer="emit_guidance",
            )
        except Exception as e:
            print(f"[emit_guidance] guidance audit write failed: {e}", file=sys.stderr)

        # Fetch context + run persona_prose synchronously.
        try:
            ctx = await fetch_chat_context(chat_id=chat_id_int)
            trace = await generate_persona_message(
                guidance,
                ctx,
                prior_ack_text=prior_ack_text,
                run_id=run_id,
            )
        except Exception as e:
            return Output(
                success=False,
                run_id=run_id,
                error=f"persona_prose generation failed: {e}",
            )

        delivered_text = trace.get("delivered_text", "") if isinstance(trace, dict) else ""

        return Output(
            success=True,
            run_id=run_id,
            artifact_id="",  # persona_response artifact id available in trace dict if needed
            delivered_text=delivered_text,
        )
