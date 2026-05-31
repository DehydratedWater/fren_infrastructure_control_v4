"""Council of Personas — multi-perspective analysis engine.

Loads persona definitions from config/council_personas.yml, runs N parallel LLM
calls (one per enabled persona), then synthesises a combined verdict with ranked
action items.

Also provides CRUD commands for managing personas via the dashboard.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
import yaml
from src import ScriptTool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = PROJECT_ROOT / "config" / "council_personas.yml"

# ---------------------------------------------------------------------------
# LLM helpers (same pattern as night_analyst / vibe_drift)
# ---------------------------------------------------------------------------

_LLM_API_URL: str | None = None
_LLM_MODEL: str | None = None


def _resolve_llm(role: str = "analytical") -> tuple[str, str]:
    global _LLM_API_URL, _LLM_MODEL
    if _LLM_API_URL is None:
        from app.vllm_resolve import get_llm_endpoint

        base_url, model = get_llm_endpoint(role)
        _LLM_API_URL = f"{base_url}/chat/completions"
        _LLM_MODEL = model
    return _LLM_API_URL, _LLM_MODEL  # type: ignore[return-value]


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    if text.lstrip().startswith("Thinking") and "{" in text:
        text = text[text.index("{") :]
    return text.strip()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"settings": {}, "personas": []}
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {"settings": {}, "personas": []}


def _save_config(cfg: dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)


def _find_persona(personas: list[dict], persona_id: str) -> dict | None:
    return next((p for p in personas if p.get("id") == persona_id), None)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Input(BaseModel):
    command: str = Field(
        description="run|list-personas|get-persona|add-persona|update-persona|remove-persona|toggle-persona"
    )
    # run
    subject: str = Field(default="", description="What to analyse (decision, project, plan)")
    context: str = Field(default="", description="Supporting context (goals, todos, recent activity)")
    persona_ids: str = Field(default="", description="Comma-separated persona IDs to include, or empty for all enabled")
    # persona CRUD
    persona_id: str = Field(default="", description="Persona ID for get/update/remove/toggle")
    persona_name: str = Field(default="", description="Display name for add/update")
    persona_prompt: str = Field(default="", description="System prompt for add/update")
    focus_areas: str = Field(default="", description="Comma-separated focus areas for add/update")
    enabled: bool = Field(default=True, description="Enabled flag for add")


class Output(BaseModel):
    success: bool = True
    command: str = ""
    verdicts: list[dict] = Field(default_factory=list)
    synthesis: str = ""
    personas: list[dict] = Field(default_factory=list)
    persona: dict = Field(default_factory=dict)
    message: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

PERSONA_USER_TEMPLATE = """\
## Subject Under Review
{subject}

## Context
{context}

## Your Task
1. Identify 2-4 specific weak points, blind spots, or missed opportunities from YOUR unique perspective.
2. For each point: explain WHY it matters and WHAT specifically should be done about it.
3. Rate each point: CRITICAL / IMPORTANT / NICE-TO-HAVE
4. End with your single most important recommendation.

Be brutally honest. Be specific. Name names. Give actionable advice, not platitudes.
"""

SYNTHESIS_SYSTEM = """\
You are the Council Synthesis Engine. You have received independent analyses from \
{n} expert personas, each reviewing the same subject from their unique perspective. \
Your job is to produce a clear, actionable verdict.

Structure your output as follows:

**CONSENSUS** — What did multiple personas agree on? These are the biggest blind spots.

**CONFLICTS** — Where did personas disagree? Present both sides briefly.

**TOP 5 ACTIONS** — Priority-ranked, concrete, with a clear next step for each. \
Number them 1-5.

**THE VERDICT** — In 2-3 sentences, the blunt truth: what is being fucked up and \
what is the single most impactful change to make right now.

Format for Telegram (markdown). Be direct, no fluff.
"""


async def _call_persona(
    sem: asyncio.Semaphore,
    persona: dict,
    subject: str,
    context: str,
    settings: dict,
) -> dict:
    """Run a single persona LLM call."""
    api_url, model = _resolve_llm(settings.get("model_role", "analytical"))
    temperature = settings.get("persona_temperature", 0.7)
    max_tokens = settings.get("max_tokens", 8192)

    user_prompt = PERSONA_USER_TEMPLATE.format(subject=subject, context=context)

    async with sem:
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=240) as client:
                    resp = await client.post(
                        api_url,
                        json={
                            "model": model,
                            "messages": [
                                {"role": "system", "content": persona["system_prompt"]},
                                {"role": "user", "content": user_prompt},
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    resp.raise_for_status()
                    content = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
                    if content:
                        return {
                            "persona_id": persona["id"],
                            "name": persona["name"],
                            "analysis": content,
                            "focus_areas": persona.get("focus_areas", []),
                        }
                    if attempt == 0:
                        await asyncio.sleep(5)
                        continue
            except Exception as e:
                if attempt == 0:
                    print(f"[council] {persona['id']} failed: {e}, retrying...")
                    await asyncio.sleep(5)
                    continue
                print(f"[council] {persona['id']} failed after retry: {e}")
                return {
                    "persona_id": persona["id"],
                    "name": persona["name"],
                    "analysis": f"(Analysis unavailable: {e})",
                    "focus_areas": persona.get("focus_areas", []),
                }
    return {
        "persona_id": persona["id"],
        "name": persona["name"],
        "analysis": "(Analysis unavailable: empty response)",
        "focus_areas": persona.get("focus_areas", []),
    }


async def _synthesise(verdicts: list[dict], settings: dict) -> str:
    """Combine all persona verdicts into a single synthesis."""
    api_url, model = _resolve_llm(settings.get("model_role", "analytical"))
    temperature = settings.get("synthesis_temperature", 0.5)
    max_tokens = settings.get("max_tokens", 8192)

    parts = []
    for v in verdicts:
        parts.append(f"### {v['name']}\n{v['analysis']}")
    all_verdicts = "\n\n---\n\n".join(parts)

    system = SYNTHESIS_SYSTEM.format(n=len(verdicts))

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                resp = await client.post(
                    api_url,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": all_verdicts},
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                resp.raise_for_status()
                content = _strip_thinking(resp.json()["choices"][0]["message"]["content"])
                if content:
                    return content
                if attempt == 0:
                    await asyncio.sleep(5)
                    continue
        except Exception as e:
            if attempt == 0:
                print(f"[council] synthesis failed: {e}, retrying...")
                await asyncio.sleep(5)
                continue
            return f"(Synthesis unavailable: {e})"
    return "(Synthesis unavailable: empty response)"


# ---------------------------------------------------------------------------
# ScriptTool
# ---------------------------------------------------------------------------


class CouncilTool(ScriptTool[Input, Output]):
    name = "council"
    description = "Run the Council of Personas — multi-perspective analysis of decisions and work"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "list-personas":
            return self._list_personas()
        if cmd == "get-persona":
            return self._get_persona(inp.persona_id)
        if cmd == "add-persona":
            return self._add_persona(inp)
        if cmd == "update-persona":
            return self._update_persona(inp)
        if cmd == "remove-persona":
            return self._remove_persona(inp.persona_id)
        if cmd == "toggle-persona":
            return self._toggle_persona(inp.persona_id)
        if cmd == "run":
            return await self._run_council(inp)

        return Output(success=False, command=cmd, error=f"Unknown command: {cmd}")

    # -- CRUD (synchronous, YAML-backed) ------------------------------------

    def _list_personas(self) -> Output:
        cfg = _load_config()
        personas = cfg.get("personas", [])
        # Strip system_prompt from list to keep output compact
        summaries = []
        for p in personas:
            summaries.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name"),
                    "enabled": p.get("enabled", True),
                    "focus_areas": p.get("focus_areas", []),
                }
            )
        return Output(success=True, command="list-personas", personas=summaries)

    def _get_persona(self, persona_id: str) -> Output:
        if not persona_id:
            return Output(success=False, command="get-persona", error="persona_id is required")
        cfg = _load_config()
        p = _find_persona(cfg.get("personas", []), persona_id)
        if not p:
            return Output(success=False, command="get-persona", error=f"Persona '{persona_id}' not found")
        return Output(success=True, command="get-persona", persona=p)

    def _add_persona(self, inp: Input) -> Output:
        if not inp.persona_id or not inp.persona_name or not inp.persona_prompt:
            return Output(
                success=False,
                command="add-persona",
                error="persona_id, persona_name, and persona_prompt are required",
            )
        cfg = _load_config()
        personas = cfg.setdefault("personas", [])
        if _find_persona(personas, inp.persona_id):
            return Output(success=False, command="add-persona", error=f"Persona '{inp.persona_id}' already exists")
        new_persona: dict[str, Any] = {
            "id": inp.persona_id,
            "name": inp.persona_name,
            "enabled": inp.enabled,
            "system_prompt": inp.persona_prompt,
            "focus_areas": [a.strip() for a in inp.focus_areas.split(",") if a.strip()] if inp.focus_areas else [],
        }
        personas.append(new_persona)
        _save_config(cfg)
        return Output(
            success=True, command="add-persona", persona=new_persona, message=f"Added persona '{inp.persona_id}'"
        )

    def _update_persona(self, inp: Input) -> Output:
        if not inp.persona_id:
            return Output(success=False, command="update-persona", error="persona_id is required")
        cfg = _load_config()
        p = _find_persona(cfg.get("personas", []), inp.persona_id)
        if not p:
            return Output(success=False, command="update-persona", error=f"Persona '{inp.persona_id}' not found")
        if inp.persona_name:
            p["name"] = inp.persona_name
        if inp.persona_prompt:
            p["system_prompt"] = inp.persona_prompt
        if inp.focus_areas:
            p["focus_areas"] = [a.strip() for a in inp.focus_areas.split(",") if a.strip()]
        _save_config(cfg)
        return Output(success=True, command="update-persona", persona=p, message=f"Updated persona '{inp.persona_id}'")

    def _remove_persona(self, persona_id: str) -> Output:
        if not persona_id:
            return Output(success=False, command="remove-persona", error="persona_id is required")
        cfg = _load_config()
        personas = cfg.get("personas", [])
        original_len = len(personas)
        cfg["personas"] = [p for p in personas if p.get("id") != persona_id]
        if len(cfg["personas"]) == original_len:
            return Output(success=False, command="remove-persona", error=f"Persona '{persona_id}' not found")
        _save_config(cfg)
        return Output(success=True, command="remove-persona", message=f"Removed persona '{persona_id}'")

    def _toggle_persona(self, persona_id: str) -> Output:
        if not persona_id:
            return Output(success=False, command="toggle-persona", error="persona_id is required")
        cfg = _load_config()
        p = _find_persona(cfg.get("personas", []), persona_id)
        if not p:
            return Output(success=False, command="toggle-persona", error=f"Persona '{persona_id}' not found")
        p["enabled"] = not p.get("enabled", True)
        _save_config(cfg)
        status = "enabled" if p["enabled"] else "disabled"
        return Output(
            success=True, command="toggle-persona", persona=p, message=f"Persona '{persona_id}' is now {status}"
        )

    # -- Council execution --------------------------------------------------

    async def _run_council(self, inp: Input) -> Output:
        if not inp.subject:
            return Output(success=False, command="run", error="subject is required")

        cfg = _load_config()
        settings = cfg.get("settings", {})
        personas = cfg.get("personas", [])

        # Filter to enabled or specific IDs
        if inp.persona_ids:
            ids = {i.strip() for i in inp.persona_ids.split(",") if i.strip()}
            active = [p for p in personas if p.get("id") in ids]
        else:
            active = [p for p in personas if p.get("enabled", True)]

        if not active:
            return Output(success=False, command="run", error="No active personas to run")

        max_concurrent = settings.get("max_concurrent", 3)
        sem = asyncio.Semaphore(max_concurrent)

        print(f"[council] Running {len(active)} personas (max_concurrent={max_concurrent})...")

        # Run all persona calls in parallel
        tasks = [_call_persona(sem, p, inp.subject, inp.context, settings) for p in active]
        verdicts = await asyncio.gather(*tasks)

        print(f"[council] All {len(verdicts)} persona analyses complete. Synthesising...")

        # Synthesise
        synthesis = await _synthesise(list(verdicts), settings)

        # Serialise for JSON output
        clean_verdicts = []
        for v in verdicts:
            clean_verdicts.append(
                {
                    "persona_id": v["persona_id"],
                    "name": v["name"],
                    "analysis": v["analysis"],
                    "focus_areas": v.get("focus_areas", []),
                }
            )

        return Output(
            success=True,
            command="run",
            verdicts=clean_verdicts,
            synthesis=synthesis,
            message=f"Council complete: {len(clean_verdicts)} personas analysed",
        )
