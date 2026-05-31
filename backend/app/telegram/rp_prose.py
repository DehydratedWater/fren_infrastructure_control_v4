"""RP Prose writer — direct API call for roleplay prose generation.

This module sits between rp_handlers.handle_message and whichever chat-completions
provider is configured. It exists so the RP bot can:

1. Generate prose with a model chosen independently from the main bot (even an
   entirely different provider) without touching `_config.py` or agent compilation.
2. Feed chat history to the model as proper OpenAI `messages[]` with correct roles
   (player → user, narration/dialogue → assistant), instead of dumping story log
   as plain text inside an agent system prompt.
3. Layer in progressive summaries + recall pins for effectively unbounded memory.
4. Keep the writer prompt verbatim (loaded from writer_core.md) so it survives
   agent compilation and any tooling that might reformat opencode agent files.

The orchestrator (`rp/game_master`) still runs via opencode as a detached
post-turn job to update state and dispatch illustrations — but it no longer
writes the prose.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OPENCODE_JSON = _PROJECT_ROOT / "opencode.json"
_WRITER_CORE_MD = _PROJECT_ROOT / "src" / "fren" / "agents" / "rp" / "prompts" / "writer_core.md"


# ── Config loading ──────────────────────────────────────────────────────────


def _expand_env(value: str) -> str:
    """Expand an `env:VAR_NAME` reference into the actual environment value.

    Checks `os.environ` first (docker/compose exports), then falls back to the
    matching pydantic setting (e.g. `env:VLLM_API_KEY` → `settings.vllm_api_key`)
    so ad-hoc `uv run` invocations that only load `.env` into pydantic still work.
    """
    if not value.startswith("env:"):
        return value
    var_name = value[4:]
    # 1. True process env (populated by `docker compose --env-file` etc.)
    from_env = os.environ.get(var_name)
    if from_env:
        return from_env
    # 2. Pydantic-settings fallback (`.env` may only be loaded into Settings)
    try:
        from app.settings import get_settings

        attr = var_name.lower()
        val = getattr(get_settings(), attr, "")
        if isinstance(val, str) and val:
            return val
    except Exception:
        pass
    return ""


@lru_cache(maxsize=1)
def _load_opencode_json() -> dict[str, Any]:
    try:
        return json.loads(_OPENCODE_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Failed to read opencode.json: %s", e)
        return {"provider": {}}


def _read_user_opencode_apikey(provider: str) -> str:
    """Look up the apiKey for a provider in the user-level opencode config.

    Opencode merges ~/.config/opencode/opencode.json with the project config;
    for builtin providers like zai-coding-plan we deliberately leave the project
    options empty so opencode's merge uses the user-level apiKey. For the RP
    bot's direct-API path we replicate that lookup in Python.
    """
    try:
        user_config = Path.home() / ".config" / "opencode" / "opencode.json"
        data = json.loads(user_config.read_text(encoding="utf-8"))
        prov = (data.get("provider") or {}).get(provider) or {}
        options = prov.get("options") or {}
        key = options.get("apiKey") or ""
        if isinstance(key, str):
            return _expand_env(key)
        return ""
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return ""


def load_model_config(
    role: Literal["prose", "orchestrator"],
    *,
    override_provider: str | None = None,
    override_model: str | None = None,
) -> tuple[str, str, str]:
    """Resolve (base_url, api_key, model_id) for the given RP role.

    Reads settings.rp_{role}_provider + settings.rp_{role}_model, looks them up
    in opencode.json, and expands `env:` references for the API key.

    If `override_provider`/`override_model` are supplied (e.g. from an active
    adventure's DB columns set via /model), they take precedence over settings.
    """
    from app.settings import get_settings

    settings = get_settings()
    if role == "prose":
        provider_key = override_provider or settings.rp_prose_provider
        model_key = override_model or settings.rp_prose_model
    else:
        provider_key = override_provider or settings.rp_orchestrator_provider
        model_key = override_model or settings.rp_orchestrator_model

    data = _load_opencode_json()
    providers = data.get("provider", {})
    prov = providers.get(provider_key) or {}
    options = prov.get("options") or {}

    base_url = options.get("baseURL", "")
    api_key_raw = options.get("apiKey", "")
    api_key = _expand_env(api_key_raw) if isinstance(api_key_raw, str) else ""

    # Provider-specific fallbacks when the project opencode.json leaves options
    # blank (e.g. zai-coding-plan, whose apiKey lives in the user-level config
    # so opencode's builtin provider can handle auth without our interference).
    if provider_key == "zai-coding-plan" and (not api_key or not base_url):
        from app.settings import get_settings as _gs

        if not api_key:
            api_key = _gs().zai_api_key or _read_user_opencode_apikey("zai-coding-plan")
        if not base_url:
            # The coding-plan endpoint accepts bare apiKey as Bearer.
            base_url = "https://api.z.ai/api/coding/paas/v4"

    models = prov.get("models") or {}
    model_entry = models.get(model_key) or {}
    model_id = model_entry.get("id") or model_key

    if not base_url:
        logger.warning(
            "RP %s: provider %r has no baseURL in opencode.json — prose call will likely fail",
            role,
            provider_key,
        )
    return base_url, api_key, model_id


def list_available_models() -> list[dict[str, str]]:
    """Return [{provider, model, id, base_url}] for every model in opencode.json.

    Used by the /model Telegram command to show what the user can switch to.
    """
    data = _load_opencode_json()
    providers = data.get("provider", {})
    out: list[dict[str, str]] = []
    for prov_key, prov in sorted(providers.items()):
        options = prov.get("options") or {}
        base_url = options.get("baseURL", "")
        models = prov.get("models") or {}
        for model_key, model_entry in sorted(models.items()):
            out.append(
                {
                    "provider": prov_key,
                    "model": model_key,
                    "id": (model_entry or {}).get("id") or model_key,
                    "base_url": base_url or "",
                }
            )
    return out


def resolve_model_arg(arg: str) -> tuple[str, str] | None:
    """Resolve a user-supplied /model argument into (provider, model).

    Accepts:
    - "provider/model" exact form (e.g. "zai-coding-plan/glm-5")
    - plain "model" — searches all providers for a matching model key or id
    - opencode variant key (e.g. "glm5", "glm51", "localqwen3527b") — resolved via
      VARIANT_PRESETS so it matches the main bot's hashtag model names
    Returns None if nothing matches.
    """
    arg = (arg or "").strip()
    if not arg:
        return None

    # 1. Try opencode variant key first (glm, glm47, glm5, glm51, localqwen3527b, ...)
    try:
        from app.agents._config import VARIANT_PRESETS  # TODO(v4-port): app.agents._config

        lower = arg.lower().lstrip("#")
        if lower in VARIANT_PRESETS:
            _postfix, model_name = VARIANT_PRESETS[lower]
            # Find which provider hosts that model in opencode.json
            for entry in list_available_models():
                if entry["model"] == model_name or entry["id"] == model_name:
                    return entry["provider"], entry["model"]
    except Exception:
        pass

    available = list_available_models()
    if "/" in arg:
        prov, mod = arg.split("/", 1)
        for entry in available:
            if entry["provider"] == prov and (entry["model"] == mod or entry["id"] == mod):
                return entry["provider"], entry["model"]
        return None
    # Bare model name — match model key or id, prefer exact key match.
    for entry in available:
        if entry["model"] == arg:
            return entry["provider"], entry["model"]
    for entry in available:
        if entry["id"] == arg:
            return entry["provider"], entry["model"]
    return None


# ── Prompt composition ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_writer_core() -> str:
    try:
        return _WRITER_CORE_MD.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.error("writer_core.md not found at %s", _WRITER_CORE_MD)
        return ""


def _format_character_header(characters: list[dict[str, Any]]) -> str:
    """Compact character list the writer can read at a glance."""
    if not characters:
        return ""
    lines: list[str] = ["## Active characters"]
    for c in characters:
        name = c.get("name") or "?"
        role = c.get("role") or ""
        location = c.get("location") or ""
        mood = c.get("mood") or ""
        personality = (c.get("personality") or "").strip()
        appearance = (c.get("appearance") or "").strip()
        goal = (c.get("current_goal") or "").strip()
        pressure = (c.get("pressure") or "").strip()
        hidden = (c.get("hidden_layer") or "").strip()
        outfit = (c.get("current_outfit") or "").strip()

        header = f"### {name}"
        if role:
            header += f" — {role}"
        lines.append(header)
        if location:
            lines.append(f"- **Location:** {location}")
        if mood:
            lines.append(f"- **Mood:** {mood}")
        if personality:
            lines.append(f"- **Personality:** {personality}")
        if appearance:
            lines.append(f"- **Appearance:** {appearance}")
        if outfit:
            lines.append(f"- **Wearing:** {outfit}")
        if goal:
            lines.append(f"- **Goal:** {goal}")
        if pressure:
            lines.append(f"- **Pressure:** {pressure}")
        if hidden:
            lines.append(f"- **Hidden layer (GM only — never reveal directly):** {hidden}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_world_header(adventure: dict[str, Any], world_aspects: list[dict[str, Any]]) -> str:
    lines: list[str] = ["## World"]
    title = adventure.get("title") or ""
    if title:
        lines.append(f"**Title:** {title}")
    setting = (adventure.get("setting") or "").strip()
    if setting:
        lines.append(f"**Setting:** {setting}")
    genre = adventure.get("genre") or ""
    tone = adventure.get("tone") or ""
    if genre or tone:
        lines.append(f"**Genre/tone:** {genre} / {tone}")
    scene = (adventure.get("current_scene") or "").strip()
    if scene:
        lines.append(f"**Current scene:** {scene}")
    t = adventure.get("inworld_time") or ""
    d = adventure.get("inworld_date") or ""
    if t or d:
        lines.append(f"**In-world time:** {d} {t}".strip())

    if world_aspects:
        lines.append("")
        lines.append("### World aspects")
        for aspect in world_aspects:
            a = aspect.get("aspect") or ""
            v = (aspect.get("value") or "").strip()
            if a and v:
                lines.append(f"- **{a}:** {v}")
    return "\n".join(lines).strip()


def _format_summaries(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return ""
    # Order: distant → mid → recent so the reading flow is chronological.
    order = {"distant": 0, "mid": 1, "recent": 2}
    ordered = sorted(summaries, key=lambda s: order.get(s.get("window", ""), 99))
    lines: list[str] = ["## Story so far"]
    for s in ordered:
        window = s.get("window") or ""
        lo = s.get("covers_from_turn")
        hi = s.get("covers_to_turn")
        text = (s.get("text") or "").strip()
        if not text:
            continue
        range_str = f" (turns {lo}-{hi})" if lo is not None and hi is not None else ""
        lines.append(f'<summary window="{window}"{range_str}>')
        lines.append(text)
        lines.append("</summary>")
    return "\n".join(lines).strip()


def _format_recall_pins(pins: list[dict[str, Any]]) -> str:
    if not pins:
        return ""
    lines: list[str] = ["## Recalled moments"]
    for p in pins:
        turn = p.get("turn")
        text = (p.get("text") or "").strip()
        if not text:
            continue
        lines.append(f'<recall_pin turn="{turn}">')
        lines.append(text)
        lines.append("</recall_pin>")
    return "\n".join(lines).strip()


def _format_bans(ban_rules: list[dict[str, Any]]) -> str:
    if not ban_rules:
        return ""
    lines = ["## BANNED PATTERNS — NEVER DO THESE"]
    for r in ban_rules:
        rule = (r.get("rule") or "").strip()
        if rule:
            lines.append(f"- {rule}")
    return "\n".join(lines).strip()


def _format_cross_summary(cross: dict[str, Any] | None) -> str:
    if not cross:
        return ""
    text = (cross.get("summary") or "").strip()
    if not text:
        return ""
    return f"## Player context (from main bot)\n{text}"


def build_system_prompt(
    *,
    adventure: dict[str, Any],
    characters: list[dict[str, Any]],
    world_aspects: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    recall_pins: list[dict[str, Any]],
    ban_rules: list[dict[str, Any]],
    cross_summary: dict[str, Any] | None,
    narrative_mode_block: str,
    writing_style_block: str,
    cot_block: str,
) -> str:
    """Assemble the full system prompt for the prose writer.

    Verbatim sections are loaded from writer_core.md and the narrative-mode /
    style / cot helper modules — nothing is reformatted by a compiler.
    """
    sections: list[str] = [_load_writer_core()]

    world_header = _format_world_header(adventure, world_aspects)
    if world_header:
        sections.append(world_header)

    char_header = _format_character_header(characters)
    if char_header:
        sections.append(char_header)

    summaries_block = _format_summaries(summaries)
    if summaries_block:
        sections.append(summaries_block)

    recall_block = _format_recall_pins(recall_pins)
    if recall_block:
        sections.append(recall_block)

    cross = _format_cross_summary(cross_summary)
    if cross:
        sections.append(cross)

    if narrative_mode_block:
        sections.append(narrative_mode_block.strip())
    if writing_style_block:
        sections.append(writing_style_block.strip())
    if cot_block:
        sections.append(cot_block.strip())

    bans = _format_bans(ban_rules)
    if bans:
        sections.append(bans)

    return "\n\n".join(s for s in sections if s)


# ── History → messages[] mapping ───────────────────────────────────────────


_ASSISTANT_ENTRY_TYPES = {"dialogue", "narration", "action", "system"}


def build_messages(
    story_log: list[dict[str, Any]],
    player_message: str,
) -> list[dict[str, str]]:
    """Map story log entries to OpenAI-format messages, ending with the new player turn.

    - entry_type == 'player'                    → {role: "user",      content: text}
    - entry_type in dialogue|narration|action|system → {role: "assistant", content: "<<Speaker>>\\n..."}
    - Consecutive assistant entries are merged into a single assistant message so the
      model sees a coherent multi-part turn (narration + dialogue together).
    """
    messages: list[dict[str, str]] = []
    pending_assistant: list[str] = []

    def flush_assistant() -> None:
        if pending_assistant:
            messages.append({"role": "assistant", "content": "\n\n".join(pending_assistant).strip()})
            pending_assistant.clear()

    for entry in story_log:
        etype = entry.get("entry_type") or "dialogue"
        speaker = (entry.get("speaker") or "").strip()
        content = (entry.get("content") or "").strip()
        if not content:
            continue

        if etype == "player":
            flush_assistant()
            messages.append({"role": "user", "content": content})
        elif etype in _ASSISTANT_ENTRY_TYPES:
            if etype == "dialogue" and speaker:
                pending_assistant.append(f"<<{speaker}>>\n{content}")
            elif etype == "narration":
                # Narration is italicized in the expected format.
                pending_assistant.append(f"*{content}*")
            elif etype == "action" and speaker:
                pending_assistant.append(f"*{speaker} {content}*")
            else:
                pending_assistant.append(content)
        else:
            # Unknown type — attach to assistant buffer as plain text.
            pending_assistant.append(content)

    flush_assistant()

    # Final new player turn
    messages.append({"role": "user", "content": player_message})
    return messages


# ── Direct API call ────────────────────────────────────────────────────────


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> or bare </think> reasoning blocks from the output."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


async def generate_prose(
    system: str,
    messages: list[dict[str, str]],
    *,
    role: Literal["prose", "orchestrator"] = "prose",
    override_provider: str | None = None,
    override_model: str | None = None,
) -> str:
    """Send a chat-completions request to the RP prose writer and return clean text.

    Runs the synchronous OpenAI client in a thread so it doesn't block the event loop.

    `override_provider`/`override_model` win over settings — used by the /model
    command to switch models on a per-adventure basis without touching .env.
    """
    from app.settings import get_settings

    settings = get_settings()
    base_url, api_key, model_id = load_model_config(
        role, override_provider=override_provider, override_model=override_model
    )
    if role == "prose":
        temperature = settings.rp_prose_temperature
        max_tokens = settings.rp_prose_max_tokens
    else:
        temperature = settings.rp_orchestrator_temperature
        max_tokens = settings.rp_orchestrator_max_tokens

    payload_messages: list[dict[str, str]] = [{"role": "system", "content": system}, *messages]

    # Only forward temperature / max_tokens when explicitly set in settings.
    # Otherwise let the backing vLLM (or provider) use its own configured defaults.
    extra_kwargs: dict[str, Any] = {}
    if temperature is not None:
        extra_kwargs["temperature"] = temperature
    if max_tokens is not None:
        extra_kwargs["max_tokens"] = max_tokens

    def _call() -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url or None,
            timeout=180,
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=payload_messages,  # type: ignore[arg-type]
            **extra_kwargs,
        )
        raw = resp.choices[0].message.content or ""
        return _strip_thinking(raw)

    try:
        return await asyncio.to_thread(_call)
    except Exception:
        logger.exception("RP prose generation failed (role=%s, model=%s)", role, model_id)
        raise
