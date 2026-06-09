"""Bot state persistence — mode (chat/work) + model selection + content mode.

Reads/writes data/bot_state.json. Same pattern as scheduler_state.json.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_PATH: Path | None = None

# Tag → model key mapping. Only the THREE selectable worker models remain:
# qwen3.5-27b (local DEFAULT) + glm-4.7 + glm-5.1.
_TAG_MAP: dict[str, str] = {
    "#localqwen3527b": "localqwen3527b",
    "#qwen": "localqwen3527b",
    "#glm47": "glm47",
    "#glm51": "glm51",
}

_TAG_RE = re.compile(
    r"(?:^|\s)(#(?:"
    r"localqwen3527b|qwen|glm47|glm51"
    r"))\b",
    re.IGNORECASE,
)

# Content classification tags (#nsfw, #secret) — separate from model tags
_CONTENT_TAG_RE = re.compile(r"(?:^|\s)(#(?:nsfw|secret))\b", re.IGNORECASE)


def _state_path() -> Path:
    global _STATE_PATH
    if _STATE_PATH is None:
        from app.settings import get_settings

        _STATE_PATH = Path(get_settings().project_root) / "data" / "bot_state.json"
    return _STATE_PATH


def _load() -> dict[str, Any]:
    try:
        with open(_state_path()) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"mode": "work", "model": "localqwen3527b", "content_mode": "sfw"}


def _save(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def get_state() -> dict[str, Any]:
    return _load()


def get_mode() -> str:
    return _load().get("mode", "work")


def set_mode(mode: str) -> None:
    state = _load()
    state["mode"] = mode
    _save(state)


def get_model() -> str:
    return _load().get("model", "localqwen3527b")


def set_model(model: str) -> None:
    state = _load()
    state["model"] = model
    _save(state)


# ── Scheduler Model ──


def get_scheduler_model() -> str:
    """Get the model used for scheduled/cron tasks. Falls back to main model if not set."""
    return _load().get("scheduler_model") or get_model()


def set_scheduler_model(model: str) -> None:
    """Set the model used for scheduled/cron tasks."""
    state = _load()
    state["scheduler_model"] = model
    _save(state)


# ── TTS Model ──


def get_tts_model() -> str:
    """Get the model used for TTS formatting. Falls back to main model if not set."""
    return _load().get("tts_model") or get_model()


def set_tts_model(model: str) -> None:
    """Set the model used for TTS formatting."""
    state = _load()
    state["tts_model"] = model
    _save(state)


# ── Local Model Default ──


def get_default_local_model() -> str:
    """Get preferred local model key. Used when forcing local (e.g. NSFW)."""
    return _load().get("default_local_model", "localqwen3527b")


def set_default_local_model(model: str) -> None:
    """Set preferred local model key."""
    state = _load()
    state["default_local_model"] = model
    _save(state)


# ── Content Mode ──


def get_content_mode() -> str:
    """Get current content mode: 'sfw' or 'nsfw'."""
    return _load().get("content_mode", "sfw")


def set_content_mode(mode: str) -> None:
    """Set content mode ('sfw' or 'nsfw')."""
    state = _load()
    state["content_mode"] = mode
    _save(state)


def is_local_model(model: str | None = None) -> bool:
    """Check if model key refers to a local (NSFW-capable) model."""
    from app.agents._config import LOCAL_MODEL_KEYS

    if model is None:
        model = get_model()
    return model in LOCAL_MODEL_KEYS


# ── Emotions / Inner Monologue ──


def get_emotions_enabled() -> bool:
    """Check if inner monologue is enabled."""
    return _load().get("emotions_enabled", True)


def set_emotions_enabled(enabled: bool) -> None:
    """Enable or disable inner monologue."""
    state = _load()
    state["emotions_enabled"] = enabled
    _save(state)


# ── Content Tags ──


def parse_content_tags(text: str) -> set[str]:
    """Extract content tags (#nsfw, #secret) from text. Returns set of tag names."""
    return {m.group(1).lstrip("#").lower() for m in _CONTENT_TAG_RE.finditer(text)}


def strip_content_tags(text: str) -> str:
    """Remove content tags from text."""
    return _CONTENT_TAG_RE.sub("", text).strip()


# ── Model Tags ──


def get_postfix(model: str | None = None) -> str:
    # _POSTFIX maps a variant NAME (e.g. "qwen35-27b") to the compiled file
    # postfix (e.g. "-qwen3527b"), derived from WORKER_VARIANTS so it never drifts.
    # (The old code unpacked VARIANT_PRESETS values — ModelPresets, not tuples —
    # which raised on every valid key and silently fell back to the z.ai default,
    # so the whole fleet ran on glm-4.5-air instead of the selected model.)
    from app.agents._config import _POSTFIX

    if model is None:
        model = get_model()
    # The state/chat model keys (#hashtag names like "localqwen3527b") use a
    # different naming than the compiled variant names ("qwen35-27b"); normalise
    # the local ones so the selected model actually routes to its variant.
    _STATE_TO_VARIANT = {
        "localqwen3527b": "qwen35-27b",
        "glm47": "glm-4.7",
        "glm51": "glm-5.1",
    }
    name = _STATE_TO_VARIANT.get((model or "").lower().lstrip("#"), model)
    return _POSTFIX.get(name, "")


def get_model_display(model: str | None = None) -> str:
    from app.agents._config import MODEL_DISPLAY

    if model is None:
        model = get_model()
    return MODEL_DISPLAY.get(model, model)


def format_header(mode: str | None = None, model: str | None = None) -> str:
    if mode is None:
        mode = get_mode()
    if model is None:
        model = get_model()
    display = get_model_display(model)
    content = get_content_mode()
    if content == "nsfw":
        return f"[{mode}|{display}|nsfw]"
    return f"[{mode}|{display}]"


def parse_model_tag(text: str) -> str | None:
    """Extract first model tag from text, return model key or None."""
    m = _TAG_RE.search(text)
    if m:
        tag = m.group(1).lower()
        return _TAG_MAP.get(tag)
    return None


def strip_model_tag(text: str) -> str:
    """Remove model tag from text."""
    return _TAG_RE.sub("", text).strip()


# ── vLLM Variant (dense/moe on port 8082) ──

_VLLM_STATE_PATH: Path | None = None

_VLLM_LABELS = {
    "dense": "Dense 27B (Qwen3.6 AutoRound)",
}


def _vllm_state_path() -> Path:
    global _VLLM_STATE_PATH
    if _VLLM_STATE_PATH is None:
        from app.settings import get_settings

        _VLLM_STATE_PATH = Path(get_settings().project_root) / "data" / "vllm_state.json"
    return _VLLM_STATE_PATH


def get_vllm_variant() -> str:
    """Get currently loaded vLLM variant ('dense', 'moe', or 'unknown')."""
    try:
        with open(_vllm_state_path()) as f:
            return json.load(f).get("variant", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


def get_vllm_display() -> str:
    """Get human-readable vLLM variant label."""
    variant = get_vllm_variant()
    return _VLLM_LABELS.get(variant, variant)
