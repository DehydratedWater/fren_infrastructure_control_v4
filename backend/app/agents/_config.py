"""Compatibility shim — v3 `fren.agents._config` symbols, backed by v4 config.

v3 code (telegram state/prose, handlers `/models`, scheduler postfix resolution)
imports a handful of variant-registry symbols from `fren.agents._config`. v4's
canonical model config lives in `app/agents/config.py` (ModelPresets +
WORKER_VARIANTS incl. the split profile). This module re-derives the exact v3
surface from that single source of truth so the ported runtime resolves model
variants / display names / file postfixes identically to v3 — no duplicated
preset data.
"""

from __future__ import annotations

from app.agents.config import (
    GLM_47,
    GLM_51,
    QWEN35_27B,
    WORKER_VARIANTS,
)

# variant name → ModelPreset. Only the three live variants:
# qwen35-27b (the local DEFAULT) + the two alt cloud passes glm-4.7 / glm-5.1.
VARIANT_PRESETS = {
    "qwen35-27b": QWEN35_27B,
    "glm-4.7": GLM_47,
    "glm-5.1": GLM_51,
}

MODEL_DISPLAY = {
    # compiled variant names
    "qwen35-27b": "Qwen3.5-27B 🏠",
    "glm-4.7": "GLM-4.7 ☁️",
    "glm-5.1": "GLM-5.1 ☁️",
    # telegram state/chat model keys (so the header renders on the chat surface)
    "localqwen3527b": "Qwen3.5-27B 🏠",
    "glm47": "GLM-4.7 ☁️",
    "glm51": "GLM-5.1 ☁️",
}

# qwen is the only LOCAL (NSFW-capable) worker now. Includes BOTH the compiled
# variant name ("qwen35-27b") AND the telegram state/chat model key
# ("localqwen3527b") so is_local_model() resolves correctly on the chat surface.
LOCAL_MODEL_KEYS = {"qwen35-27b", "localqwen3527b"}

_MODEL_PRESET_MAP = {**VARIANT_PRESETS}

# variant name → file postfix, derived from the canonical WORKER_VARIANTS so it
# can never drift from what compile actually emits.
_POSTFIX = {v.name: v.postfix for v in WORKER_VARIANTS}


def get_postfix(variant: str) -> str:
    """Return the compiled-file postfix for a model variant (matches compile)."""
    return _POSTFIX.get(variant, "")
