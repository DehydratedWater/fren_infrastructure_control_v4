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
    GLM_45_AIR,
    GLM_47,
    GLM_5,
    GLM_51,
    GLM_LOCAL,
    QWEN35_27B,
    WORKER_VARIANTS,
)

# variant name → ModelPreset (None for the split profile, handled separately).
VARIANT_PRESETS = {
    "glm-4.5-air": GLM_45_AIR,
    "glm-4.7": GLM_47,
    "glm-5": GLM_5,
    "glm-5.1": GLM_51,
    "glm-4.5-air-local": GLM_LOCAL,
    "qwen35-27b": QWEN35_27B,
    "splitqwen35": None,
}

MODEL_DISPLAY = {
    "glm-4.5-air": "GLM-4.5-Air ☁️",
    "glm-4.7": "GLM-4.7 ☁️",
    "glm-5": "GLM-5 ☁️",
    "glm-5.1": "GLM-5.1 ☁️",
    "glm-4.5-air-local": "GLM-4.5-Air (local) 🏠",
    "qwen35-27b": "Qwen3.5-27B 🏠",
    "splitqwen35": "Split Qwen3.5 (fast+deep) 🏠",
}

LOCAL_MODEL_KEYS = {"glm-4.5-air-local", "qwen35-27b", "splitqwen35"}

_MODEL_PRESET_MAP = {**VARIANT_PRESETS}

# variant name → file postfix, derived from the canonical WORKER_VARIANTS so it
# can never drift from what compile actually emits.
_POSTFIX = {v.name: v.postfix for v in WORKER_VARIANTS}


def get_postfix(variant: str) -> str:
    """Return the compiled-file postfix for a model variant (matches compile)."""
    return _POSTFIX.get(variant, "")
