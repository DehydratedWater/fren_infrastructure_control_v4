"""Resolve the active local-vLLM endpoint for tool-side LLM calls.

Port of v3's `fren/vllm_resolve.py`. Reads `data/vllm_state.json` to learn which
serving variant is up, then returns the (base_url, model) for the requested
role. Matches the endpoints declared in app/agents/config.py (the split fast /
analytical vLLM servers). Tools that need a quick local model call use this
rather than hard-coding a URL.
"""

from __future__ import annotations

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATE_FILE = _PROJECT_ROOT / "data" / "vllm_state.json"

_SINGLE_MODEL = {
    "dense": ("http://192.168.0.42:8082/v1", "qwen3.5-27b"),
    "moe": ("http://192.168.0.42:8082/v1", "qwen3.5-27b"),
}

_SPLIT_ENDPOINTS = {
    "fast": ("http://192.168.0.42:8082/v1", "qwen35-35b-a3b"),
    "analytical": ("http://192.168.0.42:8083/v1", "qwen35-27b-heretic"),
}

_FALLBACK = ("http://192.168.0.42:8082/v1", "qwen3.5-27b")


def _get_variant() -> str:
    try:
        return json.loads(_STATE_FILE.read_text()).get("variant", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


def get_llm_endpoint(role: str = "analytical") -> tuple[str, str]:
    """Return (base_url, model_name) for the active vLLM variant + role."""
    variant = _get_variant()
    if variant in _SINGLE_MODEL:
        return _SINGLE_MODEL[variant]
    if variant == "split":
        return _SPLIT_ENDPOINTS.get(role, _SPLIT_ENDPOINTS["analytical"])
    return _FALLBACK
