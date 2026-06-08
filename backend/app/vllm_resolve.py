"""Resolve the active local-vLLM endpoint for tool-side LLM calls.

Port of v3's `fren/vllm_resolve.py`. Reads `data/vllm_state.json` to learn which
serving variant is up, then returns the (base_url, model) for the requested
role. Matches the endpoints declared in app/agents/config.py (the split fast /
analytical vLLM servers). Tools that need a quick local model call use this
rather than hard-coding a URL.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATE_FILE = _PROJECT_ROOT / "data" / "vllm_state.json"

# Cache discovered served-model ids per base_url so we don't hit /models every call.
_MODEL_CACHE: dict[str, tuple[str, float]] = {}
_MODEL_CACHE_TTL = 120.0


def _served_model(base_url: str) -> str | None:
    """The model id the vLLM at base_url actually serves (from /v1/models), cached.

    The static ids below are placeholders (e.g. 'qwen3.5-27b') that 404 — vLLM
    serves the real id like 'cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8'. Returns None when
    the endpoint is unreachable so the caller can fall back."""
    now = time.monotonic()
    hit = _MODEL_CACHE.get(base_url)
    if hit and hit[1] > now:
        return hit[0]
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=4) as r:  # noqa: S310
            data = json.loads(r.read())
        mid = (data.get("data") or [{}])[0].get("id")
        if mid:
            _MODEL_CACHE[base_url] = (mid, now + _MODEL_CACHE_TTL)
            return mid
    except Exception:
        return None
    return None

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
    """Return (base_url, model_name) for the active vLLM variant + role.

    Resolves the REAL served model id from the endpoint (the static ids are
    placeholders that 404) and falls back to the known-up fallback endpoint when
    the role's endpoint (e.g. the analytical :8083) is unreachable."""
    variant = _get_variant()
    if variant in _SINGLE_MODEL:
        base, _ = _SINGLE_MODEL[variant]
    elif variant == "split":
        base, _ = _SPLIT_ENDPOINTS.get(role, _SPLIT_ENDPOINTS["analytical"])
    else:
        base, _ = _FALLBACK

    served = _served_model(base)
    if served:
        return base, served
    # Endpoint unreachable — fall back to the known-up endpoint + its served id.
    fb_base, fb_model = _FALLBACK
    return fb_base, (_served_model(fb_base) or fb_model)
