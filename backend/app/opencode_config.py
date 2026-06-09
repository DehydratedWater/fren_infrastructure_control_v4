"""Generate the global opencode.json the runtime needs.

The bot/scheduler shell out to `opencode run`, and persona_prose / rp_prose
resolve their (base_url, api_key, model) by reading the provider blocks from
`~/.config/opencode/opencode.json` (keyed by provider name, e.g.
`local-vllm-remote`). v4's entrypoint previously wrote only the `zai-coding-plan`
block, so persona_prose found no `local-vllm-*` provider → empty base_url →
fell through to api.openai.com and 401'd.

This writes the TRIMMED provider set: `local-vllm-remote` (the local Qwen-27B
on :8082 — the DEFAULT worker + the voice/prose target) plus `zai-coding-plan`
declaring the two alt cloud models glm-4.7 / glm-5.1. Secrets stay as `env:VAR`
refs (opencode expands them), never inlined.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_VLLM_TIMEOUT = 1800000


def build_config(worker_model: str = "local-vllm-remote/qwen35-27b") -> dict:
    """Build the opencode.json dict (providers + default model).

    Only the THREE live worker models are declared:
      - local-vllm-remote/qwen35-27b  — the DEFAULT/primary (local qwen, :8082;
        multimodal, so vision/video route here too);
      - zai-coding-plan/glm-4.7, zai-coding-plan/glm-5.1 — the two ALT cloud
        compilations (`-glm47` / `-glm51`).
    The split / local-glm (:5502) / analytical (:8083) / A4000-vision (:5504)
    providers were dropped — no compiled worker model references them, and the
    persona/rp prose layer resolves only `local-vllm-remote`.
    """
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": worker_model,
        "small_model": worker_model,
        "provider": {
            # Worker provider — z.ai coding plan (key from env at run time). The
            # two alt variants compile to glm-4.7 / glm-5.1, declared here so
            # `-glm47` / `-glm51` resolve instead of silently falling back.
            "zai-coding-plan": {
                "options": {"apiKey": "${ZAI_API_KEY}"},
                "models": {
                    "glm-4.7": {"id": "glm-4.7", "options": {"temperature": 0.6}},
                    "glm-5.1": {"id": "glm-5.1", "options": {"temperature": 0.6}},
                },
            },
            # Local vLLM on the A4000 (:8082) — the DEFAULT worker + the
            # persona/rp prose + interactive targets. Multimodal (vision/video).
            "local-vllm-remote": {
                "options": {
                    "apiKey": "env:VLLM_API_KEY",
                    "baseURL": "http://192.168.0.42:8082/v1",
                    "timeout": _VLLM_TIMEOUT,
                },
                "models": {
                    "qwen35-27b": {
                        "id": "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8",
                        "reasoning": True,
                        "interleaved": {"field": "reasoning_content"},
                        "limits": {"context": 262144, "output": 32768},
                        "options": {"temperature": 0.6, "topP": 0.95, "topK": 20},
                    },
                },
            },
        },
    }


# Where the config must land:
#  - the opencode CLI reads ~/.config/opencode/opencode.json
#  - persona_prose / rp_prose read the PROJECT-ROOT opencode.json (resolved as
#    parents[3] of app/telegram/rp_prose.py → /app inside the container)
def _project_root_config() -> Path:
    """The exact path persona_prose/rp_prose read — derived from rp_prose's own
    `_OPENCODE_JSON` constant so it can't drift from where they look."""
    try:
        from app.telegram.rp_prose import _OPENCODE_JSON

        return Path(_OPENCODE_JSON)
    except Exception:  # rp_prose imports python-telegram-bot etc; fall back
        return Path(__file__).resolve().parents[2] / "opencode.json"


def write_config(path: Path | None = None, *, worker_model: str | None = None) -> list[Path]:
    """Write opencode.json to all locations the runtime reads.

    The z.ai key is MATERIALISED from $ZAI_API_KEY at write time (opencode's
    `env:VAR` substitution proved unreliable for it). The local-vLLM keys stay
    as `env:VLLM_API_KEY` refs — the servers ignore the key anyway.

    Writes to BOTH ~/.config/opencode/opencode.json (the opencode CLI) AND the
    project-root opencode.json (persona_prose/rp_prose). Pass `path` to write a
    single explicit location instead.
    """
    wm = worker_model or os.environ.get("WORKER_MODEL", "local-vllm-remote/qwen35-27b")
    cfg = build_config(wm)
    cfg["provider"]["zai-coding-plan"]["options"]["apiKey"] = os.environ.get("ZAI_API_KEY", "")
    blob = json.dumps(cfg, indent=2)

    if path is not None:
        targets = [path]
    else:
        targets = [
            Path.home() / ".config" / "opencode" / "opencode.json",  # opencode CLI
            _project_root_config(),                                   # prose modules
        ]
    written: list[Path] = []
    for dest in targets:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(blob)
        written.append(dest)
    return written


if __name__ == "__main__":
    for p in write_config():
        print(f"[opencode-config] wrote {p}")
