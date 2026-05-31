"""Personality-core HTTP scorer/inference helper.

The Personality Core is a fine-tuned model served over an OpenAI-compatible
``/v1/chat/completions`` endpoint at ``settings.personality_core_host``. This
module exposes the importable helper surface for the ``app.personality``
package.

v3 has no dedicated ``fren/personality/`` package — the canonical call lives
inside ``fren/tools/personality/personality_core.py`` as
``PersonalityCoreTool._call_model``. This helper mirrors that exact request
shape (model name, max_tokens, temperature, top_p, system+user messages, the
``http://{host}/v1/chat/completions`` URL and the 60s httpx timeout) so any
caller wanting just the raw completion can use ``app.personality`` without
constructing the full ScriptTool.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.settings import get_settings

# Shared system prompt — re-exported from the tool so the helper and the
# ScriptTool stay in lockstep.
from app.tools.personality.personality_core import SYSTEM_PROMPT


def call_personality_core(internal_state: dict[str, Any], stimuli: str) -> str:
    """Call the Personality Core model and return the raw completion text.

    Faithful extraction of ``PersonalityCoreTool._call_model``: same URL,
    payload, and timeout. Synchronous (httpx.Client), matching v3.
    """
    settings = get_settings()
    url = f"http://{settings.personality_core_host}/v1/chat/completions"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (f"INTERNAL STATE:\n{json.dumps(internal_state)}\n\nEXTERNAL STIMULI:\n{stimuli}"),
        },
    ]

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            url,
            json={
                "model": "personality-core",
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
