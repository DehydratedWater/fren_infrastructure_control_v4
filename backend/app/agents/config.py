"""Model presets, variants, and split/dual-provider profiles for the fleet.

This is the v4 replacement for v3's hand-rolled `src/fren/agents/_config.py`
(ModelPreset + SplitProfile + the `_reset_model_coder()` global-mutation hack).
Everything here is built on the framework's first-class primitives:

- `ModelPreset`         — a named (provider, model_id, sampling) bundle;
- `VariantSpec`         — one compilation pass (postfix + preset);
- `SplitProfile`        — a VariantSpec that routes preset by agent `model_class`
                          (fast / analytical / vision), with vision PASSTHROUGH
                          so image agents keep their own model (v3 parity);
- worker vs live profiles — the two-target split: opencode workers run on z.ai
                          (the coding plan); the interactive/live layer runs on
                          a local OpenAI-compatible qwen, because z.ai can't be
                          driven from LangChain.

Secrets are referenced by ENV var name only (api_key_env) — never inlined.
"""

from __future__ import annotations

from src.model.core.model_preset import ModelPreset, SamplingDefaults
from src.model.core.split_profile import SplitProfile
from src.model.core.variant_spec import VariantSpec

# --- provider option blocks -------------------------------------------------
# z.ai coding plan (the worker provider — drives opencode).
_ZAI = {"api_key_env": "ZAI_API_KEY"}
# Local OpenAI-compatible vLLM servers on the LAN (no key needed).
_VLLM_REMOTE = {"base_url": "http://192.168.0.42:8082/v1", "api_key_env": "VLLM_API_KEY"}
_VLLM_ANALYTICAL = {"base_url": "http://192.168.0.42:8083/v1", "api_key_env": "VLLM_API_KEY"}
_VLLM_LOCAL = {"base_url": "http://192.168.0.42:5502/v1", "api_key_env": "VLLM_API_KEY"}
_VLLM_VISION = {"base_url": "http://192.168.0.42:5504/v1", "api_key_env": "VLLM_API_KEY"}


def _preset(name, provider, model_id, *, temperature=0.3, options=None) -> ModelPreset:
    return ModelPreset(
        name=name,
        provider=provider,
        model_id=model_id,
        sampling=SamplingDefaults(temperature=temperature),
        provider_options=dict(options or {}),
    )


# --- the worker model presets (z.ai cloud) ----------------------------------
GLM_45_AIR = _preset("glm-4.5-air", "zai-coding-plan", "glm-4.5-air", options=_ZAI)
GLM_47 = _preset("glm-4.7", "zai-coding-plan", "glm-4.7", options=_ZAI)
GLM_5 = _preset("glm-5", "zai-coding-plan", "glm-5", options=_ZAI)
GLM_51 = _preset("glm-5.1", "zai-coding-plan", "glm-5.1", options=_ZAI)

# --- local presets ----------------------------------------------------------
GLM_LOCAL = _preset("glm-4.5-air-local", "local", "glm-4.5-air-awq", options=_VLLM_LOCAL)
QWEN35_27B = _preset("qwen35-27b", "local", "qwen3.5-27b", options=_VLLM_REMOTE)

# --- split-profile presets (simultaneous multi-model serving) ---------------
SPLIT_FAST = _preset("qwen35-35b-a3b", "local", "qwen3.5-35b-a3b", options=_VLLM_REMOTE,
                     temperature=0.4)
SPLIT_ANALYTICAL = _preset("qwen35-27b-heretic", "local", "qwen3.5-27b-heretic",
                           options=_VLLM_ANALYTICAL, temperature=0.2)

# --- vision preset (used by passthrough agents under any split) -------------
QWEN_VL = _preset("qwen3-8b-vl", "local", "qwen3-8b-vl", options=_VLLM_VISION)


# --- the seven worker variants (one compile pass each) ----------------------
WORKER_VARIANTS: list[VariantSpec] = [
    VariantSpec(name="glm-4.5-air", postfix="", preset=GLM_45_AIR),
    VariantSpec(name="glm-4.7", postfix="-glm47", preset=GLM_47),
    VariantSpec(name="glm-5", postfix="-glm5", preset=GLM_5),
    VariantSpec(name="glm-5.1", postfix="-glm51", preset=GLM_51),
    VariantSpec(name="glm-4.5-air-local", postfix="-local", preset=GLM_LOCAL),
    VariantSpec(name="qwen35-27b", postfix="-qwen3527b", preset=QWEN35_27B),
    SplitProfile(
        name="splitqwen35",
        postfix="-splitqwen35",
        preset=SPLIT_ANALYTICAL,  # fallback
        class_map={"fast": SPLIT_FAST, "analytical": SPLIT_ANALYTICAL},
        default_class="analytical",
        # vision agents keep their own model under the split (v3 parity)
        passthrough_classes=("vision",),
    ),
]

# The default worker variant (what compiles to the unsuffixed primary.md).
DEFAULT_WORKER = WORKER_VARIANTS[0]


def live_profile() -> SplitProfile:
    """Interactive/live provider profile — local qwen, by model_class.

    The interactive (LangChain) layer resolves the SAME agent's model_class
    through this profile instead of the z.ai worker variant, because z.ai's
    coding plan can't be used with LangChain.
    """
    return SplitProfile(
        name="live",
        postfix="-live",
        preset=SPLIT_ANALYTICAL,
        class_map={
            "fast": SPLIT_FAST,
            "analytical": SPLIT_ANALYTICAL,
            "default": SPLIT_ANALYTICAL,
        },
        default_class="analytical",
        passthrough_classes=("vision",),
    )
