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
# NOTE on `provider` + `model_id`: the compiled `model:` line is
# `<provider>/<model_id>` (ModelPreset.qualified_model_name). These MUST match
# the provider KEY and model KEY declared in opencode.json — opencode then maps
# that model KEY to the real served `id` (e.g. cyankiwi/Qwen3.5-27B-AWQ-...)
# internally. So `model_id` here is the opencode model KEY, NOT the served id.
GLM_45_AIR = _preset("glm-4.5-air", "zai-coding-plan", "glm-4.5-air", options=_ZAI)
GLM_47 = _preset("glm-4.7", "zai-coding-plan", "glm-4.7", options=_ZAI)
GLM_5 = _preset("glm-5", "zai-coding-plan", "glm-5", options=_ZAI)
GLM_51 = _preset("glm-5.1", "zai-coding-plan", "glm-5.1", options=_ZAI)

# --- local presets ----------------------------------------------------------
# provider/model KEYS below mirror opencode.json: local-vllm/glm-4.5-air-local
# (served id cyankiwi/GLM-4.5-Air-Derestricted-AWQ-4bit) and
# local-vllm-remote/qwen35-27b (served id cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8).
GLM_LOCAL = _preset("glm-4.5-air-local", "local-vllm", "glm-4.5-air-local", options=_VLLM_LOCAL)
QWEN35_27B = _preset("qwen35-27b", "local-vllm-remote", "qwen35-27b", options=_VLLM_REMOTE)

# --- split-profile presets (simultaneous multi-model serving) ---------------
# local-vllm-fast/qwen35-35b-a3b + local-vllm-analytical/qwen35-27b-heretic.
SPLIT_FAST = _preset("qwen35-35b-a3b", "local-vllm-fast", "qwen35-35b-a3b", options=_VLLM_REMOTE,
                     temperature=0.4)
SPLIT_ANALYTICAL = _preset("qwen35-27b-heretic", "local-vllm-analytical", "qwen35-27b-heretic",
                           options=_VLLM_ANALYTICAL, temperature=0.2)

# --- vision preset (used by passthrough agents under any variant) -----------
# local-vllm-image/qwen3-8b-vl. Vision-class agents are bound to this in the
# registry and kept here across every variant via passthrough_classes.
QWEN_VL = _preset("qwen3-8b-vl", "local-vllm-image", "qwen3-8b-vl", options=_VLLM_VISION)


def _worker(name: str, postfix: str, preset: ModelPreset) -> SplitProfile:
    """A single-model worker variant that PINS vision agents to the vision model.

    v3 parity: vision-class agents (qwen3-8b-vl) keep the local vision model in
    EVERY variant, not just the split. A plain VariantSpec would rewrite every
    agent's model to `preset` — including vision agents — pointing them at a
    text-only model that can't read images.

    We route `vision` -> QWEN_VL in the class_map (rather than using
    `passthrough_classes`) on purpose: passthrough makes apply_variant return the
    agent unchanged, which DROPS the postfix and collapses all 7 variant files
    into one. Routing keeps the postfix, so `<vision_agent><postfix>.md` is still
    emitted per variant (so a -glm47 orchestrator can dispatch the vision
    subagent), exactly matching v3's 7-file-per-vision-agent output — while still
    binding the model to local-vllm-image/qwen3-8b-vl.
    """
    return SplitProfile(
        name=name,
        postfix=postfix,
        preset=preset,  # fallback for any non-vision class
        class_map={
            "default": preset,
            "fast": preset,
            "analytical": preset,
            "vision": QWEN_VL,
        },
        default_class="default",
        passthrough_classes=(),  # nothing passes through; vision is routed above
    )


# --- the seven worker variants (one compile pass each) ----------------------
WORKER_VARIANTS: list[VariantSpec] = [
    _worker("glm-4.5-air", "", GLM_45_AIR),
    _worker("glm-4.7", "-glm47", GLM_47),
    _worker("glm-5", "-glm5", GLM_5),
    _worker("glm-5.1", "-glm51", GLM_51),
    _worker("glm-4.5-air-local", "-local", GLM_LOCAL),
    _worker("qwen35-27b", "-qwen3527b", QWEN35_27B),
    SplitProfile(
        name="splitqwen35",
        postfix="-splitqwen35",
        preset=SPLIT_ANALYTICAL,  # fallback
        # vision -> QWEN_VL (route, don't passthrough) so the
        # `<agent>-splitqwen35.md` file is still emitted (passthrough would drop
        # the postfix and collapse it into the default). v3 parity: vision agents
        # bind to local-vllm-image/qwen3-8b-vl in this variant too.
        class_map={
            "fast": SPLIT_FAST,
            "analytical": SPLIT_ANALYTICAL,
            "vision": QWEN_VL,
        },
        default_class="analytical",
        passthrough_classes=(),
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
