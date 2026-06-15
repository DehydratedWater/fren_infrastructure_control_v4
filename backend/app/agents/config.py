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
# z.ai coding plan (the worker provider — drives opencode) — used by the two
# ALTERNATIVE (non-default) variants glm-4.7 / glm-5.1 only.
_ZAI = {"api_key_env": "ZAI_API_KEY"}
# Local OpenAI-compatible vLLM server on the LAN (no key needed). The local
# qwen-27B on :8082 is the DEFAULT/primary worker model for the whole fleet.
_VLLM_REMOTE = {"base_url": "http://192.168.0.42:8082/v1", "api_key_env": "VLLM_API_KEY"}
# The split/analytical endpoint is still used by the interactive live_profile
# (vllm_resolve); the local-glm (:5502) and A4000-vision (:5504) endpoints were
# dropped along with their worker variants.
_VLLM_ANALYTICAL = {"base_url": "http://192.168.0.42:8083/v1", "api_key_env": "VLLM_API_KEY"}


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
# Only the two ALTERNATIVE cloud variants remain (glm-4.5-air / glm-5 dropped).
GLM_47 = _preset("glm-4.7", "zai-coding-plan", "glm-4.7", options=_ZAI)
GLM_51 = _preset("glm-5.1", "zai-coding-plan", "glm-5.1", options=_ZAI)

# --- local preset (the DEFAULT/primary worker) ------------------------------
# provider/model KEY mirrors opencode.json: local-vllm-remote/qwen35-27b
# (served id cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8). This is the fleet DEFAULT —
# the empty-postfix variant — so the primary fleet runs on the local qwen.
QWEN35_27B = _preset("qwen35-27b", "local-vllm-remote", "qwen35-27b", options=_VLLM_REMOTE)

# LIVE preset for the interactive (LangChain/in-process) first-contact tier.
# Unlike the worker preset above, the interactive OpenAICompatClient calls vLLM
# DIRECTLY (no opencode KEY→served-id mapping), so model_id MUST be the SERVED
# id. temperature 1.0 is qwen3.5's optimal for conversational generation.
QWEN35_27B_LIVE = _preset(
    "qwen35-27b-live", "local-vllm-remote",
    "cyankiwi/Qwen3.5-27B-AWQ-BF16-INT8",
    temperature=1.0,
    # enable_thinking=false: the FAST first-contact tier wants SNAP, not a
    # reasoning trace — and Qwen3.x on vLLM returns EMPTY content unless thinking
    # is disabled for the direct (non-interleaved) path. The heavy opencode tier
    # keeps thinking ON; only this quick tier turns it off.
    options={**_VLLM_REMOTE, "extra_body": '{"chat_template_kwargs": {"enable_thinking": false}}'},
)

# --- split-profile presets (simultaneous multi-model serving) ---------------
# local-vllm-fast/qwen35-35b-a3b + local-vllm-analytical/qwen35-27b-heretic.
SPLIT_FAST = _preset("qwen35-35b-a3b", "local-vllm-fast", "qwen35-35b-a3b", options=_VLLM_REMOTE,
                     temperature=0.4)
SPLIT_ANALYTICAL = _preset("qwen35-27b-heretic", "local-vllm-analytical", "qwen35-27b-heretic",
                           options=_VLLM_ANALYTICAL, temperature=0.2)

# --- vision preset (used by passthrough agents under any variant) -----------
# The local qwen-27B (:8082) is multimodal — it HAS vision — so vision-class
# agents run on it directly; no separate vision LLM. The dead A4000 vision model
# (local-vllm-image/qwen3-8b-vl on :5504) is dropped per requirements: only the
# one local qwen-27B + the small emotional-core model (:5506) are needed.
QWEN_VL = QWEN35_27B


def _worker(name: str, postfix: str, preset: ModelPreset) -> SplitProfile:
    """A single-model worker variant that PINS vision agents to the vision model.

    v3 parity: vision-class agents (qwen3-8b-vl) keep the local vision model in
    EVERY variant, not just the split. A plain VariantSpec would rewrite every
    agent's model to `preset` — including vision agents — pointing them at a
    text-only model that can't read images.

    We route `vision` -> QWEN_VL in the class_map (rather than using
    `passthrough_classes`) on purpose: passthrough makes apply_variant return the
    agent unchanged, which DROPS the postfix and collapses the variant files into
    one. Routing keeps the postfix, so `<vision_agent><postfix>.md` is still
    emitted per variant (so a -glm47 orchestrator can dispatch the vision
    subagent). QWEN_VL == QWEN35_27B (the local qwen-27B is multimodal), so vision
    binds to local-vllm-remote/qwen35-27b in every variant.
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


# --- the three worker variants (one compile pass each) ----------------------
# DEFAULT (empty postfix) = the local qwen-27B, so the primary fleet
# (`<agent>.md` + `<agent>-primary.md`) runs locally. glm-4.7 / glm-5.1 are the
# two alternative cloud compilations (`-glm47` / `-glm51`).
WORKER_VARIANTS: list[VariantSpec] = [
    _worker("qwen35-27b", "", QWEN35_27B),  # default/primary (postfix "")
    _worker("glm-4.7", "-glm47", GLM_47),
    _worker("glm-5.1", "-glm51", GLM_51),
]

# The default worker variant (what compiles to the unsuffixed primary.md) —
# the local qwen-27B.
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
