"""Fleet registry — collect every domain's agents into one AgentRegistry.

Each agent becomes its own template slot (so it compiles to its own
`.opencode/agents/<path>.md`), and `register_with_improvements` merges any
promoted prompt from `.oac/promoted/` before registration — so a tuned agent
ships its improved prompt on the next compile with no code change.

Variants (the 7 worker passes incl. the split profile) are applied by
`CompileScript`, not here; this module just defines WHAT to compile.
"""

from __future__ import annotations

from pathlib import Path

from app.agents.config import DEFAULT_WORKER, QWEN_VL
from app.agents.domains import all_agent_defs
from src import (
    AgentDefinition,
    AgentRegistry,
    CompilationConfig,
    TemplateSlot,
    TemplateTree,
)

# The repo root that holds `.oac/promoted/` (three parents up from this file:
# backend/app/agents/registry.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def all_agents() -> list[AgentDefinition]:
    return all_agent_defs()


def build_registry(*, project_root: Path | None = None) -> AgentRegistry:
    reg = AgentRegistry()
    base_params = DEFAULT_WORKER.preset.to_model_parameters()
    # v3 parity: vision-class agents (e.g. food/product_image_indexer,
    # support/image_processor) are HARD-BOUND to the local vision model
    # (local-vllm-image/qwen3-8b-vl) at definition time — v3's
    # apply_model(builder, MODEL_VISION). Every worker variant marks "vision"
    # as a passthrough class, so these agents keep this base model across ALL
    # variants instead of being rerouted onto a text-only model. Binding the
    # vision preset HERE is what that passthrough preserves.
    vision_params = QWEN_VL.to_model_parameters()
    root = project_root or PROJECT_ROOT
    slots: list[TemplateSlot] = []
    for agent in all_agents():
        # PRODUCTION DELIVERY CONTRACT: a delivery agent (emit_guidance.py in its
        # allow-list) whose prompt does NOT already instruct emit_guidance would
        # produce ONLY invisible assistant text — delivering nothing. Inject the
        # strong DELIVERY_POSTAMBLE (modelled on goals/evening_focus) so the model
        # reliably ends its run by calling emit_guidance.py. We merge any promoted
        # improvements FIRST, then decide injection on the FINAL shipping prompt so
        # we never double-add for the ~36 agents that already instruct it (or a
        # promoted prompt that learned it). See app/agents/improve.py.
        from app.agents.improve import with_delivery_postamble
        from src.improvement.snapshot import apply_promoted_to_tree

        improved = apply_promoted_to_tree(
            agent,
            project_root=root,
            model_class=agent.model_class,
        )
        improved = with_delivery_postamble(improved)
        agent_params = vision_params if agent.model_class == "vision" else base_params
        agent_id = reg.register_agent(
            agent.header.agent_id,
            improved,
            agent_params,
        )
        # also_compile_as_primary → the compiler emits BOTH
        # `<name>.md` (mode: subagent, for Task dispatch from an orchestrator)
        # AND `<name>-primary.md` (mode: primary), the latter directly
        # spawnable via `opencode run --agent <name>-primary`. A slot not named
        # "primary" otherwise compiles subagent-only, which opencode rejects on
        # `run --agent` (falls back to its default assistant).
        slots.append(
            TemplateSlot(
                name=agent.header.agent_id,
                default_agent_id=agent_id,
                also_compile_as_primary=True,
            )
        )
    reg.register_template(TemplateTree(name="fleet", slots=slots))
    reg.create_compilation_config(
        CompilationConfig(name="prod", template_name="fleet")
    )
    return reg
