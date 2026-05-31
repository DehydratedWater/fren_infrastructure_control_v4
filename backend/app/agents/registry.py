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
from types import ModuleType

from app.agents.config import DEFAULT_WORKER
from app.agents.domains import persona
from src import (
    AgentDefinition,
    AgentRegistry,
    CompilationConfig,
    TemplateSlot,
    TemplateTree,
)

# The repo root that holds `.oac/promoted/` (two levels up from this file:
# backend/app/agents/registry.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Every domain module exposing `agents()`. Grows as domains are ported.
DOMAINS: tuple[ModuleType, ...] = (
    persona,
)


def all_agents() -> list[AgentDefinition]:
    out: list[AgentDefinition] = []
    seen: set[str] = set()
    for domain in DOMAINS:
        for agent in domain.agents():
            aid = agent.header.agent_id
            if aid in seen:
                raise ValueError(f"duplicate agent_id across domains: {aid!r}")
            seen.add(aid)
            out.append(agent)
    return out


def build_registry(*, project_root: Path | None = None) -> AgentRegistry:
    reg = AgentRegistry()
    base_params = DEFAULT_WORKER.preset.to_model_parameters()
    root = project_root or PROJECT_ROOT
    slots: list[TemplateSlot] = []
    for agent in all_agents():
        agent_id = reg.register_with_improvements(
            agent.header.agent_id,
            agent,
            base_params,
            project_root=root,
            model_class=agent.model_class,
        )
        slots.append(
            TemplateSlot(name=agent.header.agent_id, default_agent_id=agent_id)
        )
    reg.register_template(TemplateTree(name="fleet", slots=slots))
    reg.create_compilation_config(
        CompilationConfig(name="prod", template_name="fleet")
    )
    return reg
