"""Agent domains — single registration point.

Each domain module exposes `agents() -> list[AgentDefinition]` and optionally
`branches() -> list[BranchTest]`. Add a newly ported domain to `ALL_DOMAINS`
below and the registry, compiler, and improvement harness all pick it up.
"""

from __future__ import annotations

from types import ModuleType

from app.agents.domains import (
    food,
    goals,
    persona,
    profile,
    research,
    server,
    support,
)

# Append domains here as they are ported from v3.
ALL_DOMAINS: list[ModuleType] = [
    persona,
    goals,
    food,
    profile,
    server,
    research,
    support,
]


def all_agent_defs():
    """Every AgentDefinition across all domains, de-duplicated by agent_id."""
    out = []
    seen: set[str] = set()
    for domain in ALL_DOMAINS:
        for agent in domain.agents():
            aid = agent.header.agent_id
            if aid in seen:
                raise ValueError(f"duplicate agent_id across domains: {aid!r}")
            seen.add(aid)
            out.append(agent)
    return out


def all_branch_tests():
    """Every BranchTest contributed by any domain that defines branches()."""
    out = []
    for domain in ALL_DOMAINS:
        fn = getattr(domain, "branches", None)
        if callable(fn):
            out.extend(fn())
    return out
