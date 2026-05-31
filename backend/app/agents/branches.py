"""Distinguished branches — aggregated from every domain.

A branch is an orchestrator + the dispatch chain it must drive. Each domain
defines its own `branches()`; this module just collects them so the improvement
harness has one fleet-wide list. Per the "all orchestrators" coverage decision,
every orchestrator contributes at least one branch.
"""

from __future__ import annotations

from app.agents.domains import all_branch_tests
from src import BranchTest


def branches() -> list[BranchTest]:
    return all_branch_tests()
