"""Distinguished branches — multi-agent paths that get their own tests + tuning.

A branch is an orchestrator + the dispatch chain it must drive to handle a
class of task. These are tested (path-order) AND optimised as units, separately
from the leaf agents — per the "not only individual agents but also important
distinguished agent branches" requirement.

Every orchestrator in the fleet contributes at least one branch here (the
"all orchestrators" coverage decision). New branches are added as their domains
are ported.
"""

from __future__ import annotations

from src import BranchTest, SubstringEvaluator

# persona/orchestrator: a substantive message must flow
# context → thinking → responding (never skip analysis).
PERSONA_SUBSTANTIVE = BranchTest(
    name="persona/orchestrator::substantive-flow",
    entry_agent="persona/orchestrator",
    prompt="Help me plan my week around my fitness goal.",
    path=("context_analyzer", "persona/thinking", "persona/responding"),
    evaluators=(SubstringEvaluator(needle="plan", case_sensitive=False),),
)

# A trivial greeting should short-circuit to quick_ack (and NOT drag in the
# whole analysis chain).
PERSONA_TRIVIAL = BranchTest(
    name="persona/orchestrator::trivial-ack",
    entry_agent="persona/orchestrator",
    prompt="thanks!",
    path=("persona/quick_ack",),
)


def branches() -> list[BranchTest]:
    return [PERSONA_SUBSTANTIVE, PERSONA_TRIVIAL]
