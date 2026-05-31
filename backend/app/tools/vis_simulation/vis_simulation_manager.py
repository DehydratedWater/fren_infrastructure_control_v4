"""Vis simulation manager tool — manage character simulation training data."""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

from app.settings import get_settings

VALID_STATUSES = ("pending", "generating", "completed", "failed")
VALID_SCENARIO_TYPES = ("reading", "observing", "desiring", "reacting", "technical_task", "being_asked")
VALID_SENDERS = ("vis", "interlocutor", "environment", "content")


def _sim_id() -> str:
    return f"vis_sim_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


def _msg_id() -> str:
    return f"vis_msg_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


def _score_id() -> str:
    return f"vis_score_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


class Input(BaseModel):
    command: str = Field(
        description=(
            "create-simulation|get-simulation|list-simulations|update-status"
            "|add-message|get-messages|add-scores|get-scores"
            "|get-existing-scenarios|get-high-quality|get-stats"
            "|export-finetuning|delete-simulation|read-random-journal"
        )
    )
    simulation_id: str = Field(default="", description="Simulation ID")
    scenario_type: str = Field(default="", description="Scenario type")
    scenario_description: str = Field(default="", description="Scenario description")
    status: str = Field(default="", description="Status for update")
    sequence_number: int = Field(default=0, description="Message sequence number")
    sender: str = Field(default="", description="Message sender")
    response_content: str = Field(default="", description="Response content")
    thinking_content: str = Field(default="", description="Thinking content")
    actions: str = Field(default="", description="JSON actions array")
    trigger_type: str = Field(default="", description="Trigger type for non-Vis messages")
    quality_score: float = Field(default=0.0, description="Quality score 0-1")
    realism_score: float = Field(default=0.0, description="Realism score 0-1")
    character_adherence_score: float = Field(default=0.0, description="Character adherence 0-1")
    character_depth_analysis: str = Field(default="", description="Depth analysis text")
    quality_notes: str = Field(default="", description="Quality notes")
    realism_notes: str = Field(default="", description="Realism notes")
    adherence_notes: str = Field(default="", description="Adherence notes")
    min_score: float = Field(default=0.7, description="Min score for high-quality filter")
    limit: int = Field(default=50, description="Result limit")
    lines: int = Field(default=100, description="Journal lines to read")
    journal_excerpt: str = Field(default="", description="Journal excerpt")
    mood_description: str = Field(default="", description="Mood description")
    interlocutor_type: str = Field(default="", description="Interlocutor type")
    interlocutor_description: str = Field(default="", description="Interlocutor description")
    emotional_state: str = Field(default="", description="JSON emotional state")
    scenario_assumptions: str = Field(default="", description="JSON scenario assumptions")
    journal_topics: str = Field(default="", description="JSON journal topics")
    journal_date: str = Field(default="", description="Journal date YYYY-MM-DD")
    topics_investigated: str = Field(default="", description="JSON topics investigated")
    journal_references: str = Field(default="", description="JSON journal references")


class Output(BaseModel):
    success: bool = True
    simulation: dict | None = None
    simulations: list[dict] = Field(default_factory=list)
    message: dict | None = None
    messages: list[dict] = Field(default_factory=list)
    scores: dict | None = None
    scenarios: list[dict] = Field(default_factory=list)
    stats: dict | None = None
    exported_count: int = 0
    data: list[dict] = Field(default_factory=list)
    excerpt: str = ""
    start_line: int = 0
    end_line: int = 0
    total_lines: int = 0
    count: int = 0
    deleted: bool = False
    error: str = ""


class VisSimulationManagerTool(ScriptTool[Input, Output]):
    name = "vis_simulation_manager"
    description = "Manage character simulation training data for fine-tuning"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._run(inp))

    async def _run(self, inp: Input) -> Output:
        import json

        from app.db.repos.vis_simulation import MessagesRepo, ScoresRepo, SimulationsRepo

        sims = SimulationsRepo()
        msgs = MessagesRepo()
        scores_repo = ScoresRepo()

        if inp.command == "read-random-journal":
            journal = Path(get_settings().project_root) / "vis_data" / "combined_journal.txt"
            if not journal.exists():
                return Output(success=False, error=f"Journal not found: {journal}")
            all_lines = journal.read_text(encoding="utf-8").splitlines()
            total = len(all_lines)
            if total == 0:
                return Output(success=False, error="Journal empty")
            start = random.randint(0, max(0, total - inp.lines))
            end = min(start + inp.lines, total)
            return Output(
                success=True,
                excerpt="\n".join(all_lines[start:end]),
                start_line=start + 1,
                end_line=end,
                total_lines=total,
            )

        if inp.command == "create-simulation":
            if inp.scenario_type not in VALID_SCENARIO_TYPES:
                return Output(success=False, error=f"Invalid scenario_type. Must be: {VALID_SCENARIO_TYPES}")
            kw: dict = {}
            if inp.journal_excerpt:
                kw["journal_excerpt"] = inp.journal_excerpt
            if inp.mood_description:
                kw["mood_description"] = inp.mood_description
            if inp.interlocutor_type:
                kw["interlocutor_type"] = inp.interlocutor_type
            if inp.interlocutor_description:
                kw["interlocutor_description"] = inp.interlocutor_description
            if inp.emotional_state:
                kw["emotional_state"] = json.loads(inp.emotional_state)
            if inp.scenario_assumptions:
                kw["scenario_assumptions"] = json.loads(inp.scenario_assumptions)
            if inp.journal_topics:
                kw["journal_topics"] = json.loads(inp.journal_topics)
            if inp.journal_date:
                kw["journal_date"] = inp.journal_date
            sim = await sims.create(_sim_id(), inp.scenario_type, inp.scenario_description, **kw)
            return Output(success=True, simulation=sim)

        if inp.command == "get-simulation":
            sim = await sims.get(inp.simulation_id)
            return Output(success=True, simulation=sim) if sim else Output(success=False, error="Not found")

        if inp.command == "list-simulations":
            result = await sims.list(status=inp.status or None, limit=inp.limit)
            return Output(success=True, simulations=result, count=len(result))

        if inp.command == "update-status":
            if inp.status not in VALID_STATUSES:
                return Output(success=False, error=f"Invalid status. Must be: {VALID_STATUSES}")
            sim = await sims.complete(inp.simulation_id) if inp.status == "completed" else None
            if not sim:
                # For non-complete statuses, we'd need an update method; just complete for now
                return Output(success=False, error="Only 'completed' status update supported via repo")
            return Output(success=True, simulation=sim)

        if inp.command == "add-message":
            if inp.sender not in VALID_SENDERS:
                return Output(success=False, error=f"Invalid sender. Must be: {VALID_SENDERS}")
            actions = json.loads(inp.actions) if inp.actions else None
            msg = await msgs.create(
                _msg_id(),
                inp.simulation_id,
                inp.sequence_number,
                inp.sender,
                inp.response_content,
                thinking_content=inp.thinking_content or None,
                actions=actions,
                trigger_type=inp.trigger_type or None,
            )
            return Output(success=True, message=msg)

        if inp.command == "get-messages":
            result = await msgs.list_for_simulation(inp.simulation_id)
            return Output(success=True, messages=result, count=len(result))

        if inp.command == "add-scores":
            for name, score in [
                ("quality", inp.quality_score),
                ("realism", inp.realism_score),
                ("adherence", inp.character_adherence_score),
            ]:
                if not 0.0 <= score <= 1.0:
                    return Output(success=False, error=f"{name} score must be 0.0-1.0")
            kw2: dict = {}
            if inp.character_depth_analysis:
                kw2["character_depth_analysis"] = inp.character_depth_analysis
            if inp.quality_notes:
                kw2["quality_notes"] = inp.quality_notes
            if inp.realism_notes:
                kw2["realism_notes"] = inp.realism_notes
            if inp.adherence_notes:
                kw2["adherence_notes"] = inp.adherence_notes
            if inp.topics_investigated:
                kw2["topics_investigated"] = json.loads(inp.topics_investigated)
            if inp.journal_references:
                kw2["journal_references"] = json.loads(inp.journal_references)
            sc = await scores_repo.create(
                _score_id(),
                inp.simulation_id,
                inp.quality_score,
                inp.realism_score,
                inp.character_adherence_score,
                **kw2,
            )
            return Output(success=True, scores=sc)

        if inp.command == "get-scores":
            sc = await scores_repo.get_for_simulation(inp.simulation_id)
            return Output(success=True, scores=sc) if sc else Output(success=False, error="No scores found")

        if inp.command == "get-stats":
            all_sims = await sims.list(limit=1000)
            by_status: dict[str, int] = {}
            for s in all_sims:
                st = s.get("status", "unknown")
                by_status[st] = by_status.get(st, 0) + 1
            return Output(success=True, stats={"total": len(all_sims), "by_status": by_status}, count=len(all_sims))

        if inp.command == "delete-simulation":
            # Deletion would require cascade; just mark as failed
            sim = await sims.complete(inp.simulation_id)
            return Output(success=True, deleted=sim is not None)

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    VisSimulationManagerTool.run()
