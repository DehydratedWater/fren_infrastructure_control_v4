"""Night Analysis — query findings and run history."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(description="latest-report|list-findings|search-findings|list-runs|list-actionable")
    domain: str = Field(
        default="", description="Filter by domain (health, productivity, habits, goals, emotions, etc.)"
    )
    category: str = Field(
        default="", description="Filter by category (correlation, anomaly, hypothesis, suggestion, trend)"
    )
    query: str = Field(default="", description="Search query for search-findings")
    limit: int = Field(default=20, description="Result limit")


class Output(BaseModel):
    success: bool = True
    result: dict = Field(default_factory=dict)
    findings: list[dict] = Field(default_factory=list)
    runs: list[dict] = Field(default_factory=list)
    count: int = 0
    error: str = ""


class NightAnalysisTool(ScriptTool[Input, Output]):
    name = "night_analysis"
    description = "Query night analysis findings, reports, and run history"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.night_analysis import NightAnalysisFindingsRepo, NightAnalysisRunsRepo

        runs_repo = NightAnalysisRunsRepo()
        findings_repo = NightAnalysisFindingsRepo()
        cmd = inp.command

        if cmd == "latest-report":
            run = await runs_repo.get_latest()
            if not run or run.get("status") != "completed":
                return Output(success=True, result={"message": "No completed analysis run found"})
            findings = await findings_repo.list_by_run(run["id"], limit=inp.limit)
            return Output(
                success=True,
                result={
                    "run_id": run["id"],
                    "started_at": str(run.get("started_at", "")),
                    "completed_at": str(run.get("completed_at", "")),
                    "total_queries": run.get("total_queries", 0),
                    "total_llm_calls": run.get("total_llm_calls", 0),
                    "findings_count": run.get("findings_count", 0),
                    "report_cache_id": run.get("report_cache_id", ""),
                },
                findings=findings,
                count=len(findings),
            )

        if cmd == "list-findings":
            if inp.domain:
                findings = await findings_repo.list_by_domain(inp.domain, limit=inp.limit)
            elif inp.category:
                findings = await findings_repo.list_by_category(inp.category, limit=inp.limit)
            else:
                findings = await findings_repo.get_latest_run_findings(limit=inp.limit)
            return Output(success=True, findings=findings, count=len(findings))

        if cmd == "list-actionable":
            findings = await findings_repo.list_actionable(limit=inp.limit)
            return Output(success=True, findings=findings, count=len(findings))

        if cmd == "search-findings":
            if not inp.query:
                return Output(success=False, error="query required for search-findings")
            from app.services.embeddings import get_embedding

            embedding = get_embedding(inp.query)
            findings = await findings_repo.search_by_embedding(embedding, limit=inp.limit)
            return Output(success=True, findings=findings, count=len(findings))

        if cmd == "list-runs":
            runs = await runs_repo.list_recent(limit=inp.limit)
            return Output(success=True, runs=runs, count=len(runs))

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    NightAnalysisTool.run()
