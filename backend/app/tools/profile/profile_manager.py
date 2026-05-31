"""Profile Manager — categories, discoveries, hypotheses, observations, runs."""

from __future__ import annotations

import asyncio
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field


class Input(BaseModel):
    command: str = Field(
        description="Categories: list-categories|get-category; "
        "Discoveries: create-discovery|list-discoveries|update-discovery|confirm-discovery|invalidate-discovery|search-discoveries; "
        "Hypotheses: create-hypothesis|list-hypotheses|update-hypothesis|validate-hypothesis|promote-hypothesis|disprove-hypothesis|get-for-validation; "
        "Observations: add-observation|list-observations|increment-observation|promote-observation; "
        "Runs: start-run|complete-run|get-last-run|list-runs; "
        "Knowledge: get-knowledge|compile-knowledge|get-summary"
    )
    # IDs (UUIDs)
    discovery_id: str = Field(default="", description="Discovery ID (UUID)")
    hypothesis_id: str = Field(default="", description="Hypothesis ID (UUID)")
    observation_id: str = Field(default="", description="Observation ID (UUID)")
    run_id: str = Field(default="", description="Analysis run ID (UUID)")
    category_id: str = Field(default="", description="Category ID (UUID)")
    category_name: str = Field(default="", description="Category name")
    # Content fields
    text: str = Field(default="", description="Discovery/hypothesis/observation text")
    confidence: float = Field(default=-1.0, description="Confidence score (0.0-1.0)")
    status: str = Field(default="", description="Status filter or value")
    # Observation fields
    pattern_type: str = Field(default="", description="temporal|behavioral|emotional|relational")
    source_type: str = Field(default="", description="chat|journal")
    source_reference: str = Field(default="", description="Source reference")
    # Run fields
    run_type: str = Field(default="", description="periodic|manual|focused")
    focus_area: str = Field(default="", description="Focus area for analysis run")
    discoveries_found: int = Field(default=0, description="Discoveries found in run")
    hypotheses_generated: int = Field(default=0, description="Hypotheses generated")
    hypotheses_validated: int = Field(default=0, description="Hypotheses validated")
    notes: str = Field(default="", description="Notes")
    # Knowledge
    min_confidence: float = Field(default=0.6, description="Minimum confidence for knowledge")
    compact: bool = Field(default=False, description="Compact output format")
    limit: int = Field(default=50, description="Result limit")
    # Content classification
    sensitivity: str = Field(default="public", description="Sensitivity: public|nsfw|secret")
    public_summary: str = Field(default="", description="Sanitized summary for external models")
    clearance: str = Field(default="", description="Clearance level: full (see all) or public (filter sensitive)")


class Output(BaseModel):
    success: bool = True
    item: dict = Field(default_factory=dict)
    items: list[dict] = Field(default_factory=list)
    count: int = 0
    summary: str = ""
    error: str = ""


class ProfileManagerTool(ScriptTool[Input, Output]):
    name = "profile_manager"
    description = "Manage user profile analysis: categories, discoveries, hypotheses, observations, and analysis runs"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    def _get_clearance(self, inp: Input) -> str:
        """Resolve clearance from input or FREN_CLEARANCE env var."""
        import os

        if inp.clearance:
            return inp.clearance
        return os.environ.get("FREN_CLEARANCE", "full")

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.profile import ProfileRepo

        repo = ProfileRepo()
        cmd = inp.command
        clearance = self._get_clearance(inp)

        # ── Categories ──
        if cmd == "list-categories":
            cats = await repo.list_categories()
            return Output(success=True, items=cats, count=len(cats))

        if cmd == "get-category":
            cat = await repo.get_category(inp.category_name)
            if not cat:
                return Output(success=False, error=f"Category not found: {inp.category_name}")
            return Output(success=True, item=cat)

        # ── Discoveries ──
        if cmd == "create-discovery":
            d = await repo.create_discovery(
                inp.text,
                category_id=inp.category_id or None,
                confidence=inp.confidence if inp.confidence >= 0 else 0.5,
                sensitivity=inp.sensitivity,
                public_summary=inp.public_summary or None,
            )
            return Output(success=True, item=d)

        if cmd == "list-discoveries":
            ds = await repo.list_discoveries(
                status=inp.status or "active",
                category_id=inp.category_id or None,
                limit=inp.limit,
                clearance=clearance,
            )
            return Output(success=True, items=ds, count=len(ds))

        if cmd == "update-discovery":
            fields: dict[str, Any] = {}
            if inp.text:
                fields["discovery"] = inp.text
            if inp.confidence >= 0:
                fields["confidence"] = inp.confidence
            if inp.status:
                fields["status"] = inp.status
            if inp.sensitivity != "public":
                fields["sensitivity"] = inp.sensitivity
            if inp.public_summary:
                fields["public_summary"] = inp.public_summary
            d = await repo.update_discovery(inp.discovery_id, **fields)
            if not d:
                return Output(success=False, error=f"Discovery not found: {inp.discovery_id}")
            return Output(success=True, item=d)

        if cmd == "confirm-discovery":
            d = await repo.update_discovery(
                inp.discovery_id,
                confidence=min(1.0, (inp.confidence if inp.confidence >= 0 else 0.0) + 0.1),
            )
            if not d:
                return Output(success=False, error=f"Discovery not found: {inp.discovery_id}")
            return Output(success=True, item=d)

        if cmd == "invalidate-discovery":
            d = await repo.update_discovery(inp.discovery_id, status="invalidated")
            if not d:
                return Output(success=False, error=f"Discovery not found: {inp.discovery_id}")
            return Output(success=True, item=d)

        if cmd == "search-discoveries":
            ds = await repo.list_discoveries(status="active", limit=inp.limit)
            query = inp.text.lower()
            matched = [d for d in ds if query in str(d.get("discovery", "")).lower()]
            return Output(success=True, items=matched, count=len(matched))

        # ── Hypotheses ──
        if cmd == "create-hypothesis":
            h = await repo.create_hypothesis(
                inp.text,
                category_id=inp.category_id or None,
                confidence_score=inp.confidence if inp.confidence >= 0 else 0.0,
                sensitivity=inp.sensitivity,
            )
            return Output(success=True, item=h)

        if cmd == "list-hypotheses":
            hs = await repo.list_hypotheses(
                status=inp.status or None,
                limit=inp.limit,
                clearance=clearance,
            )
            return Output(success=True, items=hs, count=len(hs))

        if cmd == "update-hypothesis":
            fields = {}
            if inp.text:
                fields["hypothesis"] = inp.text
            if inp.confidence >= 0:
                fields["confidence_score"] = inp.confidence
            if inp.status:
                fields["status"] = inp.status
            h = await repo.update_hypothesis(inp.hypothesis_id, **fields)
            if not h:
                return Output(success=False, error=f"Hypothesis not found: {inp.hypothesis_id}")
            return Output(success=True, item=h)

        if cmd == "validate-hypothesis":
            h = await repo.update_hypothesis(inp.hypothesis_id, status="validating")
            if not h:
                return Output(success=False, error=f"Hypothesis not found: {inp.hypothesis_id}")
            return Output(success=True, item=h)

        if cmd == "promote-hypothesis":
            h = await repo.update_hypothesis(inp.hypothesis_id, status="confirmed")
            if not h:
                return Output(success=False, error=f"Hypothesis not found: {inp.hypothesis_id}")
            # Create a discovery from the confirmed hypothesis
            d = await repo.create_discovery(
                str(h.get("hypothesis", "")),
                category_id=str(h["category_id"]) if h.get("category_id") else None,
                confidence=float(h.get("confidence_score", 0.7)),
            )
            return Output(success=True, item=d)

        if cmd == "disprove-hypothesis":
            h = await repo.update_hypothesis(inp.hypothesis_id, status="disproven")
            if not h:
                return Output(success=False, error=f"Hypothesis not found: {inp.hypothesis_id}")
            return Output(success=True, item=h)

        if cmd == "get-for-validation":
            hs = await repo.list_hypotheses(status="pending", limit=inp.limit)
            return Output(success=True, items=hs, count=len(hs))

        # ── Observations ──
        if cmd == "add-observation":
            o = await repo.create_observation(
                inp.text,
                pattern_type=inp.pattern_type or None,
                source_type=inp.source_type or None,
                source_reference=inp.source_reference or None,
                sensitivity=inp.sensitivity,
            )
            return Output(success=True, item=o)

        if cmd == "list-observations":
            obs = await repo.list_observations(
                pattern_type=inp.pattern_type or None,
                limit=inp.limit,
                clearance=clearance,
            )
            return Output(success=True, items=obs, count=len(obs))

        if cmd == "increment-observation":
            # Increment occurrence count via raw update
            from app.db.session import fetch_one as _fo
            from app.db.session import get_async_session as _gs

            async with _gs() as s:
                o = await _fo(
                    s,
                    """
                    UPDATE pattern_observations
                    SET occurrence_count = occurrence_count + 1, last_seen_at = NOW()
                    WHERE id = :oid::uuid RETURNING *
                """,
                    {"oid": inp.observation_id},
                )
            if not o:
                return Output(success=False, error=f"Observation not found: {inp.observation_id}")
            return Output(success=True, item=o)

        if cmd == "promote-observation":
            # Promote observation to hypothesis
            from app.db.session import fetch_one as _fo
            from app.db.session import get_async_session as _gs

            async with _gs() as s:
                o = await _fo(
                    s, "SELECT * FROM pattern_observations WHERE id = :oid::uuid", {"oid": inp.observation_id}
                )
            if not o:
                return Output(success=False, error=f"Observation not found: {inp.observation_id}")
            h = await repo.create_hypothesis(
                str(o.get("observation", "")),
                category_id=inp.category_id or None,
            )
            return Output(success=True, item=h)

        # ── Analysis Runs ──
        if cmd == "start-run":
            r = await repo.create_run(
                inp.run_type or "manual",
                focus_area=inp.focus_area or None,
            )
            return Output(success=True, item=r)

        if cmd == "complete-run":
            counters: dict[str, Any] = {}
            if inp.discoveries_found:
                counters["discoveries_found"] = inp.discoveries_found
            if inp.hypotheses_generated:
                counters["hypotheses_generated"] = inp.hypotheses_generated
            if inp.hypotheses_validated:
                counters["hypotheses_validated"] = inp.hypotheses_validated
            if inp.notes:
                counters["notes"] = inp.notes
            r = await repo.complete_run(inp.run_id, **counters)
            if not r:
                return Output(success=False, error=f"Run not found: {inp.run_id}")
            return Output(success=True, item=r)

        if cmd == "get-last-run":
            from app.db.session import fetch_one as _fo
            from app.db.session import get_async_session as _gs

            conds = ["1=1"]
            params: dict[str, Any] = {}
            if inp.run_type:
                conds.append("run_type = :rt")
                params["rt"] = inp.run_type
            async with _gs() as s:
                r = await _fo(
                    s,
                    f"""
                    SELECT * FROM analysis_runs WHERE {" AND ".join(conds)}
                    ORDER BY started_at DESC LIMIT 1
                """,
                    params,
                )
            if not r:
                return Output(success=False, error="No analysis runs found")
            return Output(success=True, item=r)

        if cmd == "list-runs":
            from app.db.session import fetch_all as _fa
            from app.db.session import get_async_session as _gs

            conds = ["1=1"]
            params2: dict[str, Any] = {"limit": inp.limit}
            if inp.status:
                conds.append("status = :st")
                params2["st"] = inp.status
            async with _gs() as s:
                runs = await _fa(
                    s,
                    f"""
                    SELECT * FROM analysis_runs WHERE {" AND ".join(conds)}
                    ORDER BY started_at DESC LIMIT :limit
                """,
                    params2,
                )
            return Output(success=True, items=runs, count=len(runs))

        # ── Knowledge Compilation ──
        if cmd == "get-knowledge":
            ds = await repo.list_discoveries(
                status="active",
                limit=200,
                clearance=clearance,
            )
            filtered = [d for d in ds if float(d.get("confidence", 0)) >= inp.min_confidence]
            if inp.compact:
                filtered = [
                    {k: v for k, v in d.items() if k in ("discovery", "confidence", "category_id")} for d in filtered
                ]
            return Output(success=True, items=filtered, count=len(filtered))

        if cmd == "compile-knowledge":
            cats = await repo.list_categories()
            ds = await repo.list_discoveries(status="active", limit=200, clearance=clearance)
            filtered = [d for d in ds if float(d.get("confidence", 0)) >= inp.min_confidence]
            lines = ["# Profile Knowledge\n"]
            for cat in cats:
                cat_id = str(cat.get("id", ""))
                cat_ds = [d for d in filtered if str(d.get("category_id", "")) == cat_id]
                if cat_ds:
                    lines.append(f"\n## {cat.get('name', 'Unknown')}")
                    for d in cat_ds:
                        lines.append(f"- [{d.get('confidence', 0):.0%}] {d.get('discovery', '')}")
            return Output(success=True, summary="\n".join(lines), count=len(filtered))

        if cmd == "get-summary":
            cats = await repo.list_categories()
            ds = await repo.list_discoveries(status="active", limit=500)
            hs = await repo.list_hypotheses(limit=500)
            obs = await repo.list_observations(limit=500)
            summary = {
                "categories": len(cats),
                "discoveries_active": len(ds),
                "hypotheses_total": len(hs),
                "hypotheses_pending": len([h for h in hs if h.get("status") == "pending"]),
                "hypotheses_confirmed": len([h for h in hs if h.get("status") == "confirmed"]),
                "observations": len(obs),
            }
            return Output(success=True, item=summary)

        return Output(success=False, error=f"Unknown command: {cmd}")


if __name__ == "__main__":
    ProfileManagerTool.run()
