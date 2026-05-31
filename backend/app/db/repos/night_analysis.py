"""Night analysis repository -- runs + findings CRUD and search."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


def _format_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


class NightAnalysisRunsRepo:
    async def create(self) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(  # type: ignore[return-value]
                s,
                "INSERT INTO night_analysis_runs DEFAULT VALUES RETURNING *",
                {},
            )

    async def update_progress(
        self,
        run_id: int,
        *,
        phases_completed: int | None = None,
        total_queries: int | None = None,
        total_llm_calls: int | None = None,
        findings_count: int | None = None,
    ) -> dict[str, Any] | None:
        sets = []
        params: dict[str, Any] = {"rid": run_id}
        if phases_completed is not None:
            sets.append("phases_completed = :phases")
            params["phases"] = phases_completed
        if total_queries is not None:
            sets.append("total_queries = :queries")
            params["queries"] = total_queries
        if total_llm_calls is not None:
            sets.append("total_llm_calls = :llm")
            params["llm"] = total_llm_calls
        if findings_count is not None:
            sets.append("findings_count = :fc")
            params["fc"] = findings_count
        if not sets:
            return None
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE night_analysis_runs SET {', '.join(sets)} WHERE id = :rid RETURNING *",
                params,
            )

    async def complete(
        self,
        run_id: int,
        *,
        report_cache_id: str = "",
        findings_count: int = 0,
        total_queries: int = 0,
        total_llm_calls: int = 0,
    ) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE night_analysis_runs
                SET status = 'completed',
                    completed_at = :now,
                    report_cache_id = :rcid,
                    findings_count = :fc,
                    total_queries = :tq,
                    total_llm_calls = :tl,
                    phases_completed = 6
                WHERE id = :rid
                RETURNING *
                """,
                {
                    "rid": run_id,
                    "now": now,
                    "rcid": report_cache_id,
                    "fc": findings_count,
                    "tq": total_queries,
                    "tl": total_llm_calls,
                },
            )

    async def fail(self, run_id: int, error: str) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE night_analysis_runs
                SET status = 'failed', completed_at = :now, error = :err
                WHERE id = :rid RETURNING *
                """,
                {"rid": run_id, "now": now, "err": error},
            )

    async def get_latest(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "SELECT * FROM night_analysis_runs ORDER BY started_at DESC LIMIT 1",
                {},
            )

    async def list_recent(self, *, limit: int = 10) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM night_analysis_runs ORDER BY started_at DESC LIMIT :limit",
                {"limit": limit},
            )


class NightAnalysisFindingsRepo:
    async def create(
        self,
        run_id: int,
        category: str,
        domain: str,
        title: str,
        content: str,
        *,
        analysis_type: str = "",
        confidence: float = 0.5,
        relevance_to_goals: float = 0.5,
        actionable: bool = False,
        data_sources: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "rid": run_id,
            "cat": category,
            "dom": domain,
            "title": title,
            "content": content,
            "atype": analysis_type,
            "conf": confidence,
            "rel": relevance_to_goals,
            "act": actionable,
            "dsrc": data_sources or [],
        }
        if embedding:
            params["embedding"] = _format_vector(embedding)
            sql = """
                INSERT INTO night_analysis_findings
                    (run_id, category, domain, title, content, analysis_type,
                     confidence, relevance_to_goals, actionable, data_sources, embedding)
                VALUES (:rid, :cat, :dom, :title, :content, :atype,
                        :conf, :rel, :act, CAST(:dsrc AS text[]), CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO night_analysis_findings
                    (run_id, category, domain, title, content, analysis_type,
                     confidence, relevance_to_goals, actionable, data_sources)
                VALUES (:rid, :cat, :dom, :title, :content, :atype,
                        :conf, :rel, :act, CAST(:dsrc AS text[]))
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def list_by_run(self, run_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM night_analysis_findings WHERE run_id = :rid ORDER BY confidence DESC LIMIT :limit",
                {"rid": run_id, "limit": limit},
            )

    async def list_by_domain(self, domain: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM night_analysis_findings WHERE domain = :dom ORDER BY created_at DESC LIMIT :limit",
                {"dom": domain, "limit": limit},
            )

    async def list_by_category(self, category: str, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM night_analysis_findings WHERE category = :cat ORDER BY created_at DESC LIMIT :limit",
                {"cat": category, "limit": limit},
            )

    async def list_actionable(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM night_analysis_findings WHERE actionable = true ORDER BY relevance_to_goals DESC, created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def search_by_embedding(
        self,
        embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        vec = _format_vector(embedding)
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT *, 1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM night_analysis_findings
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :limit
                """,
                {"vec": vec, "limit": limit},
            )

    async def get_latest_run_findings(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT f.* FROM night_analysis_findings f
                JOIN night_analysis_runs r ON f.run_id = r.id
                WHERE r.status = 'completed'
                ORDER BY r.started_at DESC, f.confidence DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )
