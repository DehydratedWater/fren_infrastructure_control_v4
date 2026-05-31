"""Techtree repositories — commits, interests, analysis runs, state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class TechtreeCommitRepo:
    async def create(
        self,
        commit_sha: str,
        author_name: str,
        author_email: str,
        committed_at: datetime,
        message: str,
        *,
        files_changed: int = 0,
        insertions: int = 0,
        deletions: int = 0,
        areas: list[str] | None = None,
        analysis: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO techtree_tracked_commits
                    (commit_sha, author_name, author_email, committed_at,
                     message, files_changed, insertions, deletions, areas, analysis)
                VALUES
                    (:sha, :author, :email, :committed_at,
                     :msg, :files, :ins, :dels, :areas, :analysis)
                RETURNING *
                """,
                {
                    "sha": commit_sha,
                    "author": author_name,
                    "email": author_email,
                    "committed_at": committed_at,
                    "msg": message,
                    "files": files_changed,
                    "ins": insertions,
                    "dels": deletions,
                    "areas": json.dumps(areas or []),
                    "analysis": analysis,
                },
            )  # type: ignore[return-value]

    async def get(self, commit_sha: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s, "SELECT * FROM techtree_tracked_commits WHERE commit_sha = :sha", {"sha": commit_sha}
            )

    async def exists(self, commit_sha: str) -> bool:
        async with get_async_session() as s:
            row = await fetch_one(
                s, "SELECT 1 FROM techtree_tracked_commits WHERE commit_sha = :sha", {"sha": commit_sha}
            )
            return row is not None

    async def list_recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM techtree_tracked_commits ORDER BY committed_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def list_unnotified(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM techtree_tracked_commits WHERE notified = FALSE ORDER BY committed_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def mark_notified(self, commit_shas: list[str]) -> int:
        if not commit_shas:
            return 0
        async with get_async_session() as s:
            # Use ANY(:shas) with array parameter
            r = await execute_sql(
                s,
                "UPDATE techtree_tracked_commits SET notified = TRUE WHERE commit_sha = ANY(:shas)",
                {"shas": commit_shas},
            )
            return r.rowcount

    async def list_by_author(self, author_name: str, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM techtree_tracked_commits
                WHERE author_name ILIKE :name
                ORDER BY committed_at DESC LIMIT :limit
                """,
                {"name": f"%{author_name}%", "limit": limit},
            )

    async def list_by_area(self, area: str, *, limit: int = 50) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM techtree_tracked_commits
                WHERE areas @> CAST(:area AS jsonb)
                ORDER BY committed_at DESC LIMIT :limit
                """,
                {"area": json.dumps([area]), "limit": limit},
            )

    async def update_analysis(
        self, commit_sha: str, analysis: str, areas: list[str] | None = None
    ) -> dict[str, Any] | None:
        sets = ["analysis = :analysis"]
        params: dict[str, Any] = {"sha": commit_sha, "analysis": analysis}
        if areas is not None:
            sets.append("areas = :areas")
            params["areas"] = json.dumps(areas)
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE techtree_tracked_commits SET {', '.join(sets)} WHERE commit_sha = :sha RETURNING *",
                params,
            )

    async def get_stats(self, *, days: int = 30) -> dict[str, Any]:
        async with get_async_session() as s:
            total = await fetch_one(
                s,
                """
                SELECT COUNT(*) as total_commits,
                       COALESCE(SUM(insertions), 0) as total_insertions,
                       COALESCE(SUM(deletions), 0) as total_deletions,
                       COALESCE(SUM(files_changed), 0) as total_files
                FROM techtree_tracked_commits
                WHERE committed_at > NOW() - CAST(:days || ' days' AS interval)
                """,
                {"days": str(days)},
            )
            by_author = await fetch_all(
                s,
                """
                SELECT author_name, COUNT(*) as commits,
                       COALESCE(SUM(insertions), 0) as insertions,
                       COALESCE(SUM(deletions), 0) as deletions
                FROM techtree_tracked_commits
                WHERE committed_at > NOW() - CAST(:days || ' days' AS interval)
                GROUP BY author_name ORDER BY commits DESC
                """,
                {"days": str(days)},
            )
            return {
                "days": days,
                "totals": total or {},
                "by_author": by_author,
            }


class TechtreeInterestRepo:
    async def create(
        self,
        interest_id: str,
        name: str,
        *,
        description: str = "",
        paths: list[str] | None = None,
        keywords: list[str] | None = None,
        owner: str = "",
        enabled: bool = True,
        priority: int = 50,
        instructions: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO techtree_interests
                    (interest_id, name, description, paths, keywords, owner, enabled, priority, instructions)
                VALUES
                    (:iid, :name, :desc, :paths, :kw, :owner, :enabled, :priority, :instructions)
                RETURNING *
                """,
                {
                    "iid": interest_id,
                    "name": name,
                    "desc": description,
                    "paths": json.dumps(paths or []),
                    "kw": json.dumps(keywords or []),
                    "owner": owner,
                    "enabled": enabled,
                    "priority": priority,
                    "instructions": instructions,
                },
            )  # type: ignore[return-value]

    async def get(self, interest_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM techtree_interests WHERE interest_id = :iid", {"iid": interest_id})

    async def list_all(self, *, limit: int = 100) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM techtree_interests ORDER BY priority ASC, created_at ASC LIMIT :limit",
                {"limit": limit},
            )

    async def list_enabled(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM techtree_interests WHERE enabled = TRUE ORDER BY priority ASC",
                {},
            )

    async def update(self, interest_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"iid": interest_id}
        idx = 1
        for k, v in fields.items():
            if v is not None:
                pk = f"p{idx}"
                if k in ("paths", "keywords") and isinstance(v, list):
                    params[pk] = json.dumps(v)
                else:
                    params[pk] = v
                sets.append(f"{k} = :{pk}")
                idx += 1
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"UPDATE techtree_interests SET {', '.join(sets)} WHERE interest_id = :iid RETURNING *",
                params,
            )

    async def delete(self, interest_id: str) -> bool:
        async with get_async_session() as s:
            r = await execute_sql(
                s, "DELETE FROM techtree_interests WHERE interest_id = :iid RETURNING id", {"iid": interest_id}
            )
            return r.fetchone() is not None

    async def toggle(self, interest_id: str, enabled: bool) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                "UPDATE techtree_interests SET enabled = :enabled, updated_at = NOW() WHERE interest_id = :iid RETURNING *",
                {"iid": interest_id, "enabled": enabled},
            )


class TechtreeAnalysisRunRepo:
    async def create(
        self,
        run_id: str,
        run_type: str,
        *,
        commits_analyzed: list[str] | None = None,
        summary: str = "",
        feature_suggestions: list[dict] | None = None,
        code_trends: dict | None = None,
        notified_via: str = "",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO techtree_analysis_runs
                    (run_id, run_type, commits_analyzed, summary, feature_suggestions, code_trends, notified_via)
                VALUES
                    (:rid, :rtype, :commits, :summary, :suggestions, :trends, :notified)
                RETURNING *
                """,
                {
                    "rid": run_id,
                    "rtype": run_type,
                    "commits": json.dumps(commits_analyzed or []),
                    "summary": summary,
                    "suggestions": json.dumps(feature_suggestions or []),
                    "trends": json.dumps(code_trends or {}),
                    "notified": notified_via,
                },
            )  # type: ignore[return-value]

    async def get(self, run_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM techtree_analysis_runs WHERE run_id = :rid", {"rid": run_id})

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                "SELECT * FROM techtree_analysis_runs ORDER BY created_at DESC LIMIT :limit",
                {"limit": limit},
            )

    async def get_latest(self) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM techtree_analysis_runs ORDER BY created_at DESC LIMIT 1", {})


class TechtreeStateRepo:
    async def get(self, state_key: str) -> str | None:
        async with get_async_session() as s:
            row = await fetch_one(
                s, "SELECT state_value FROM techtree_state WHERE state_key = :key", {"key": state_key}
            )
            return row["state_value"] if row else None

    async def set(self, state_key: str, state_value: str) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO techtree_state (state_key, state_value, updated_at)
                VALUES (:key, :val, NOW())
                ON CONFLICT (state_key) DO UPDATE SET state_value = :val, updated_at = NOW()
                RETURNING *
                """,
                {"key": state_key, "val": state_value},
            )  # type: ignore[return-value]

    async def list_all(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM techtree_state ORDER BY state_key", {})
