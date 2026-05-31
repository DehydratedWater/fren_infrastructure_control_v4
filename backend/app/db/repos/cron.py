"""Cron and workflow execution repositories."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class CronExecutionsRepo:
    async def create(
        self,
        execution_id: str,
        mode: str,
        started_at: datetime,
        *,
        triggered_by: str = "cron",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO cron_executions (execution_id, mode, started_at, triggered_by)
                VALUES (:eid, :mode, :sa, :tb)
                RETURNING *
            """,
                {"eid": execution_id, "mode": mode, "sa": started_at, "tb": triggered_by},
            )  # type: ignore[return-value]

    async def complete(
        self,
        execution_id: str,
        *,
        exit_code: int = 0,
        status: str = "completed",
        error_output: str | None = None,
        log_file: str | None = None,
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE cron_executions
                SET completed_at = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM NOW() - started_at),
                    exit_code = :ec, status = :st, error_output = :err,
                    log_file = COALESCE(:log_file, log_file)
                WHERE execution_id = :eid RETURNING *
            """,
                {
                    "eid": execution_id,
                    "ec": exit_code,
                    "st": status,
                    "err": error_output,
                    "log_file": log_file,
                },
            )

    async def reconcile_stale_running(
        self,
        *,
        mode: str | None = None,
        older_than_seconds: int = 600,
        status: str = "abandoned",
        error_output: str = "Recovered stale running execution",
    ) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        params: dict[str, Any] = {
            "cutoff": cutoff,
            "status": status,
            "error_output": error_output,
        }
        where = "WHERE status = 'running' AND started_at < :cutoff"
        if mode:
            where += " AND mode = :mode"
            params["mode"] = mode

        async with get_async_session() as s:
            result = await execute_sql(
                s,
                f"""
                UPDATE cron_executions
                SET completed_at = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM NOW() - started_at),
                    status = :status,
                    exit_code = COALESCE(exit_code, -2),
                    error_output = COALESCE(error_output, :error_output)
                {where}
                """,
                params,
            )
            return result.rowcount

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM cron_executions ORDER BY started_at DESC LIMIT :limit
            """,
                {"limit": limit},
            )


class WorkflowExecutionsRepo:
    async def create(
        self,
        execution_id: str,
        workflow_id: str,
        workflow_name: str,
        started_at: datetime,
        *,
        input_text: str | None = None,
        triggered_by: str = "manual",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO workflow_executions (execution_id, workflow_id, workflow_name,
                    input_text, triggered_by, started_at)
                VALUES (:eid, :wid, :wname, :inp, :tb, :sa)
                RETURNING *
            """,
                {
                    "eid": execution_id,
                    "wid": workflow_id,
                    "wname": workflow_name,
                    "inp": input_text,
                    "tb": triggered_by,
                    "sa": started_at,
                },
            )  # type: ignore[return-value]

    async def complete(
        self, execution_id: str, *, exit_code: int = 0, output: str | None = None, error: str | None = None
    ) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE workflow_executions
                SET completed_at = NOW(),
                    duration_seconds = EXTRACT(EPOCH FROM NOW() - started_at),
                    exit_code = :ec, status = CASE WHEN :ec = 0 THEN 'completed' ELSE 'failed' END,
                    output = :out, error = :err
                WHERE execution_id = :eid RETURNING *
            """,
                {"eid": execution_id, "ec": exit_code, "out": output, "err": error},
            )

    async def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM workflow_executions ORDER BY started_at DESC LIMIT :limit
            """,
                {"limit": limit},
            )
