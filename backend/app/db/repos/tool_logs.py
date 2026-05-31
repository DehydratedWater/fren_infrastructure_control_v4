"""Tool execution logging repository."""

from __future__ import annotations

import json
import os
import random
from typing import Any

from app.db.session import execute_sql, fetch_all, get_async_session


def _summarize(data: dict, max_len: int = 500) -> str:
    """JSON-dump with string values truncated to max_len."""

    def _trunc(v: Any) -> Any:
        if isinstance(v, str) and len(v) > max_len:
            return v[:max_len] + "..."
        if isinstance(v, dict):
            return {k: _trunc(val) for k, val in v.items() if val is not None and val != ""}
        if isinstance(v, list):
            return [_trunc(i) for i in v[:20]]
        return v

    cleaned = {k: _trunc(v) for k, v in data.items() if v is not None and v != "" and v != []}
    return json.dumps(cleaned, default=str, ensure_ascii=False)


class ToolLogsRepo:
    async def log_execution(
        self,
        *,
        tool_name: str,
        input_data: dict | None,
        output_data: dict | None,
        success: bool,
        error: str | None,
        duration_ms: int,
    ) -> None:
        command = input_data.get("command") if input_data else None
        agent_name = os.environ.get("FREN_AGENT_NAME", "") or ""
        session_id = os.environ.get("OPENCODE_SESSION_ID", "") or ""

        input_summary = _summarize(input_data) if input_data else None
        output_summary = _summarize(output_data) if output_data else None

        async with get_async_session() as s:
            await execute_sql(
                s,
                """
                INSERT INTO tool_execution_logs
                    (tool_name, command, agent_name, session_id,
                     input_summary, output_summary, success, error_message, duration_ms)
                VALUES
                    (:tool_name, :command, :agent_name, :session_id,
                     :input_summary, :output_summary, :success, :error_message, :duration_ms)
                """,
                {
                    "tool_name": tool_name,
                    "command": command or None,
                    "agent_name": agent_name or None,
                    "session_id": session_id or None,
                    "input_summary": input_summary,
                    "output_summary": output_summary,
                    "success": success,
                    "error_message": error,
                    "duration_ms": duration_ms,
                },
            )

            # Auto-prune ~1% of calls
            if random.random() < 0.01:
                await execute_sql(
                    s,
                    "DELETE FROM tool_execution_logs WHERE created_at < NOW() - INTERVAL '7 days'",
                )

    async def list_recent(
        self, *, tool_name: str | None = None, hours: int = 24, limit: int = 50
    ) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            if tool_name:
                return await fetch_all(
                    s,
                    """
                    SELECT * FROM tool_execution_logs
                    WHERE tool_name = :tool_name
                      AND created_at > NOW() - CAST(:hours AS INTEGER) * INTERVAL '1 hour'
                    ORDER BY created_at DESC LIMIT :limit
                    """,
                    {"tool_name": tool_name, "hours": hours, "limit": limit},
                )
            return await fetch_all(
                s,
                """
                SELECT * FROM tool_execution_logs
                WHERE created_at > NOW() - CAST(:hours AS INTEGER) * INTERVAL '1 hour'
                ORDER BY created_at DESC LIMIT :limit
                """,
                {"hours": hours, "limit": limit},
            )

    async def list_errors(self, *, hours: int = 24, limit: int = 20) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM tool_execution_logs
                WHERE NOT success
                  AND created_at > NOW() - CAST(:hours AS INTEGER) * INTERVAL '1 hour'
                ORDER BY created_at DESC LIMIT :limit
                """,
                {"hours": hours, "limit": limit},
            )

    async def get_tool_stats(self, *, hours: int = 24) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT tool_name,
                       COUNT(*) AS total_calls,
                       ROUND(AVG(duration_ms)) AS avg_duration_ms,
                       COUNT(*) FILTER (WHERE NOT success) AS error_count
                FROM tool_execution_logs
                WHERE created_at > NOW() - CAST(:hours AS INTEGER) * INTERVAL '1 hour'
                GROUP BY tool_name
                ORDER BY total_calls DESC
                """,
                {"hours": hours},
            )
