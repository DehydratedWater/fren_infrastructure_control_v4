"""Goal auto-updater repository — state and logs."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class GoalAutoUpdaterRepo:
    async def get_state(self) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM goal_auto_updater_state WHERE id = 1")

    async def update_state(
        self,
        *,
        last_processed_activity_id: int | None = None,
        last_processed_event_id: int | None = None,
        last_processed_habit_occurrence_id: int | None = None,
        last_processed_todo_id: int | None = None,
        total_updates_made: int | None = None,
    ) -> dict[str, Any]:
        sets = ["last_run_at = NOW()", "updated_at = NOW()"]
        params: dict[str, Any] = {}

        if last_processed_activity_id is not None:
            sets.append("last_processed_activity_id = :last_activity")
            params["last_activity"] = last_processed_activity_id
        if last_processed_event_id is not None:
            sets.append("last_processed_event_id = :last_event")
            params["last_event"] = last_processed_event_id
        if last_processed_habit_occurrence_id is not None:
            sets.append("last_processed_habit_occurrence_id = :last_habit")
            params["last_habit"] = last_processed_habit_occurrence_id
        if last_processed_todo_id is not None:
            sets.append("last_processed_todo_id = :last_todo")
            params["last_todo"] = last_processed_todo_id
        if total_updates_made is not None:
            sets.append("total_updates_made = :total")
            params["total"] = total_updates_made

        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE goal_auto_updater_state SET {", ".join(sets)}
                WHERE id = 1 RETURNING *
            """,
                params,
            )

    async def reset_state(self) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE goal_auto_updater_state
                SET last_run_at = NULL,
                    last_processed_activity_id = 0,
                    last_processed_event_id = 0,
                    last_processed_habit_occurrence_id = 0,
                    last_processed_todo_id = 0,
                    total_updates_made = 0,
                    updated_at = NOW()
                WHERE id = 1 RETURNING *
            """,
            )

    async def log_update(
        self,
        goal_id: str,
        previous_progress: int,
        new_progress: int,
        evidence_type: str,
        evidence_id: str,
        evidence_summary: str,
        match_reason: str,
        confidence_score: float,
        triggered_by: str = "cron",
    ) -> dict[str, Any]:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        log_id = f"log_{now}_{evidence_id[:8]}"

        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                INSERT INTO goal_auto_update_logs
                    (log_id, goal_id, previous_progress, new_progress, progress_delta,
                     evidence_type, evidence_id, evidence_summary, match_reason,
                     confidence_score, triggered_by)
                VALUES
                    (:log_id, :goal_id, :prev, :new, :delta,
                     :evidence_type, :evidence_id, :summary, :reason,
                     :confidence, :triggered_by)
                ON CONFLICT (log_id) DO NOTHING
                RETURNING *
            """,
                {
                    "log_id": log_id,
                    "goal_id": goal_id,
                    "prev": previous_progress,
                    "new": new_progress,
                    "delta": new_progress - previous_progress,
                    "evidence_type": evidence_type,
                    "evidence_id": evidence_id,
                    "summary": evidence_summary,
                    "reason": match_reason,
                    "confidence": confidence_score,
                    "triggered_by": triggered_by,
                },
            )
            if row is None:
                return await fetch_one(
                    s,
                    "SELECT * FROM goal_auto_update_logs WHERE log_id = :log_id",
                    {"log_id": log_id},
                )
            return row

    async def log_skipped(
        self,
        goal_id: str,
        current_progress: int,
        evidence_type: str,
        evidence_id: str,
        evidence_summary: str,
        match_reason: str,
        confidence_score: float,
        skip_reason: str,
        triggered_by: str = "cron",
    ) -> dict[str, Any]:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        log_id = f"log_{now}_{evidence_id[:8]}_skipped"

        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                INSERT INTO goal_auto_update_logs
                    (log_id, goal_id, previous_progress, new_progress, progress_delta,
                     evidence_type, evidence_id, evidence_summary, match_reason,
                     confidence_score, triggered_by)
                VALUES
                    (:log_id, :goal_id, :progress, :progress, 0,
                     :evidence_type, :evidence_id, :summary, :reason,
                     :confidence, :triggered_by)
                ON CONFLICT (log_id) DO NOTHING
                RETURNING *
            """,
                {
                    "log_id": log_id,
                    "goal_id": goal_id,
                    "progress": current_progress,
                    "evidence_type": evidence_type,
                    "evidence_id": evidence_id,
                    "summary": f"SKIPPED: {skip_reason} | {evidence_summary}",
                    "reason": match_reason,
                    "confidence": confidence_score,
                    "triggered_by": triggered_by,
                },
            )
            if row is None:
                return await fetch_one(
                    s,
                    "SELECT * FROM goal_auto_update_logs WHERE log_id = :log_id",
                    {"log_id": log_id},
                )
            return row

    async def get_logs(
        self,
        *,
        goal_id: str | None = None,
        evidence_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conds = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if goal_id:
            conds.append("goal_id = :goal_id")
            params["goal_id"] = goal_id
        if evidence_type:
            conds.append("evidence_type = :evidence_type")
            params["evidence_type"] = evidence_type

        where = "WHERE " + " AND ".join(conds) if conds else "WHERE 1=1"

        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM goal_auto_update_logs
                {where}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """,
                params,
            )

    async def count_updates_today(self, goal_id: str, target_date: date | None = None) -> int:
        if target_date is None:
            target_date = date.today()

        async with get_async_session() as s:
            row = await fetch_one(
                s,
                """
                SELECT COUNT(*) as cnt FROM goal_auto_update_logs
                WHERE goal_id = :goal_id
                  AND progress_delta > 0
                  AND DATE(created_at) = :target_date
            """,
                {"goal_id": goal_id, "target_date": target_date},
            )
            return row["cnt"] if row else 0

    async def get_last_processed_ids(self) -> dict[str, int]:
        state = await self.get_state()
        return {
            "activity": state.get("last_processed_activity_id", 0),
            "event": state.get("last_processed_event_id", 0),
            "habit_occurrence": state.get("last_processed_habit_occurrence_id", 0),
            "todo": state.get("last_processed_todo_id", 0),
        }
