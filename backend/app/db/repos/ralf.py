"""Ralf workflow repositories — process, stages, attempts, logs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


def _make_ralf_id() -> str:
    now = datetime.now(UTC)
    return f"ralf_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class RalfProcessesRepo:
    """Top-level ralf workflow state."""

    async def create(
        self,
        *,
        user_request: str,
        content_class: str = "public",
        model: str | None = None,
        max_total_attempts: int = 40,
        deadline_at: datetime | None = None,
    ) -> dict[str, Any]:
        ralf_id = _make_ralf_id()
        sql = """
            INSERT INTO ralf_processes
                (ralf_id, user_request, content_class, model, status, last_heartbeat, created_at,
                 max_total_attempts, deadline_at)
            VALUES
                (:ralf_id, :user_request, :content_class, :model, 'planning', :now, :now,
                 :max_total_attempts, :deadline_at)
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "user_request": user_request,
            "content_class": content_class,
            "model": model,
            "now": datetime.now(UTC),
            "max_total_attempts": max_total_attempts,
            "deadline_at": deadline_at,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {"ralf_id": ralf_id}

    async def set_budget(
        self,
        ralf_id: str,
        *,
        max_total_attempts: int | None = None,
        deadline_at: datetime | None = None,
    ) -> dict[str, Any]:
        sql = """
            UPDATE ralf_processes
            SET max_total_attempts = COALESCE(:max_total_attempts, max_total_attempts),
                deadline_at = COALESCE(:deadline_at, deadline_at)
            WHERE ralf_id = :ralf_id
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "max_total_attempts": max_total_attempts,
            "deadline_at": deadline_at,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def total_attempts(self, ralf_id: str) -> int:
        sql = "SELECT COUNT(*) AS n FROM ralf_step_attempts WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id})
            return int(row["n"]) if row else 0

    async def get(self, ralf_id: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM ralf_processes WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id})
            return dict(row) if row else None

    async def list_active(self) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM ralf_processes
            WHERE status IN ('planning', 'plan_review', 'running', 'stuck')
            ORDER BY created_at DESC
        """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {})
            return [dict(r) for r in rows]

    async def list_all(self, *, limit: int = 50) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM ralf_processes
            ORDER BY created_at DESC
            LIMIT :limit
        """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"limit": limit})
            return [dict(r) for r in rows]

    async def update_status(
        self,
        ralf_id: str,
        status: str,
        *,
        last_error: str | None = None,
        stuck_reason: str | None = None,
    ) -> dict[str, Any]:
        sql = """
            UPDATE ralf_processes
            SET status = :status,
                last_heartbeat = :now,
                last_error = COALESCE(:last_error, last_error),
                stuck_reason = COALESCE(:stuck_reason, stuck_reason)
            WHERE ralf_id = :ralf_id
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "status": status,
            "last_error": last_error,
            "stuck_reason": stuck_reason,
            "now": datetime.now(UTC),
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def heartbeat(self, ralf_id: str) -> None:
        sql = "UPDATE ralf_processes SET last_heartbeat = :now WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "now": datetime.now(UTC)})

    async def set_task_name(self, ralf_id: str, task_name: str) -> None:
        sql = "UPDATE ralf_processes SET task_name = :task_name WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "task_name": task_name})

    async def set_total_stages(self, ralf_id: str, total_stages: int) -> None:
        sql = "UPDATE ralf_processes SET total_stages = :total_stages WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "total_stages": total_stages})

    async def set_current_stage(self, ralf_id: str, current_stage: int) -> None:
        sql = """
            UPDATE ralf_processes
            SET current_stage = :current_stage, last_heartbeat = :now
            WHERE ralf_id = :ralf_id
        """
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "current_stage": current_stage, "now": datetime.now(UTC)})

    async def set_completed(self, ralf_id: str) -> None:
        sql = """
            UPDATE ralf_processes
            SET status = 'completed', completed_at = :now, last_heartbeat = :now
            WHERE ralf_id = :ralf_id
        """
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "now": datetime.now(UTC)})

    async def set_failed(self, ralf_id: str, stuck_reason: str) -> None:
        sql = """
            UPDATE ralf_processes
            SET status = 'failed', stuck_reason = :stuck_reason,
                completed_at = :now, last_heartbeat = :now
            WHERE ralf_id = :ralf_id
        """
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "stuck_reason": stuck_reason, "now": datetime.now(UTC)})


class RalfStagesRepo:
    """Plan stages within a ralf process."""

    async def create(
        self,
        *,
        ralf_id: str,
        stage_number: int,
        stage_name: str,
        goal: str,
        finalization_criteria: str,
        notes: str = "",
    ) -> dict[str, Any]:
        sql = """
            INSERT INTO ralf_stages
                (ralf_id, stage_number, stage_name, goal, finalization_criteria, notes)
            VALUES
                (:ralf_id, :stage_number, :stage_name, :goal, :finalization_criteria, :notes)
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "stage_name": stage_name,
            "goal": goal,
            "finalization_criteria": finalization_criteria,
            "notes": notes,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def list_for_ralf(self, ralf_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM ralf_stages WHERE ralf_id = :ralf_id ORDER BY stage_number"
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"ralf_id": ralf_id})
            return [dict(r) for r in rows]

    async def get_stage(self, ralf_id: str, stage_number: int) -> dict[str, Any] | None:
        sql = """
            SELECT * FROM ralf_stages
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
        """
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id, "stage_number": stage_number})
            return dict(row) if row else None

    async def update_status(
        self,
        ralf_id: str,
        stage_number: int,
        status: str,
        *,
        notes: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        started_at_clause = (
            "started_at = COALESCE(started_at, :now)" if status == "in_progress" else "started_at = started_at"
        )
        completed_at_clause = ", completed_at = :now" if status in ("approved", "failed") else ""

        sql = f"""
            UPDATE ralf_stages
            SET status = :status,
                {started_at_clause}
                {completed_at_clause},
                notes = COALESCE(:notes, notes)
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "status": status,
            "notes": notes,
            "now": now,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def delete_all_for_ralf(self, ralf_id: str) -> None:
        sql = "DELETE FROM ralf_stages WHERE ralf_id = :ralf_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id})

    async def update_criteria(
        self,
        ralf_id: str,
        stage_number: int,
        *,
        goal: str | None = None,
        finalization_criteria: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        sql = """
            UPDATE ralf_stages
            SET goal = COALESCE(:goal, goal),
                finalization_criteria = COALESCE(:finalization_criteria, finalization_criteria),
                notes = COALESCE(:notes, notes)
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "goal": goal,
            "finalization_criteria": finalization_criteria,
            "notes": notes,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}


class RalfStepAttemptsRepo:
    """Attempts per stage — each attempt is one executor run."""

    async def create(
        self,
        *,
        ralf_id: str,
        stage_number: int,
    ) -> dict[str, Any]:
        # Compute next attempt number
        count_sql = """
            SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
            FROM ralf_step_attempts
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
        """
        async with get_async_session() as s:
            row = await fetch_one(s, count_sql, {"ralf_id": ralf_id, "stage_number": stage_number})
            next_attempt = row["next_attempt"] if row else 1

        insert_sql = """
            INSERT INTO ralf_step_attempts
                (ralf_id, stage_number, attempt_number)
            VALUES
                (:ralf_id, :stage_number, :attempt_number)
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "attempt_number": next_attempt,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, insert_sql, params)
            return dict(row) if row else {}

    async def get_latest(self, ralf_id: str, stage_number: int) -> dict[str, Any] | None:
        sql = """
            SELECT * FROM ralf_step_attempts
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
            ORDER BY attempt_number DESC LIMIT 1
        """
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id, "stage_number": stage_number})
            return dict(row) if row else None

    async def get(self, ralf_id: str, stage_number: int, attempt_number: int) -> dict[str, Any] | None:
        sql = """
            SELECT * FROM ralf_step_attempts
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number AND attempt_number = :attempt_number
        """
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                sql,
                {"ralf_id": ralf_id, "stage_number": stage_number, "attempt_number": attempt_number},
            )
            return dict(row) if row else None

    async def count_attempts(self, ralf_id: str, stage_number: int) -> int:
        sql = """
            SELECT COUNT(*) AS n FROM ralf_step_attempts
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
        """
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id, "stage_number": stage_number})
            return int(row["n"]) if row else 0

    async def update_outcome(
        self,
        ralf_id: str,
        stage_number: int,
        attempt_number: int,
        outcome: str,
        *,
        evaluator_verdict: str | None = None,
        evaluator_notes: str | None = None,
    ) -> dict[str, Any]:
        completed_at_clause = (
            ", completed_at = :now" if outcome in ("approved", "retry", "impossible", "awaiting_eval") else ""
        )
        sql = f"""
            UPDATE ralf_step_attempts
            SET outcome = :outcome,
                evaluator_verdict = COALESCE(:evaluator_verdict, evaluator_verdict),
                evaluator_notes = COALESCE(:evaluator_notes, evaluator_notes)
                {completed_at_clause}
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number AND attempt_number = :attempt_number
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "attempt_number": attempt_number,
            "outcome": outcome,
            "evaluator_verdict": evaluator_verdict,
            "evaluator_notes": evaluator_notes,
            "now": datetime.now(UTC),
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def set_session_id(self, ralf_id: str, stage_number: int, attempt_number: int, session_id: str) -> None:
        sql = """
            UPDATE ralf_step_attempts
            SET session_id = :session_id
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number AND attempt_number = :attempt_number
        """
        async with get_async_session() as s:
            await execute_sql(
                s,
                sql,
                {
                    "ralf_id": ralf_id,
                    "stage_number": stage_number,
                    "attempt_number": attempt_number,
                    "session_id": session_id,
                },
            )


class RalfStepLogsRepo:
    """Executor reasoning logs per attempt."""

    async def append(
        self,
        *,
        ralf_id: str,
        stage_number: int,
        attempt_number: int,
        log_type: str,
        log_entry: str,
    ) -> dict[str, Any]:
        sql = """
            INSERT INTO ralf_step_logs
                (ralf_id, stage_number, attempt_number, log_type, log_entry)
            VALUES
                (:ralf_id, :stage_number, :attempt_number, :log_type, :log_entry)
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "attempt_number": attempt_number,
            "log_type": log_type,
            "log_entry": log_entry,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def list_for_attempt(self, ralf_id: str, stage_number: int, attempt_number: int) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM ralf_step_logs
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number AND attempt_number = :attempt_number
            ORDER BY created_at
        """
        async with get_async_session() as s:
            rows = await fetch_all(
                s,
                sql,
                {"ralf_id": ralf_id, "stage_number": stage_number, "attempt_number": attempt_number},
            )
            return [dict(r) for r in rows]

    async def list_for_stage(self, ralf_id: str, stage_number: int) -> list[dict[str, Any]]:
        sql = """
            SELECT * FROM ralf_step_logs
            WHERE ralf_id = :ralf_id AND stage_number = :stage_number
            ORDER BY attempt_number, created_at
        """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"ralf_id": ralf_id, "stage_number": stage_number})
            return [dict(r) for r in rows]


class RalfKVRepo:
    """Run-scoped key-value memory — structured data shared across agent runs."""

    async def set(
        self,
        *,
        ralf_id: str,
        key: str,
        value: str,
        explanation: str,
        value_type: str = "text",
        created_by: str = "",
    ) -> dict[str, Any]:
        sql = """
            INSERT INTO ralf_kv (ralf_id, key, value, explanation, value_type, created_by)
            VALUES (:ralf_id, :key, :value, :explanation, :value_type, :created_by)
            ON CONFLICT (ralf_id, key) DO UPDATE
                SET value = EXCLUDED.value,
                    explanation = EXCLUDED.explanation,
                    value_type = EXCLUDED.value_type,
                    created_by = COALESCE(EXCLUDED.created_by, ralf_kv.created_by),
                    updated_at = NOW()
            RETURNING *
        """
        params = {
            "ralf_id": ralf_id,
            "key": key,
            "value": value,
            "explanation": explanation,
            "value_type": value_type,
            "created_by": created_by or None,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def get(self, ralf_id: str, key: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM ralf_kv WHERE ralf_id = :ralf_id AND key = :key"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id, "key": key})
            return dict(row) if row else None

    async def list_for_ralf(self, ralf_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM ralf_kv WHERE ralf_id = :ralf_id ORDER BY key"
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"ralf_id": ralf_id})
            return [dict(r) for r in rows]

    async def delete(self, ralf_id: str, key: str) -> bool:
        sql = "DELETE FROM ralf_kv WHERE ralf_id = :ralf_id AND key = :key"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"ralf_id": ralf_id, "key": key})
            return True


class RalfAmendmentsRepo:
    """User refinements folded into a running ralf.

    Added by Twily's post-run hook when the chat agent classifies a user message
    as a refinement of an active ralf (ralf_action=amend). Read by the executor
    at stage start; cleared via mark_read once folded into the stage plan.
    """

    async def add(
        self,
        *,
        ralf_id: str,
        note: str,
        stage_number_when_added: int | None = None,
    ) -> dict[str, Any]:
        sql = """
            INSERT INTO ralf_amendments (ralf_id, note, stage_number_when_added)
            VALUES (:ralf_id, :note, :stage)
            RETURNING *
        """
        params = {"ralf_id": ralf_id, "note": note, "stage": stage_number_when_added}
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def list_for_ralf(
        self,
        ralf_id: str,
        *,
        unread_only: bool = False,
    ) -> list[dict[str, Any]]:
        if unread_only:
            sql = """
                SELECT * FROM ralf_amendments
                WHERE ralf_id = :ralf_id AND read_at IS NULL
                ORDER BY created_at, id
            """
        else:
            sql = """
                SELECT * FROM ralf_amendments
                WHERE ralf_id = :ralf_id
                ORDER BY created_at, id
            """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"ralf_id": ralf_id})
            return [dict(r) for r in rows]

    async def count_unread(self, ralf_id: str) -> int:
        sql = """
            SELECT COUNT(*) AS n FROM ralf_amendments
            WHERE ralf_id = :ralf_id AND read_at IS NULL
        """
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"ralf_id": ralf_id})
            return int(row["n"]) if row else 0

    async def mark_read(
        self,
        ralf_id: str,
        *,
        up_to_id: int | None = None,
        stage_number_when_read: int | None = None,
    ) -> int:
        """Mark unread amendments as read. If up_to_id given, only mark <= that id.
        Returns number of rows marked."""
        if up_to_id is not None:
            sql = """
                UPDATE ralf_amendments
                SET read_at = :now, stage_number_when_read = :stage
                WHERE ralf_id = :ralf_id AND read_at IS NULL AND id <= :up_to_id
            """
            params = {
                "ralf_id": ralf_id,
                "up_to_id": up_to_id,
                "stage": stage_number_when_read,
                "now": datetime.now(UTC),
            }
        else:
            sql = """
                UPDATE ralf_amendments
                SET read_at = :now, stage_number_when_read = :stage
                WHERE ralf_id = :ralf_id AND read_at IS NULL
            """
            params = {
                "ralf_id": ralf_id,
                "stage": stage_number_when_read,
                "now": datetime.now(UTC),
            }
        async with get_async_session() as s:
            result = await execute_sql(s, sql, params)
            return getattr(result, "rowcount", 0) or 0


class RalfLocksRepo:
    """Cross-ralf resource coordination via named locks with TTL."""

    async def acquire(
        self,
        *,
        resource_key: str,
        ralf_id: str,
        ttl_seconds: int = 600,
        stage_number: int | None = None,
        notes: str = "",
    ) -> dict[str, Any] | None:
        """Try to acquire lock. Returns row on success, None if held by another ralf."""
        from datetime import timedelta

        now = datetime.now(UTC)
        expires = now + timedelta(seconds=ttl_seconds)
        # Clean up expired locks first
        await self._cleanup_expired()
        # Try insert
        sql = """
            INSERT INTO ralf_locks (resource_key, holder_ralf_id, holder_stage_number, acquired_at, expires_at, notes)
            VALUES (:resource_key, :ralf_id, :stage_number, :now, :expires, :notes)
            ON CONFLICT (resource_key) DO UPDATE
                SET holder_ralf_id = EXCLUDED.holder_ralf_id,
                    holder_stage_number = EXCLUDED.holder_stage_number,
                    acquired_at = EXCLUDED.acquired_at,
                    expires_at = EXCLUDED.expires_at,
                    notes = EXCLUDED.notes
                WHERE ralf_locks.expires_at < NOW() OR ralf_locks.holder_ralf_id = :ralf_id
            RETURNING *
        """
        params = {
            "resource_key": resource_key,
            "ralf_id": ralf_id,
            "stage_number": stage_number,
            "now": now,
            "expires": expires,
            "notes": notes or None,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else None

    async def release(self, resource_key: str, ralf_id: str) -> bool:
        sql = "DELETE FROM ralf_locks WHERE resource_key = :resource_key AND holder_ralf_id = :ralf_id"
        async with get_async_session() as s:
            await execute_sql(s, sql, {"resource_key": resource_key, "ralf_id": ralf_id})
            return True

    async def get(self, resource_key: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM ralf_locks WHERE resource_key = :resource_key"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"resource_key": resource_key})
            return dict(row) if row else None

    async def list_held_by(self, ralf_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM ralf_locks WHERE holder_ralf_id = :ralf_id"
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"ralf_id": ralf_id})
            return [dict(r) for r in rows]

    async def _cleanup_expired(self) -> None:
        sql = "DELETE FROM ralf_locks WHERE expires_at < NOW()"
        async with get_async_session() as s:
            await execute_sql(s, sql, {})
