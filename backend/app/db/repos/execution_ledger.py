"""Execution ledger repository — DB-backed artifact store for inter-agent coordination.

Replaces thought_transfer's destructive file-based reads with non-destructive,
versioned, run-scoped artifacts. Multiple consumers can read the same artifact.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from app.db.session import execute_sql, fetch_all, fetch_one, get_async_session


class ExecutionLedgerRepo:
    """Manages execution_runs and execution_artifacts tables."""

    # ── Runs ──

    async def start_run(
        self,
        *,
        interaction_mode: str,
        domain: str = "",
        owner: str = "",
        root_message_id: int | None = None,
        root_session_id: str = "",
    ) -> dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:16]}"
        sql = """
            INSERT INTO execution_runs
                (run_id, interaction_mode, domain, owner, root_message_id, root_session_id, started_at)
            VALUES
                (:run_id, :interaction_mode, :domain, :owner, :root_message_id, :root_session_id, :started_at)
            RETURNING *
        """
        params = {
            "run_id": run_id,
            "interaction_mode": interaction_mode,
            "domain": domain,
            "owner": owner,
            "root_message_id": root_message_id,
            "root_session_id": root_session_id,
            "started_at": datetime.now(UTC),
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {"run_id": run_id}

    async def ensure_run(
        self,
        run_id: str,
        *,
        interaction_mode: str = "orphan",
        domain: str = "",
        owner: str = "",
    ) -> None:
        """Upsert a run row for the given run_id so artifacts can be attached.

        Used by emit_guidance.py and other tools that may be invoked outside a
        pre-existing run (e.g. orphan calls, debugging). If the run already
        exists, this is a no-op — `ON CONFLICT DO NOTHING` keeps original
        interaction_mode/domain/owner intact.
        """
        sql = """
            INSERT INTO execution_runs
                (run_id, interaction_mode, domain, owner, started_at)
            VALUES
                (:run_id, :interaction_mode, :domain, :owner, :started_at)
            ON CONFLICT (run_id) DO NOTHING
        """
        params = {
            "run_id": run_id,
            "interaction_mode": interaction_mode,
            "domain": domain,
            "owner": owner,
            "started_at": datetime.now(UTC),
        }
        async with get_async_session() as s:
            await execute_sql(s, sql, params)

    async def complete_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        contract_passed: bool | None = None,
    ) -> dict[str, Any]:
        sql = """
            UPDATE execution_runs
            SET status = :status,
                completed_at = :completed_at,
                contract_passed = :contract_passed
            WHERE run_id = :run_id
            RETURNING *
        """
        params = {
            "run_id": run_id,
            "status": status,
            "completed_at": datetime.now(UTC),
            "contract_passed": contract_passed,
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {}

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        sql = "SELECT * FROM execution_runs WHERE run_id = :run_id"
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"run_id": run_id})
            return dict(row) if row else None

    async def supersede_run(self, run_id: str, superseded_by: str) -> bool:
        sql = """
            UPDATE execution_runs
            SET status = 'superseded', superseded_by = :superseded_by, completed_at = :now
            WHERE run_id = :run_id AND status = 'running'
        """
        async with get_async_session() as s:
            await execute_sql(
                s,
                sql,
                {
                    "run_id": run_id,
                    "superseded_by": superseded_by,
                    "now": datetime.now(UTC),
                },
            )
            return True

    async def get_active_run_for_message(self, root_message_id: int) -> dict[str, Any] | None:
        sql = """
            SELECT * FROM execution_runs
            WHERE root_message_id = :mid AND status = 'running'
            ORDER BY started_at DESC LIMIT 1
        """
        async with get_async_session() as s:
            row = await fetch_one(s, sql, {"mid": root_message_id})
            return dict(row) if row else None

    # ── Artifacts ──

    async def write_artifact(
        self,
        run_id: str,
        artifact_type: str,
        payload: dict | str,
        *,
        producer: str = "",
    ) -> dict[str, Any]:
        # Auto-version: find max version for this run+type, increment
        version_sql = """
            SELECT COALESCE(MAX(version), 0) + 1 AS next_ver
            FROM execution_artifacts
            WHERE run_id = :run_id AND artifact_type = :artifact_type
        """
        async with get_async_session() as s:
            ver_row = await fetch_one(
                s,
                version_sql,
                {
                    "run_id": run_id,
                    "artifact_type": artifact_type,
                },
            )
            next_ver = ver_row["next_ver"] if ver_row else 1

        artifact_id = f"art_{uuid.uuid4().hex[:16]}"
        payload_json = json.dumps(payload) if isinstance(payload, dict) else json.dumps({"content": payload})

        sql = """
            INSERT INTO execution_artifacts
                (artifact_id, run_id, artifact_type, version, producer, payload, created_at)
            VALUES
                (:artifact_id, :run_id, :artifact_type, :version, :producer,
                 CAST(:payload AS jsonb), :created_at)
            RETURNING *
        """
        params = {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "artifact_type": artifact_type,
            "version": next_ver,
            "producer": producer,
            "payload": payload_json,
            "created_at": datetime.now(UTC),
        }
        async with get_async_session() as s:
            row = await fetch_one(s, sql, params)
            return dict(row) if row else {"artifact_id": artifact_id}

    async def read_artifact(
        self,
        run_id: str,
        artifact_type: str,
        *,
        consumer: str = "",
    ) -> dict[str, Any] | None:
        """Read the latest version of an artifact. Marks consumed but does NOT delete."""
        sql = """
            SELECT * FROM execution_artifacts
            WHERE run_id = :run_id AND artifact_type = :artifact_type
            ORDER BY version DESC LIMIT 1
        """
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                sql,
                {
                    "run_id": run_id,
                    "artifact_type": artifact_type,
                },
            )
            if not row:
                return None

            # Mark as consumed (non-destructive)
            if consumer:
                await execute_sql(
                    s,
                    """
                    UPDATE execution_artifacts
                    SET consumed_by = :consumer, consumed_at = :now, status = 'consumed'
                    WHERE artifact_id = :aid
                    """,
                    {
                        "consumer": consumer,
                        "now": datetime.now(UTC),
                        "aid": row["artifact_id"],
                    },
                )

            result = dict(row)
            # Parse payload back from jsonb
            if isinstance(result.get("payload"), str):
                result["payload"] = json.loads(result["payload"])
            return result

    async def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        sql = """
            SELECT artifact_id, artifact_type, version, producer, status, created_at
            FROM execution_artifacts
            WHERE run_id = :run_id
            ORDER BY created_at
        """
        async with get_async_session() as s:
            rows = await fetch_all(s, sql, {"run_id": run_id})
            return [dict(r) for r in rows]

    async def has_artifact(self, run_id: str, artifact_type: str) -> bool:
        sql = """
            SELECT 1 FROM execution_artifacts
            WHERE run_id = :run_id AND artifact_type = :artifact_type
            LIMIT 1
        """
        async with get_async_session() as s:
            row = await fetch_one(
                s,
                sql,
                {
                    "run_id": run_id,
                    "artifact_type": artifact_type,
                },
            )
            return row is not None

    async def get_run_status(self, run_id: str) -> dict[str, Any]:
        """Get run with its artifact summary."""
        run = await self.get_run(run_id)
        if not run:
            return {"error": f"Run {run_id} not found"}
        artifacts = await self.list_artifacts(run_id)
        return {
            **run,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        }
