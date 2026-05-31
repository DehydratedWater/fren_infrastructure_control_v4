"""Profile repository — categories, discoveries, hypotheses, observations, runs."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class ProfileRepo:
    # ── Categories ──

    async def list_categories(self) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(s, "SELECT * FROM profile_categories ORDER BY name")

    async def get_category(self, name: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(s, "SELECT * FROM profile_categories WHERE name = :n", {"n": name})

    # ── Discoveries ──

    async def create_discovery(
        self,
        discovery: str,
        *,
        category_id: str | None = None,
        confidence: float = 0.5,
        metadata: dict | None = None,
        sensitivity: str = "public",
        public_summary: str | None = None,
        embedding: list[float] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "cid": category_id,
            "disc": discovery,
            "conf": confidence,
            "meta": json.dumps(metadata or {}),
            "sensitivity": sensitivity,
            "public_summary": public_summary,
        }
        if embedding:
            vec = "[" + ",".join(str(v) for v in embedding) + "]"
            params["embedding"] = vec
            sql = """
                INSERT INTO profile_discoveries (category_id, discovery, confidence,
                    first_observed_at, last_confirmed_at, metadata, sensitivity, public_summary, embedding)
                VALUES (CAST(:cid AS uuid), :disc, :conf, NOW(), NOW(), CAST(:meta AS jsonb),
                    :sensitivity, :public_summary, CAST(:embedding AS vector))
                RETURNING *
            """
        else:
            sql = """
                INSERT INTO profile_discoveries (category_id, discovery, confidence,
                    first_observed_at, last_confirmed_at, metadata, sensitivity, public_summary)
                VALUES (CAST(:cid AS uuid), :disc, :conf, NOW(), NOW(), CAST(:meta AS jsonb),
                    :sensitivity, :public_summary)
                RETURNING *
            """
        async with get_async_session() as s:
            return await fetch_one(s, sql, params)  # type: ignore[return-value]

    async def list_discoveries(
        self,
        *,
        status: str = "active",
        category_id: str | None = None,
        limit: int = 50,
        clearance: str = "full",
    ) -> list[dict[str, Any]]:
        conds = ["status = :status"]
        params: dict[str, Any] = {"status": status, "limit": limit}
        if category_id:
            conds.append("category_id = CAST(:cid AS uuid)")
            params["cid"] = category_id
        if clearance != "full":
            # For public clearance, only return public discoveries
            conds.append("sensitivity = 'public'")
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM profile_discoveries
                WHERE {" AND ".join(conds)}
                ORDER BY confidence DESC LIMIT :limit
            """,
                params,
            )

    async def update_discovery(self, discovery_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"did": discovery_id}
        for k, v in fields.items():
            if v is not None:
                if k == "metadata":
                    v = json.dumps(v)
                    sets.append(f"{k} = CAST(:{k} AS jsonb)")
                else:
                    sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE profile_discoveries SET {", ".join(sets)}
                WHERE id = CAST(:did AS uuid) RETURNING *
            """,
                params,
            )

    # ── Hypotheses ──

    async def create_hypothesis(
        self,
        hypothesis: str,
        *,
        category_id: str | None = None,
        confidence_score: float = 0.0,
        sensitivity: str = "public",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO profile_hypotheses (category_id, hypothesis, confidence_score, sensitivity)
                VALUES (CAST(:cid AS uuid), :hyp, :conf, :sensitivity)
                RETURNING *
            """,
                {"cid": category_id, "hyp": hypothesis, "conf": confidence_score, "sensitivity": sensitivity},
            )  # type: ignore[return-value]

    async def list_hypotheses(
        self, *, status: str | None = None, limit: int = 50, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status:
            conds.append("status = :status")
            params["status"] = status
        if clearance != "full":
            conds.append("sensitivity = 'public'")
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM profile_hypotheses WHERE {where}
                ORDER BY confidence_score DESC LIMIT :limit
            """,
                params,
            )

    async def update_hypothesis(self, hypothesis_id: str, **fields: Any) -> dict[str, Any] | None:
        sets = ["updated_at = NOW()"]
        params: dict[str, Any] = {"hid": hypothesis_id}
        for k, v in fields.items():
            if v is not None:
                sets.append(f"{k} = :{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE profile_hypotheses SET {", ".join(sets)}
                WHERE id = CAST(:hid AS uuid) RETURNING *
            """,
                params,
            )

    # ── Pattern observations ──

    async def create_observation(
        self,
        observation: str,
        *,
        pattern_type: str | None = None,
        source_type: str | None = None,
        source_reference: str | None = None,
        sensitivity: str = "public",
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO pattern_observations (pattern_type, observation,
                    source_type, source_reference, first_seen_at, last_seen_at, sensitivity)
                VALUES (:pt, :obs, :st, :sr, NOW(), NOW(), :sensitivity)
                RETURNING *
            """,
                {
                    "pt": pattern_type,
                    "obs": observation,
                    "st": source_type,
                    "sr": source_reference,
                    "sensitivity": sensitivity,
                },
            )  # type: ignore[return-value]

    async def list_observations(
        self, *, pattern_type: str | None = None, limit: int = 50, clearance: str = "full"
    ) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if pattern_type:
            conds.append("pattern_type = :pt")
            params["pt"] = pattern_type
        if clearance != "full":
            conds.append("sensitivity = 'public'")
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM pattern_observations WHERE {where}
                ORDER BY created_at DESC LIMIT :limit
            """,
                params,
            )

    # ── Analysis runs ──

    async def create_run(self, run_type: str, *, focus_area: str | None = None) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO analysis_runs (run_type, focus_area)
                VALUES (:rt, :fa) RETURNING *
            """,
                {"rt": run_type, "fa": focus_area},
            )  # type: ignore[return-value]

    async def complete_run(self, run_id: str, **counters: Any) -> dict[str, Any] | None:
        sets = ["status = 'completed'", "completed_at = NOW()"]
        params: dict[str, Any] = {"rid": run_id}
        for k, v in counters.items():
            sets.append(f"{k} = :{k}")
            params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                UPDATE analysis_runs SET {", ".join(sets)}
                WHERE id = CAST(:rid AS uuid) RETURNING *
            """,
                params,
            )
