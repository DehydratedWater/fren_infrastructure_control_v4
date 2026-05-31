"""Vis simulation repositories — simulations, messages, scores."""

from __future__ import annotations

import json
from typing import Any

from app.db.session import fetch_all, fetch_one, get_async_session


class SimulationsRepo:
    async def create(
        self,
        simulation_id: str,
        scenario_type: str,
        scenario_description: str,
        **kw: Any,
    ) -> dict[str, Any]:
        jsonb_keys = (
            "scenario_assumptions",
            "journal_topics",
            "emotional_state",
            "generation_params",
            "fine_tuning_context",
        )
        params: dict[str, Any] = {
            "sid": simulation_id,
            "stype": scenario_type,
            "sdesc": scenario_description,
        }
        cols = ["simulation_id", "scenario_type", "scenario_description"]
        vals = [":sid", ":stype", ":sdesc"]
        for k, v in kw.items():
            if v is not None:
                cols.append(k)
                if k in jsonb_keys:
                    v = json.dumps(v)
                    vals.append(f"CAST(:{k} AS jsonb)")
                else:
                    vals.append(f":{k}")
                params[k] = v
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                INSERT INTO vis_simulations ({", ".join(cols)})
                VALUES ({", ".join(vals)}) RETURNING *
            """,
                params,
            )  # type: ignore[return-value]

    async def get(self, simulation_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s, "SELECT * FROM vis_simulations WHERE simulation_id = :sid", {"sid": simulation_id}
            )

    async def list(self, *, status: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        conds: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status:
            conds.append("status = :status")
            params["status"] = status
        where = " AND ".join(conds) if conds else "1=1"
        async with get_async_session() as s:
            return await fetch_all(
                s,
                f"""
                SELECT * FROM vis_simulations WHERE {where}
                ORDER BY created_at DESC LIMIT :limit
            """,
                params,
            )

    async def complete(self, simulation_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                UPDATE vis_simulations SET status = 'completed', completed_at = NOW()
                WHERE simulation_id = :sid RETURNING *
            """,
                {"sid": simulation_id},
            )


class MessagesRepo:
    async def create(
        self,
        message_id: str,
        simulation_id: str,
        sequence_number: int,
        sender: str,
        response_content: str,
        *,
        thinking_content: str | None = None,
        actions: list | None = None,
        trigger_type: str | None = None,
    ) -> dict[str, Any]:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                INSERT INTO vis_simulation_messages (message_id, simulation_id,
                    sequence_number, sender, thinking_content, response_content,
                    actions, trigger_type)
                VALUES (:mid, :sid, :seq, :sender, :think, :resp, CAST(:acts AS jsonb), :tt)
                RETURNING *
            """,
                {
                    "mid": message_id,
                    "sid": simulation_id,
                    "seq": sequence_number,
                    "sender": sender,
                    "think": thinking_content,
                    "resp": response_content,
                    "acts": json.dumps(actions or []),
                    "tt": trigger_type,
                },
            )  # type: ignore[return-value]

    async def list_for_simulation(self, simulation_id: str) -> list[dict[str, Any]]:
        async with get_async_session() as s:
            return await fetch_all(
                s,
                """
                SELECT * FROM vis_simulation_messages
                WHERE simulation_id = :sid ORDER BY sequence_number
            """,
                {"sid": simulation_id},
            )


class ScoresRepo:
    async def create(
        self,
        score_id: str,
        simulation_id: str,
        quality_score: float,
        realism_score: float,
        character_adherence_score: float,
        **kw: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "scid": score_id,
            "sid": simulation_id,
            "qs": quality_score,
            "rs": realism_score,
            "cas": character_adherence_score,
        }
        extra_cols: list[str] = []
        extra_vals: list[str] = []
        jsonb_keys = ("topics_investigated", "journal_references")
        for k, v in kw.items():
            if v is not None:
                extra_cols.append(k)
                if k in jsonb_keys:
                    v = json.dumps(v)
                    extra_vals.append(f"CAST(:{k} AS jsonb)")
                else:
                    extra_vals.append(f":{k}")
                params[k] = v
        cols = ["score_id", "simulation_id", "quality_score", "realism_score", "character_adherence_score", *extra_cols]
        vals = [":scid", ":sid", ":qs", ":rs", ":cas", *extra_vals]
        async with get_async_session() as s:
            return await fetch_one(
                s,
                f"""
                INSERT INTO vis_simulation_scores ({", ".join(cols)})
                VALUES ({", ".join(vals)}) RETURNING *
            """,
                params,
            )  # type: ignore[return-value]

    async def get_for_simulation(self, simulation_id: str) -> dict[str, Any] | None:
        async with get_async_session() as s:
            return await fetch_one(
                s,
                """
                SELECT * FROM vis_simulation_scores WHERE simulation_id = :sid
            """,
                {"sid": simulation_id},
            )
