"""Nudge Strategist — strategic persuasion campaign management.

Analyzes priority/goal fulfillment, creates nudge campaigns, tracks effectiveness,
rotates tactics, measures user reactions.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from src import ScriptTool
from pydantic import BaseModel, Field

# Tactic pools by escalation level
TACTIC_LEVELS: dict[int, list[str]] = {
    1: ["gentle_reminder", "curiosity", "celebration"],
    2: ["accountability", "pleading", "tempting"],
    3: ["nagging", "suggestive_image", "negotiation"],
    4: ["shame", "fear_of_loss", "suggestive_image"],
    5: ["fear_of_loss"],
}

DEFAULT_FIRST_TACTIC = "gentle_reminder"


class Input(BaseModel):
    command: str = Field(
        description="analyze|plan-campaign|get-active|record-nudge|measure-response|"
        "rotate-tactic|update-campaign|get-effectiveness|pause-campaign|resume-campaign|reset-daily"
    )
    campaign_id: str = Field(default="", description="Campaign ID")
    target_type: str = Field(default="", description="priority|goal|habit")
    target_id: str = Field(default="", description="Target entity ID")
    tactic: str = Field(default="", description="Tactic name")
    rationale: str = Field(default="", description="Rationale for tactic/change")
    escalation_level: int = Field(default=0, description="Escalation level (1-5)")
    message: str = Field(default="", description="Nudge message sent")
    influence_type: str = Field(default="", description="Type for influence_attempts logging")
    reaction_type: str = Field(default="", description="ignored|acknowledged|effective")
    hours: int = Field(default=6, description="Lookback hours for analysis")
    days: int = Field(default=30, description="Lookback days for reports")
    notes: str = Field(default="")


class Output(BaseModel):
    success: bool = True
    campaign: dict = Field(default_factory=dict)
    campaigns: list[dict] = Field(default_factory=list)
    analysis: dict = Field(default_factory=dict)
    effectiveness: dict = Field(default_factory=dict)
    count: int = 0
    error: str = ""


class NudgeStrategistTool(ScriptTool[Input, Output]):
    name = "nudge_strategist"
    description = "Strategic nudge campaign management — analyze priorities, plan campaigns, track effectiveness"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        cmd = inp.command

        if cmd == "analyze":
            return await self._analyze(inp.hours)
        if cmd == "plan-campaign":
            return await self._plan_campaign(inp)
        if cmd == "get-active":
            return await self._get_active()
        if cmd == "record-nudge":
            return await self._record_nudge(inp)
        if cmd == "measure-response":
            return await self._measure_response(inp.campaign_id)
        if cmd == "rotate-tactic":
            return await self._rotate_tactic(inp)
        if cmd == "update-campaign":
            return await self._update_campaign(inp)
        if cmd == "get-effectiveness":
            return await self._get_effectiveness(inp.days)
        if cmd == "pause-campaign":
            return await self._pause_campaign(inp)
        if cmd == "resume-campaign":
            return await self._resume_campaign(inp.campaign_id)
        if cmd == "reset-daily":
            return await self._reset_daily()
        return Output(success=False, error=f"Unknown command: {cmd}")

    async def _analyze(self, hours: int) -> Output:
        from app.db.repos.activity_blocks import ActivityBlocksRepo
        from app.db.repos.chat import ChatMessagesRepo
        from app.db.repos.commitments import CommitmentsRepo
        from app.db.repos.goals import GoalsRepo
        from app.db.repos.habits import HabitsRepo
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo
        from app.db.repos.priorities import PrioritiesRepo
        from app.db.repos.todos import TodosRepo
        from app.db.repos.user_config import UserConfigRepo

        now = datetime.now(UTC)

        # Load config
        config_row = await UserConfigRepo().get("nudge_strategist_config")
        config = {}
        if config_row:
            import json

            raw = config_row.get("config_value", "{}")
            config = json.loads(raw) if isinstance(raw, str) else raw

        global_max = config.get("global_max_nudges_per_day", 8)

        # Active campaigns
        campaigns = await NudgeCampaignsRepo().get_active()
        total_nudges_today = sum(c.get("nudges_today", 0) for c in campaigns)

        # Priorities needing attention (high importance but low real_importance = neglected)
        priorities = await PrioritiesRepo().list(status="active")
        priorities_needing = []
        for p in priorities:
            importance = float(p.get("importance", 0))
            real_importance = p.get("real_importance")
            ri = float(real_importance) if real_importance is not None else None
            if importance >= 0.6 and (ri is not None and ri < importance * 0.5):
                priorities_needing.append(
                    {
                        "priority_id": p.get("priority_id"),
                        "title": p.get("title", "")[:80],
                        "importance": importance,
                        "real_importance": ri,
                        "gap": round(importance - (ri or 0), 2),
                    }
                )

        # Stalled goals (active but no progress change in last N days)
        goals = await GoalsRepo().list_active()
        stalled = []
        for g in goals:
            progress = g.get("progress_percent", 0)
            updated = g.get("updated_at")
            if updated:
                if isinstance(updated, str):
                    updated = datetime.fromisoformat(updated)
                if hasattr(updated, "tzinfo") and updated.tzinfo is None:
                    updated = updated.replace(tzinfo=UTC)
                days_since = (now - updated).days
                if days_since >= 3 and progress < 100:
                    stalled.append(
                        {
                            "goal_id": g.get("goal_id"),
                            "title": g.get("title", "")[:80],
                            "progress": progress,
                            "days_stalled": days_since,
                            "level": g.get("level"),
                        }
                    )
        stalled.sort(key=lambda x: x["days_stalled"], reverse=True)

        # Neglected habits (due today but not completed, or low completion rate)
        habits_due = await HabitsRepo().get_due_today()
        neglected_habits = []
        for h in habits_due:
            if h.get("status") == "pending":
                neglected_habits.append(
                    {
                        "habit_id": h.get("habit_id"),
                        "title": h.get("title", "")[:80],
                        "importance": h.get("importance_level", 1),
                    }
                )

        # Overdue todos
        overdue = await TodosRepo().get_overdue()
        overdue_summary = [
            {"todo_id": t.get("todo_id"), "title": t.get("title", "")[:80], "priority": t.get("priority")}
            for t in overdue[:10]
        ]

        # Unfulfilled commitments
        pending_commitments = await CommitmentsRepo().get_pending()
        unfulfilled = [
            {"commitment_id": c.get("commitment_id"), "text": c.get("commitment_text", "")[:100]}
            for c in pending_commitments[:5]
        ]

        # Recent activity (what user's been doing instead)
        recent_blocks = await ActivityBlocksRepo().get_recent_blocks(hours=hours)
        activity_types = {}
        for b in recent_blocks:
            atype = b.get("activity_type", "unknown")
            activity_types[atype] = activity_types.get(atype, 0) + 1
        activity_summary = ", ".join(f"{k}({v})" for k, v in sorted(activity_types.items(), key=lambda x: -x[1])[:5])

        # Chat engagement trend
        recent_msgs = await ChatMessagesRepo().get_recent(limit=30)
        user_msgs = [m for m in recent_msgs if m.get("sender") == "user"]
        twily_msgs = [m for m in recent_msgs if m.get("sender") == "twily"]

        # Suggested targets — top items not yet in active campaigns
        existing_targets = {(c["target_type"], c["target_id"]) for c in campaigns}
        suggested = []
        for p in priorities_needing[:3]:
            key = ("priority", p["priority_id"])
            if key not in existing_targets:
                suggested.append({"type": "priority", "id": p["priority_id"], "title": p["title"], "urgency": p["gap"]})
        for g in stalled[:3]:
            key = ("goal", g["goal_id"])
            if key not in existing_targets:
                suggested.append(
                    {"type": "goal", "id": g["goal_id"], "title": g["title"], "urgency": g["days_stalled"] / 10}
                )
        for h in neglected_habits[:2]:
            key = ("habit", h["habit_id"])
            if key not in existing_targets:
                suggested.append(
                    {"type": "habit", "id": h["habit_id"], "title": h["title"], "urgency": h["importance"] / 5}
                )
        suggested.sort(key=lambda x: x.get("urgency", 0), reverse=True)

        return Output(
            success=True,
            analysis={
                "priorities_needing_attention": priorities_needing[:5],
                "stalled_goals": stalled[:5],
                "neglected_habits": neglected_habits[:5],
                "overdue_todos": overdue_summary,
                "unfulfilled_commitments": unfulfilled,
                "user_activity_summary": activity_summary or "no recent activity",
                "user_messages_last_30": len(user_msgs),
                "twily_messages_last_30": len(twily_msgs),
                "suggested_targets": suggested[:5],
                "active_campaigns_count": len(campaigns),
                "global_nudges_remaining_today": max(0, global_max - total_nudges_today),
            },
        )

    async def _plan_campaign(self, inp: Input) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not inp.target_type or not inp.target_id:
            return Output(success=False, error="target_type and target_id required")

        repo = NudgeCampaignsRepo()

        # Check for existing active campaign on this target
        existing = await repo.get_by_target(inp.target_type, inp.target_id)
        if existing:
            return Output(success=False, error="Active campaign already exists for this target", campaign=existing)

        # Resolve title
        title = await self._resolve_title(inp.target_type, inp.target_id)
        if not title:
            return Output(success=False, error=f"{inp.target_type} {inp.target_id} not found")

        # Pick initial tactic
        tactic = inp.tactic or DEFAULT_FIRST_TACTIC
        rationale = inp.rationale or f"Initial campaign for {inp.target_type}: {title}"

        cid = f"nudge_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{id(inp) % 0xFFFF:04x}"
        campaign = await repo.create(
            campaign_id=cid,
            target_type=inp.target_type,
            target_id=inp.target_id,
            target_title=title,
            current_tactic=tactic,
            tactic_rationale=rationale,
            escalation_level=inp.escalation_level or 1,
            notes=inp.notes or None,
        )
        return Output(success=True, campaign=campaign)

    async def _resolve_title(self, target_type: str, target_id: str) -> str | None:
        if target_type == "priority":
            from app.db.repos.priorities import PrioritiesRepo

            p = await PrioritiesRepo().get(target_id)
            return p.get("title") if p else None
        if target_type == "goal":
            from app.db.repos.goals import GoalsRepo

            g = await GoalsRepo().get(target_id)
            return g.get("title") if g else None
        if target_type == "habit":
            from app.db.repos.habits import HabitsRepo

            h = await HabitsRepo().get(target_id)
            return h.get("title") if h else None
        return None

    async def _get_active(self) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        campaigns = await NudgeCampaignsRepo().get_active()
        return Output(success=True, campaigns=campaigns, count=len(campaigns))

    async def _record_nudge(self, inp: Input) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo
        from app.db.repos.strategies import InfluenceRepo

        if not inp.campaign_id:
            return Output(success=False, error="campaign_id required")

        repo = NudgeCampaignsRepo()
        campaign = await repo.record_nudge(inp.campaign_id)
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {inp.campaign_id}")

        # Also log to influence_attempts for historical tracking
        if inp.message:
            aid = f"nudge_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{id(inp) % 0xFFFF:04x}"
            await InfluenceRepo().create(
                attempt_id=aid,
                influence_type=inp.influence_type or campaign.get("current_tactic", "nudge"),
                message_sent=inp.message[:500],
                date=datetime.now(UTC).strftime("%Y-%m-%d"),
                sent_at=datetime.now(UTC).isoformat(),
                goal_id=campaign.get("target_id") if campaign.get("target_type") == "goal" else None,
                campaign_id=inp.campaign_id,
            )

        return Output(success=True, campaign=campaign)

    async def _measure_response(self, campaign_id: str) -> Output:
        from app.db.repos.chat import ChatMessagesRepo
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not campaign_id:
            return Output(success=False, error="campaign_id required")

        repo = NudgeCampaignsRepo()
        campaign = await repo.get(campaign_id)
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {campaign_id}")

        last_nudge = campaign.get("last_nudge_at")
        if not last_nudge:
            return Output(success=True, campaign=campaign, analysis={"reaction": "no_nudge_sent"})

        if isinstance(last_nudge, str):
            last_nudge = datetime.fromisoformat(last_nudge)
        if hasattr(last_nudge, "tzinfo") and last_nudge.tzinfo is None:
            last_nudge = last_nudge.replace(tzinfo=UTC)

        # Check for chat activity after nudge
        recent_msgs = await ChatMessagesRepo().get_recent(limit=30)
        user_msgs_after = []
        for m in recent_msgs:
            if m.get("sender") != "user":
                continue
            ts = m.get("timestamp_unix", 0)
            if ts and float(ts) > last_nudge.timestamp():
                user_msgs_after.append(m)

        if not user_msgs_after:
            # No user activity since nudge
            reaction = "ignored"
        else:
            # User was active — check if they acted on the target
            acted = await self._check_target_action(campaign, last_nudge)
            reaction = "effective" if acted else "acknowledged"

        updated = await repo.record_reaction(campaign_id, reaction)
        return Output(
            success=True,
            campaign=updated or campaign,
            analysis={"reaction": reaction, "user_messages_after_nudge": len(user_msgs_after)},
        )

    async def _check_target_action(self, campaign: dict, since: datetime) -> bool:
        """Check if the user took action on the campaign target since the given time."""
        target_type = campaign.get("target_type")
        target_id = campaign.get("target_id")

        if target_type == "goal":
            # Check goal auto-update logs for recent updates
            from app.db.session import fetch_all, get_async_session

            async with get_async_session() as s:
                logs = await fetch_all(
                    s,
                    """
                    SELECT * FROM goal_auto_update_logs
                    WHERE goal_id = :gid AND created_at >= CAST(:since AS timestamptz)
                    LIMIT 1
                    """,
                    {"gid": target_id, "since": since},
                )
                if logs:
                    return True

        elif target_type == "habit":
            # Check habit occurrences completed since nudge
            from app.db.session import fetch_all, get_async_session

            async with get_async_session() as s:
                occs = await fetch_all(
                    s,
                    """
                    SELECT * FROM habit_occurrences
                    WHERE habit_id = :hid AND status = 'completed'
                      AND completed_at >= CAST(:since AS timestamptz)
                    LIMIT 1
                    """,
                    {"hid": target_id, "since": since},
                )
                if occs:
                    return True

        elif target_type == "priority":
            # Check if any todos linked to this priority were completed
            from app.db.session import fetch_all, get_async_session

            async with get_async_session() as s:
                todos = await fetch_all(
                    s,
                    """
                    SELECT t.* FROM todos t
                    JOIN priority_mappings pm ON pm.entity_id = t.todo_id AND pm.entity_type = 'todo'
                    WHERE pm.priority_id = :pid AND t.status = 'completed'
                      AND t.completed_at >= CAST(:since AS timestamptz)
                    LIMIT 1
                    """,
                    {"pid": target_id, "since": since},
                )
                if todos:
                    return True

        return False

    async def _rotate_tactic(self, inp: Input) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not inp.campaign_id:
            return Output(success=False, error="campaign_id required")
        if not inp.tactic:
            return Output(success=False, error="tactic required")

        esc = inp.escalation_level if inp.escalation_level > 0 else 1
        rationale = inp.rationale or f"Rotating to {inp.tactic} at escalation {esc}"

        repo = NudgeCampaignsRepo()
        campaign = await repo.rotate_tactic(inp.campaign_id, inp.tactic, rationale, esc)
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {inp.campaign_id}")
        return Output(success=True, campaign=campaign)

    async def _update_campaign(self, inp: Input) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not inp.campaign_id:
            return Output(success=False, error="campaign_id required")

        fields = {}
        if inp.notes:
            fields["notes"] = inp.notes
        if inp.escalation_level > 0:
            fields["escalation_level"] = inp.escalation_level

        if not fields:
            return Output(success=False, error="No fields to update")

        campaign = await NudgeCampaignsRepo().update(inp.campaign_id, **fields)
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {inp.campaign_id}")
        return Output(success=True, campaign=campaign)

    async def _get_effectiveness(self, days: int) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        report = await NudgeCampaignsRepo().get_effectiveness_report(days=days)
        return Output(success=True, effectiveness={"by_tactic": report}, count=len(report))

    async def _pause_campaign(self, inp: Input) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not inp.campaign_id:
            return Output(success=False, error="campaign_id required")

        pause_hours = inp.hours or 4
        until = datetime.now(UTC) + timedelta(hours=pause_hours)

        campaign = await NudgeCampaignsRepo().update(
            inp.campaign_id,
            paused_until=until,
            notes=inp.notes or f"Paused for {pause_hours}h",
        )
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {inp.campaign_id}")
        return Output(success=True, campaign=campaign)

    async def _resume_campaign(self, campaign_id: str) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        if not campaign_id:
            return Output(success=False, error="campaign_id required")

        campaign = await NudgeCampaignsRepo().update(campaign_id, paused_until=None)
        if not campaign:
            return Output(success=False, error=f"Campaign not found: {campaign_id}")
        return Output(success=True, campaign=campaign)

    async def _reset_daily(self) -> Output:
        from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

        await NudgeCampaignsRepo().reset_daily_counts()
        return Output(success=True)


if __name__ == "__main__":
    NudgeStrategistTool.run()
