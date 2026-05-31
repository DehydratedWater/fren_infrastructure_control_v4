"""Goal Progress Auto-Updater — automatically update goal progress from evidence."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx
from src import ScriptTool
from pydantic import BaseModel, Field

from app.db.repos.activity_blocks import ActivityBlocksRepo
from app.db.repos.goal_auto_updater import GoalAutoUpdaterRepo
from app.db.repos.goals import GoalsRepo
from app.db.repos.user_config import UserConfigRepo
from app.db.session import close_engine, fetch_all, get_async_session
from app.vllm_resolve import get_llm_endpoint

_base, _model = get_llm_endpoint()
_LLM_API_URL = f"{_base}/chat/completions"
_LLM_MODEL = _model

_LLM_MATCH_PROMPT = """\
You are judging whether recent events/activities/habits contribute to a goal.

Goal: {goal_title}
{goal_description}
Matching question: {matching_question}

For each numbered item below, answer YES or NO — does this item represent progress toward the goal?
Be generous with indirect contributions (e.g. a walk counts for a weight loss goal).
But reject clearly unrelated items (e.g. vaccination doesn't count for a coding goal).

{evidence_list}

Return ONLY a JSON array of the item numbers that are YES, like: [1, 3, 5]
If none match, return: []"""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL)
    if text.startswith("Thinking Process:"):
        text = re.sub(r"^Thinking Process:.*?(?=^#{1,2} )", "", text, flags=re.DOTALL | re.MULTILINE)
    if text.startswith("Thinking Process:"):
        text = ""
    return text.strip()


async def _llm_match_evidence(
    goal: dict,
    evidence_items: list[Evidence],
) -> set[int]:
    """Ask LLM which evidence items match a goal. Returns set of matching indices."""
    meta = goal.get("metadata") or {}
    tracking = meta.get("tracking") or {}
    question = tracking.get("matching_question", "")
    if not question:
        return set()

    evidence_lines = []
    for i, ev in enumerate(evidence_items):
        evidence_lines.append(f"{i + 1}. [{ev.type}] {ev.summary}")

    if not evidence_lines:
        return set()

    prompt = _LLM_MATCH_PROMPT.format(
        goal_title=goal["title"],
        goal_description=f"Description: {goal['description']}" if goal.get("description") else "",
        matching_question=question,
        evidence_list="\n".join(evidence_lines),
    )

    payload = {
        "model": _LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16384,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(_LLM_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = _strip_thinking(data["choices"][0]["message"]["content"])
        # Extract JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            indices = json.loads(text[start:end])
            return {i - 1 for i in indices if isinstance(i, int)}  # Convert 1-indexed to 0-indexed
    except Exception:
        pass
    return set()


@dataclass
class MatchResult:
    confidence: float
    reason: str
    extra: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


@dataclass
class Evidence:
    type: str
    id: str
    summary: str
    data: dict[str, Any]
    db_id: int


class Input(BaseModel):
    command: str = Field(description="run|logs|config-get|config-set|state|state-reset|dry-run")
    lookback_hours: int = Field(default=24, description="Hours to look back for evidence")
    goal_id: str = Field(default="", description="Filter logs by goal ID")
    evidence_type: str = Field(default="", description="Filter logs by evidence type")
    limit: int = Field(default=50, description="Limit for logs")
    config_key: str = Field(default="", description="Config key for set/get")
    config_value: str = Field(default="", description="Config value to set")


class Output(BaseModel):
    success: bool = True
    updates_made: int = 0
    updates_skipped: int = 0
    logs: list[dict] = Field(default_factory=list)
    config: dict | None = None
    state: dict | None = None
    proposed_updates: list[dict] = Field(default_factory=list)
    error: str = ""
    message: str = ""


class GoalProgressAutoUpdaterTool(ScriptTool[Input, Output]):
    name = "goal_progress_auto_updater"
    description = "Automatically update goal progress based on activities, events, habits, and todos"

    DEFAULT_CONFIG: ClassVar[dict] = {
        "enabled": True,
        "sensitivity": "medium",
        "max_auto_updates_per_day": 10,
        "max_auto_progress_cap": 80,
        "lookback_hours": 24,
        "min_confidence_threshold": 0.6,
        "trigger_mode": "hybrid",
        "cron_interval_minutes": 60,
        "activity_mappings": {},
        "event_mappings": {},
        "progress_rules": {
            "activity_duration_based": {"enabled": True, "base_minutes": 60, "progress_per_base": 5},
            "event_count_based": {"enabled": True, "progress_per_event": 2},
            "habit_completion": {"enabled": True, "progress_per_completion": 3},
            "todo_completion": {"enabled": True, "progress_per_completion": 10},
        },
        "categories_to_watch": [],
        "activity_types_to_watch": [],
        "excluded_goal_levels": [],
        "excluded_goal_keywords": [],
        "log_all_matches": False,
        "require_frozen_activities": True,
        "notification_on_update": False,
    }

    SENSITIVITY_SETTINGS: ClassVar[dict] = {
        "low": {"min_confidence": 0.85, "multiplier": 0.5},
        "medium": {"min_confidence": 0.6, "multiplier": 1.0},
        "high": {"min_confidence": 0.4, "multiplier": 1.5},
    }

    def execute(self, inp: Input) -> Output:
        try:
            return asyncio.run(self._dispatch(inp))
        finally:
            with contextlib.suppress(Exception):
                asyncio.run(close_engine())

    async def _dispatch(self, inp: Input) -> Output:
        if inp.command == "run":
            return await self._run_update_cycle(inp)
        if inp.command == "logs":
            return await self._get_logs(inp)
        if inp.command == "config-get":
            return await self._get_config(inp)
        if inp.command == "config-set":
            return await self._set_config(inp)
        if inp.command == "state":
            return await self._get_state()
        if inp.command == "state-reset":
            return await self._reset_state()
        if inp.command == "dry-run":
            return await self._dry_run(inp)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    # =========================================================================
    # Main update cycle
    # =========================================================================

    async def _run_update_cycle(self, inp: Input) -> Output:
        config_repo = UserConfigRepo()
        state_repo = GoalAutoUpdaterRepo()

        config = await self._load_config(config_repo)
        if not config.get("enabled", True):
            return Output(success=True, message="Auto-updater is disabled")

        lookback = inp.lookback_hours or config.get("lookback_hours", 24)
        triggered_by = "manual" if inp.command == "run" else "cron"

        # Fetch evidence
        activities = await self._fetch_activities(config, lookback)
        events = await self._fetch_events(config, lookback)
        habits = await self._fetch_habit_completions(config, lookback)
        todos = await self._fetch_completed_todos(config, lookback)

        all_evidence = activities + events + habits + todos

        if not all_evidence:
            return Output(success=True, message="No evidence found", updates_made=0)

        # Fetch active goals
        goals = await self._fetch_active_goals(config)
        if not goals:
            return Output(success=True, message="No active goals", updates_made=0)

        # Pre-compute LLM matches for goals that have matching questions
        llm_matches: dict[str, set[int]] = {}  # goal_id → set of matching evidence indices
        for goal in goals:
            meta = goal.get("metadata") or {}
            if meta.get("tracking", {}).get("matching_question"):
                matched_indices = await _llm_match_evidence(goal, all_evidence)
                if matched_indices:
                    llm_matches[goal["goal_id"]] = matched_indices

        # Process matches and updates
        updates_made = 0
        updates_skipped = 0
        last_activity_id = 0
        last_event_id = 0
        last_habit_id = 0
        last_todo_id = 0

        for goal in goals:
            for ev_idx, evidence in enumerate(all_evidence):
                # Track max IDs for state update
                if evidence.type == "activity":
                    last_activity_id = max(last_activity_id, evidence.db_id)
                elif evidence.type == "event":
                    last_event_id = max(last_event_id, evidence.db_id)
                elif evidence.type == "habit":
                    last_habit_id = max(last_habit_id, evidence.db_id)
                elif evidence.type == "todo":
                    last_todo_id = max(last_todo_id, evidence.db_id)

                # Use LLM match if available, otherwise fall back to rule-based
                if goal["goal_id"] in llm_matches:
                    if ev_idx in llm_matches[goal["goal_id"]]:
                        match = MatchResult(confidence=0.85, reason="llm_match")
                    else:
                        # Still check explicit links (habit/todo with linked_goal_id)
                        match = self._match_evidence_to_goal(evidence, goal, config)
                        if match and match.reason not in ("explicit_habit_link", "explicit_todo_link"):
                            match = None  # LLM said no, trust it over keyword heuristics
                else:
                    match = self._match_evidence_to_goal(evidence, goal, config)
                if not match:
                    continue

                if match.confidence < config.get("min_confidence_threshold", 0.6):
                    if config.get("log_all_matches", False):
                        await state_repo.log_skipped(
                            goal_id=goal["goal_id"],
                            current_progress=goal["progress_percent"],
                            evidence_type=evidence.type,
                            evidence_id=evidence.id,
                            evidence_summary=evidence.summary,
                            match_reason="below_threshold",
                            confidence_score=match.confidence,
                            skip_reason=f"Confidence {match.confidence:.2f} below threshold",
                            triggered_by=triggered_by,
                        )
                    updates_skipped += 1
                    continue

                # Check if update is allowed
                can_update, skip_reason = await self._can_update_progress(goal, config, state_repo)
                if not can_update:
                    await state_repo.log_skipped(
                        goal_id=goal["goal_id"],
                        current_progress=goal["progress_percent"],
                        evidence_type=evidence.type,
                        evidence_id=evidence.id,
                        evidence_summary=evidence.summary,
                        match_reason=match.reason,
                        confidence_score=match.confidence,
                        skip_reason=skip_reason,
                        triggered_by=triggered_by,
                    )
                    updates_skipped += 1
                    continue

                # Calculate and apply update
                delta = self._calculate_progress_delta(match, evidence, config)
                new_progress = min(goal["progress_percent"] + delta, 100)

                goals_repo = GoalsRepo()
                await goals_repo.update(goal["goal_id"], progress_percent=new_progress)

                # Log the update
                await state_repo.log_update(
                    goal_id=goal["goal_id"],
                    previous_progress=goal["progress_percent"],
                    new_progress=new_progress,
                    evidence_type=evidence.type,
                    evidence_id=evidence.id,
                    evidence_summary=evidence.summary,
                    match_reason=match.reason,
                    confidence_score=match.confidence,
                    triggered_by=triggered_by,
                )

                updates_made += 1

        # Update state
        await state_repo.update_state(
            last_processed_activity_id=last_activity_id,
            last_processed_event_id=last_event_id,
            last_processed_habit_occurrence_id=last_habit_id,
            last_processed_todo_id=last_todo_id,
            total_updates_made=(await state_repo.get_state()).get("total_updates_made", 0) + updates_made,
        )

        return Output(
            success=True,
            updates_made=updates_made,
            updates_skipped=updates_skipped,
            message=f"Processed {len(all_evidence)} evidence items against {len(goals)} goals",
        )

    # =========================================================================
    # Evidence fetching
    # =========================================================================

    async def _fetch_activities(self, config: dict, lookback_hours: int) -> list[Evidence]:
        repo = ActivityBlocksRepo()
        blocks = await repo.get_recent_blocks(hours=lookback_hours)

        evidence_list = []
        for block in blocks:
            if config.get("require_frozen_activities", True) and block.get("frozen_at") is None:
                continue

            activity_type = block.get("activity_type", "")
            if config.get("activity_types_to_watch") and activity_type not in config["activity_types_to_watch"]:
                continue

            duration = None
            if block.get("ended_at") and block.get("started_at"):
                duration = int((block["ended_at"] - block["started_at"]).total_seconds() / 60)

            evidence_list.append(
                Evidence(
                    type="activity",
                    id=f"activity_{block['id']}",
                    db_id=block["id"],
                    summary=f"{activity_type}: {block.get('title', '')[:50]}",
                    data={
                        "activity_type": activity_type,
                        "title": block.get("title", ""),
                        "description": block.get("description", ""),
                        "tags": block.get("tags", []),
                        "duration_minutes": duration,
                    },
                )
            )

        return evidence_list

    async def _fetch_events(self, config: dict, lookback_hours: int) -> list[Evidence]:
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

        async with get_async_session() as s:
            events = await fetch_all(
                s,
                """
                SELECT * FROM events
                WHERE occurred_at >= :cutoff
                ORDER BY occurred_at DESC
            """,
                {"cutoff": cutoff},
            )

        evidence_list = []
        categories_to_watch = config.get("categories_to_watch", [])
        for event in events:
            if categories_to_watch and event.get("category") not in categories_to_watch:
                continue

            evidence_list.append(
                Evidence(
                    type="event",
                    id=event.get("event_id", f"event_{event['id']}"),
                    db_id=event["id"],
                    summary=f"{event.get('category')}: {event.get('title', '')[:50]}",
                    data={
                        "category": event.get("category"),
                        "subcategory": event.get("subcategory"),
                        "title": event.get("title", ""),
                        "value": event.get("value"),
                        "duration_minutes": event.get("duration_minutes"),
                    },
                )
            )

        return evidence_list

    async def _fetch_habit_completions(self, config: dict, lookback_hours: int) -> list[Evidence]:
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

        async with get_async_session() as s:
            occurrences = await fetch_all(
                s,
                """
                SELECT ho.*, h.title as habit_title, h.linked_goal_id
                FROM habit_occurrences ho
                JOIN habits h ON ho.habit_id = h.habit_id
                WHERE ho.status = 'completed'
                  AND ho.completed_at >= :cutoff
                ORDER BY ho.completed_at DESC
            """,
                {"cutoff": cutoff},
            )

        evidence_list = []
        for occ in occurrences:
            evidence_list.append(
                Evidence(
                    type="habit",
                    id=occ.get("occurrence_id", f"habit_occ_{occ['id']}"),
                    db_id=occ["id"],
                    summary=f"Habit completed: {occ.get('habit_title', '')[:50]}",
                    data={
                        "habit_id": occ.get("habit_id"),
                        "habit_title": occ.get("habit_title"),
                        "linked_goal_id": occ.get("linked_goal_id"),
                        "scheduled_date": str(occ.get("scheduled_date")),
                    },
                )
            )

        return evidence_list

    async def _fetch_completed_todos(self, config: dict, lookback_hours: int) -> list[Evidence]:
        cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

        async with get_async_session() as s:
            todos = await fetch_all(
                s,
                """
                SELECT * FROM todos
                WHERE status = 'completed'
                  AND completed_at >= :cutoff
                ORDER BY completed_at DESC
            """,
                {"cutoff": cutoff},
            )

        evidence_list = []
        for todo in todos:
            evidence_list.append(
                Evidence(
                    type="todo",
                    id=todo.get("todo_id", f"todo_{todo['id']}"),
                    db_id=todo["id"],
                    summary=f"Todo completed: {todo.get('title', '')[:50]}",
                    data={
                        "todo_id": todo.get("todo_id"),
                        "title": todo.get("title", ""),
                        "linked_goal_id": todo.get("linked_goal_id"),
                        "goal_alignment": todo.get("goal_alignment"),
                    },
                )
            )

        return evidence_list

    async def _fetch_active_goals(self, config: dict) -> list[dict]:
        repo = GoalsRepo()
        goals = await repo.list_active()

        excluded_levels = config.get("excluded_goal_levels", [])
        excluded_keywords = config.get("excluded_goal_keywords", [])

        filtered = []
        for goal in goals:
            if goal["level"] in excluded_levels:
                continue
            if any(kw.lower() in goal["title"].lower() for kw in excluded_keywords):
                continue
            filtered.append(goal)

        return filtered

    # =========================================================================
    # Matching logic
    # =========================================================================

    def _match_evidence_to_goal(self, evidence: Evidence, goal: dict, config: dict) -> MatchResult | None:
        goal_title = goal["title"].lower()
        goal_metadata = goal.get("metadata", {})
        goal_tags = goal_metadata.get("tags", [])

        if evidence.type == "activity":
            return self._match_activity_to_goal(evidence, goal_title, goal_tags, config)
        elif evidence.type == "event":
            return self._match_event_to_goal(evidence, goal_title, config)
        elif evidence.type == "habit":
            return self._match_habit_to_goal(evidence, goal, config)
        elif evidence.type == "todo":
            return self._match_todo_to_goal(evidence, goal)

        return None

    def _match_activity_to_goal(
        self, evidence: Evidence, goal_title: str, goal_tags: list, config: dict
    ) -> MatchResult | None:
        data = evidence.data
        activity_tags = data.get("tags", [])
        activity_type = data.get("activity_type", "")
        activity_title = data.get("title", "").lower()

        # Tag match (highest confidence)
        if set(goal_tags) & set(activity_tags):
            return MatchResult(confidence=0.95, reason="tag_match")

        # Activity type mapping
        activity_mappings = config.get("activity_mappings", {})
        mapped_goals = activity_mappings.get(activity_type, [])
        for keyword in mapped_goals:
            if keyword.lower() in goal_title:
                return MatchResult(confidence=0.8, reason="activity_type_mapping")

        # Keyword in title/description — skip common verbs
        _act_stop = frozenset(
            {
                "with",
                "from",
                "that",
                "this",
                "have",
                "been",
                "will",
                "about",
                "make",
                "take",
                "more",
                "into",
                "just",
                "also",
                "complete",
                "create",
                "update",
                "track",
                "start",
                "build",
                "plan",
            }
        )
        keyword_matches = 0
        for word in goal_title.split():
            if len(word) > 3 and word not in _act_stop and word in activity_title:
                keyword_matches += 1

        if keyword_matches >= 2:
            return MatchResult(confidence=0.7, reason="keyword_match")
        if keyword_matches == 1 and config.get("sensitivity") == "high":
            return MatchResult(confidence=0.5, reason="single_keyword_match")

        return None

    # Built-in semantic associations: event categories that relate to goal keywords.
    # These are always checked, so fitness events naturally connect to fitness/weight goals
    # without requiring manual event_mappings config.
    _BUILTIN_CATEGORY_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "walk": ["weight", "fitness", "health", "exercise", "kg", "lose", "active", "steps", "cardio"],
        "workout": ["weight", "fitness", "health", "exercise", "kg", "lose", "muscle", "strength", "gym"],
        "exercise": ["weight", "fitness", "health", "exercise", "kg", "lose", "muscle", "active"],
        "weight": ["weight", "kg", "lose", "fitness", "health", "body"],
        "eating": ["weight", "diet", "nutrition", "health", "kg", "calories"],
        "drinking": ["hydration", "water", "health"],
        "medication": ["health", "medication", "treatment"],
        "shower": ["hygiene", "routine", "self-care"],
    }

    def _match_event_to_goal(self, evidence: Evidence, goal_title: str, config: dict) -> MatchResult | None:
        data = evidence.data
        category = data.get("category", "").lower()
        subcategory = (data.get("subcategory") or "").lower()
        event_title = data.get("title", "").lower()

        # Explicit category mapping from config (highest priority)
        event_mappings = config.get("event_mappings", {})
        mapped_goals = event_mappings.get(category, [])
        for keyword in mapped_goals:
            if keyword.lower() in goal_title:
                return MatchResult(confidence=0.85, reason="event_category_mapping")

        # Built-in semantic category matching — connects e.g. walk events to weight goals
        builtin_keywords = self._BUILTIN_CATEGORY_KEYWORDS.get(category, [])
        for keyword in builtin_keywords:
            if keyword in goal_title:
                return MatchResult(confidence=0.75, reason="builtin_category_match")

        # Subcategory match
        if subcategory and subcategory in goal_title:
            return MatchResult(confidence=0.8, reason="subcategory_match")

        # Event title keyword match — require >4 chars and exclude common verbs/stopwords
        _event_stop = frozenset(
            {
                "with",
                "from",
                "that",
                "this",
                "have",
                "been",
                "will",
                "about",
                "make",
                "take",
                "more",
                "into",
                "just",
                "also",
                "some",
                "than",
                "them",
                "then",
                "each",
                "when",
                "what",
                "your",
                "their",
                "complete",
                "create",
                "update",
                "track",
                "start",
                "build",
                "plan",
                "daily",
                "week",
                "month",
                "year",
                "after",
                "before",
                "during",
                "until",
                "took",
                "went",
                "made",
                "done",
                "back",
                "home",
                "good",
                "well",
                "came",
                "evening",
                "morning",
                "night",
            }
        )
        for word in goal_title.split():
            if len(word) > 4 and word not in _event_stop and word in event_title:
                return MatchResult(confidence=0.65, reason="event_title_match")

        return None

    # Built-in semantic associations for habit titles → goal keywords.
    # If any word in the habit title matches a key, those keywords are checked against the goal.
    _BUILTIN_HABIT_KEYWORDS: ClassVar[dict[str, list[str]]] = {
        "walk": ["weight", "fitness", "health", "kg", "lose", "active", "cardio"],
        "workout": ["weight", "fitness", "health", "kg", "lose", "muscle", "strength"],
        "exercise": ["weight", "fitness", "health", "kg", "lose", "active"],
        "run": ["weight", "fitness", "health", "kg", "cardio", "active"],
        "gym": ["weight", "fitness", "health", "kg", "muscle", "strength"],
        "meditat": ["mental", "mindful", "stress", "calm", "focus"],
        "read": ["reading", "book", "learn", "knowledge"],
        "water": ["hydration", "health", "drink"],
        "sleep": ["sleep", "rest", "health", "recovery"],
    }

    def _match_habit_to_goal(self, evidence: Evidence, goal: dict, config: dict) -> MatchResult | None:
        data = evidence.data
        linked_goal_id = data.get("linked_goal_id")
        habit_title = (data.get("habit_title") or "").lower()
        goal_title = goal["title"].lower()

        # Explicit link (highest confidence)
        if linked_goal_id == goal.get("goal_id"):
            return MatchResult(confidence=1.0, reason="explicit_habit_link")

        # Title keyword match
        for word in goal_title.split():
            if len(word) > 3 and word in habit_title:
                return MatchResult(confidence=0.7, reason="habit_title_match")

        # Built-in semantic habit matching — e.g. habit "daily walk" → goal "reach 80kg"
        for habit_key, goal_keywords in self._BUILTIN_HABIT_KEYWORDS.items():
            if habit_key in habit_title:
                for keyword in goal_keywords:
                    if keyword in goal_title:
                        return MatchResult(confidence=0.7, reason="builtin_habit_match")

        return None

    def _match_todo_to_goal(self, evidence: Evidence, goal: dict) -> MatchResult | None:
        data = evidence.data
        linked_goal_id = data.get("linked_goal_id")

        # Explicit link
        if linked_goal_id == goal.get("goal_id"):
            alignment = data.get("goal_alignment", {})
            weight = alignment.get("contribution_weight", 0.5) if alignment else 0.5
            return MatchResult(confidence=1.0, reason="explicit_todo_link", extra={"weight": weight})

        return None

    # =========================================================================
    # Progress calculation
    # =========================================================================

    def _calculate_progress_delta(self, match: MatchResult, evidence: Evidence, config: dict) -> int:
        rules = config.get("progress_rules", {})
        sensitivity = config.get("sensitivity", "medium")
        sensitivity_mult = self.SENSITIVITY_SETTINGS.get(sensitivity, {}).get("multiplier", 1.0)

        if match.reason in ("tag_match", "activity_type_mapping", "event_category_mapping", "llm_match"):
            # Activity with duration
            if evidence.type == "activity" and evidence.data.get("duration_minutes"):
                dur_rules = rules.get("activity_duration_based", {})
                if dur_rules.get("enabled", True):
                    base_minutes = dur_rules.get("base_minutes", 60)
                    progress_per_base = dur_rules.get("progress_per_base", 5)
                    ratio = evidence.data["duration_minutes"] / base_minutes
                    return int(min(progress_per_base * ratio, 15) * sensitivity_mult)
            # Event-based / LLM-matched
            event_rules = rules.get("event_count_based", {})
            if event_rules.get("enabled", True):
                return int(event_rules.get("progress_per_event", 2) * sensitivity_mult)

        elif match.reason in ("builtin_category_match", "builtin_habit_match"):
            # Semantic category matches — treat like event-based progress
            event_rules = rules.get("event_count_based", {})
            if event_rules.get("enabled", True):
                return int(event_rules.get("progress_per_event", 2) * sensitivity_mult)

        elif match.reason in ("keyword_match", "subcategory_match", "habit_title_match"):
            event_rules = rules.get("event_count_based", {})
            if event_rules.get("enabled", True):
                return int(event_rules.get("progress_per_event", 2) * 0.7 * sensitivity_mult)

        elif match.reason == "explicit_habit_link":
            habit_rules = rules.get("habit_completion", {})
            if habit_rules.get("enabled", True):
                return int(habit_rules.get("progress_per_completion", 3) * sensitivity_mult)

        elif match.reason == "explicit_todo_link":
            todo_rules = rules.get("todo_completion", {})
            if todo_rules.get("enabled", True):
                weight = match.extra.get("weight", 0.5)
                return int(todo_rules.get("progress_per_completion", 10) * weight * sensitivity_mult)

        elif match.reason in ("single_keyword_match",):
            event_rules = rules.get("event_count_based", {})
            if event_rules.get("enabled", True):
                return int(event_rules.get("progress_per_event", 2) * 0.3 * sensitivity_mult)

        # event_title_match is intentionally excluded — too noisy (common verbs match unrelated goals)
        return 0

    # =========================================================================
    # Rate limiting and caps
    # =========================================================================

    async def _can_update_progress(self, goal: dict, config: dict, state_repo: GoalAutoUpdaterRepo) -> tuple[bool, str]:
        from datetime import date

        current_progress = goal.get("progress_percent", 0)
        cap = config.get("max_auto_progress_cap", 80)

        if current_progress >= cap:
            return False, f"Progress ({current_progress}%) at auto-update cap ({cap}%)"

        today_updates = await state_repo.count_updates_today(goal["goal_id"], date.today())
        max_per_day = config.get("max_auto_updates_per_day", 10)

        if today_updates >= max_per_day:
            return False, f"Rate limit reached ({today_updates}/{max_per_day} updates today)"

        return True, ""

    # =========================================================================
    # Config and state management
    # =========================================================================

    async def _load_config(self, config_repo: UserConfigRepo | None = None) -> dict:
        if config_repo is None:
            config_repo = UserConfigRepo()

        result = await config_repo.get("goal_auto_updater_config")
        if result and result.get("config_value"):
            try:
                return json.loads(result["config_value"])
            except (json.JSONDecodeError, TypeError):
                pass
        return self.DEFAULT_CONFIG.copy()

    async def _get_config(self, inp: Input) -> Output:
        config = await self._load_config()
        if inp.config_key:
            value = config.get(inp.config_key)
            return Output(success=True, config={inp.config_key: value})
        return Output(success=True, config=config)

    async def _set_config(self, inp: Input) -> Output:
        if not inp.config_key:
            return Output(success=False, error="config_key is required")

        config_repo = UserConfigRepo()
        current = await self._load_config(config_repo)

        try:
            value = json.loads(inp.config_value)
        except (json.JSONDecodeError, TypeError):
            value = inp.config_value

        current[inp.config_key] = value

        await config_repo.set("goal_auto_updater_config", json.dumps(current))
        return Output(success=True, config=current)

    async def _get_state(self) -> Output:
        state_repo = GoalAutoUpdaterRepo()
        state = await state_repo.get_state()
        return Output(success=True, state=state)

    async def _reset_state(self) -> Output:
        state_repo = GoalAutoUpdaterRepo()
        state = await state_repo.reset_state()
        return Output(success=True, state=state, message="State reset successfully")

    # =========================================================================
    # Logs
    # =========================================================================

    async def _get_logs(self, inp: Input) -> Output:
        state_repo = GoalAutoUpdaterRepo()
        logs = await state_repo.get_logs(
            goal_id=inp.goal_id or None,
            evidence_type=inp.evidence_type or None,
            limit=inp.limit,
        )
        return Output(success=True, logs=logs)

    # =========================================================================
    # Dry run
    # =========================================================================

    async def _dry_run(self, inp: Input) -> Output:
        config_repo = UserConfigRepo()
        config = await self._load_config(config_repo)

        lookback = inp.lookback_hours or config.get("lookback_hours", 24)

        # Fetch evidence and goals
        activities = await self._fetch_activities(config, lookback)
        events = await self._fetch_events(config, lookback)
        habits = await self._fetch_habit_completions(config, lookback)
        todos = await self._fetch_completed_todos(config, lookback)

        all_evidence = activities + events + habits + todos
        goals = await self._fetch_active_goals(config)

        # Pre-compute LLM matches for dry-run too
        llm_matches: dict[str, set[int]] = {}
        for goal in goals:
            meta = goal.get("metadata") or {}
            if meta.get("tracking", {}).get("matching_question"):
                matched_indices = await _llm_match_evidence(goal, all_evidence)
                if matched_indices:
                    llm_matches[goal["goal_id"]] = matched_indices

        proposed = []
        for goal in goals:
            for ev_idx, evidence in enumerate(all_evidence):
                # Use LLM match if available
                if goal["goal_id"] in llm_matches:
                    if ev_idx in llm_matches[goal["goal_id"]]:
                        match = MatchResult(confidence=0.85, reason="llm_match")
                    else:
                        match = self._match_evidence_to_goal(evidence, goal, config)
                        if match and match.reason not in ("explicit_habit_link", "explicit_todo_link"):
                            match = None
                else:
                    match = self._match_evidence_to_goal(evidence, goal, config)
                if not match or match.confidence < config.get("min_confidence_threshold", 0.6):
                    continue

                can_update, skip_reason = await self._can_update_progress(goal, config, GoalAutoUpdaterRepo())
                if not can_update:
                    proposed.append(
                        {
                            "goal_id": goal["goal_id"],
                            "goal_title": goal["title"],
                            "current_progress": goal["progress_percent"],
                            "proposed_progress": goal["progress_percent"],
                            "delta": 0,
                            "evidence_type": evidence.type,
                            "evidence_summary": evidence.summary,
                            "match_reason": match.reason,
                            "confidence": match.confidence,
                            "skipped": True,
                            "skip_reason": skip_reason,
                        }
                    )
                    continue

                delta = self._calculate_progress_delta(match, evidence, config)
                new_progress = min(goal["progress_percent"] + delta, 100)

                proposed.append(
                    {
                        "goal_id": goal["goal_id"],
                        "goal_title": goal["title"],
                        "current_progress": goal["progress_percent"],
                        "proposed_progress": new_progress,
                        "delta": delta,
                        "evidence_type": evidence.type,
                        "evidence_summary": evidence.summary,
                        "match_reason": match.reason,
                        "confidence": match.confidence,
                        "skipped": False,
                    }
                )

        return Output(success=True, proposed_updates=proposed, message=f"Found {len(proposed)} proposed updates")


if __name__ == "__main__":
    GoalProgressAutoUpdaterTool.run()
