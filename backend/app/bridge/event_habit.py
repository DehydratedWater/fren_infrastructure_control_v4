"""Event → Habit bridge — auto-complete habit occurrences from detected events.

v3 parity port of ``scripts/event_habit_bridge.py``, restructured per project
law: the MATCHING POLICY (event → habit) is a pure function driven by a
JSON-able policy dict (``DEFAULT_POLICY``), so the decision logic is
autoloop-optimisable via ``src.improvement.autoresearch`` — see
``app/bridge/event_habit_probes.py`` (mirrors ``app/delivery/gate_probes.py``).
The promoted winner lands in ``.oac/promoted/policy:event_habit_bridge.json``
where :func:`active_policy` picks it up; otherwise DEFAULT_POLICY ships.

Only :func:`run_bridge` touches the DB, and only through the existing repos:
events are READ (``EventsRepo.list_since_id``), habit occurrences are WRITTEN
exclusively through ``HabitsRepo.create_occurrence`` /
``HabitsRepo.complete_occurrence`` (the same methods the habit manager uses —
completing fires the DB ``update_habit_streak`` trigger), and the cursor lives
in ``agent_notes`` like every other periodic extractor.
"""

from __future__ import annotations

import logging
from datetime import UTC, date as date_type, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

COMPONENT_ID = "policy:event_habit_bridge"

STATE_KEY = "event_habit_bridge_cursor"

# The baseline policy. JSON-able by construction: this exact dict is the
# autoloop's baseline_definition and mutated copies are what get promoted.
DEFAULT_POLICY: dict = {
    # Fallback map: event category → keywords searched in the habit's
    # title/category/tags (used when dynamic matching misses). Mixed PL/EN —
    # the user logs events in both languages.
    "category_synonyms": {
        "walk": ["walking", "spacer", "stroll"],
        "workout": ["gym", "exercise", "training", "trening"],
        "exercise": ["workout", "gym", "training", "trening"],
        "medication": ["meds", "leki", "lek", "medicine"],
        "shower": ["hygiene", "prysznic"],
        "drinking": ["water", "hydrat", "nawodnie"],
        "eating": ["meal", "posiłek", "jedzenie", "food"],
    },
    # Words shorter than this never count for event-title ↔ habit-title overlap
    # (v3 used > 3 chars).
    "min_title_word_len": 4,
    # Subcategory must be at least this long to match into the habit title
    # (v3 used > 2 chars).
    "min_subcategory_len": 3,
    # Only auto-complete an occurrence when the event's date is within this
    # many days of "today" — a backfilled weeks-old event must not silently
    # flip a historic streak.
    "max_event_age_days": 2,
    # Per-strategy confidences + the threshold a match must clear.
    "confidence_category": 0.9,
    "confidence_subcategory": 0.85,
    "confidence_title_overlap": 0.7,
    "confidence_synonym": 0.65,
    "confidence_threshold": 0.6,
}


class MatchDecision(BaseModel):
    """Outcome of matching one event against one habit."""

    matched: bool
    confidence: float = 0.0
    reason: str = ""


class CompletionDecision(BaseModel):
    """One habit occurrence the bridge has decided to complete."""

    habit_id: str
    habit_title: str
    event_id: str
    event_title: str
    event_category: str
    event_date: str  # YYYY-MM-DD
    occurrence_id: str
    confidence: float
    reason: str

    @property
    def notes(self) -> str:
        return f"Auto-completed from {self.event_category} event: {self.event_title}"


# ── pure matching policy ─────────────────────────────────────────────────────


def match_event_to_habit(event: dict, habit: dict, policy: dict | None = None) -> MatchDecision:
    """Match one event to one habit using the policy dict. Pure — no I/O.

    Strategies (highest-confidence hit wins):
      1. event category appears in the habit's title/category/tags
      2. event subcategory appears in the habit title (e.g. "concerta" →
         habit "take concerta")
      3. event-title words overlap habit-title words (>= min_title_word_len)
      4. category synonyms from the fallback map appear in the habit text
    The decision only ``matched`` when the best confidence clears
    ``confidence_threshold``.
    """
    p = {**DEFAULT_POLICY, **(policy or {})}

    category = (event.get("category") or "").lower()
    event_sub = (event.get("subcategory") or "").lower()
    event_title = (event.get("title") or "").lower()

    habit_title = (habit.get("title") or "").lower()
    habit_category = (habit.get("category") or "").lower()
    habit_tags = [str(t).lower() for t in (habit.get("tags") or [])]
    habit_text = f"{habit_title} {habit_category} {' '.join(habit_tags)}"

    candidates: list[tuple[float, str]] = []

    if category and category in habit_text:
        candidates.append((float(p["confidence_category"]), f"category '{category}' in habit text"))

    if event_sub and len(event_sub) >= int(p["min_subcategory_len"]) and event_sub in habit_title:
        candidates.append((float(p["confidence_subcategory"]), f"subcategory '{event_sub}' in habit title"))

    min_len = int(p["min_title_word_len"])
    event_words = {w for w in event_title.split() if len(w) >= min_len}
    habit_words = {w for w in habit_title.split() if len(w) >= min_len}
    overlap = event_words & habit_words
    if overlap:
        candidates.append((float(p["confidence_title_overlap"]), f"title words {sorted(overlap)} overlap"))

    synonyms = (p.get("category_synonyms") or {}).get(category, [])
    hits = [syn for syn in synonyms if syn in habit_text]
    if hits:
        candidates.append((float(p["confidence_synonym"]), f"synonyms {hits} for '{category}' in habit text"))

    if not candidates:
        return MatchDecision(matched=False, reason="no strategy matched")

    confidence, reason = max(candidates, key=lambda c: c[0])
    if confidence < float(p["confidence_threshold"]):
        return MatchDecision(matched=False, confidence=confidence, reason=f"below threshold: {reason}")
    return MatchDecision(matched=True, confidence=confidence, reason=reason)


def event_date_str(event: dict) -> str:
    """Extract the YYYY-MM-DD date for an event (v3 logic, pure)."""
    if event.get("date"):
        d = event["date"]
        return d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
    if event.get("occurred_at"):
        occ = event["occurred_at"]
        if hasattr(occ, "date"):
            return occ.date().isoformat()
        return str(occ)[:10]
    return datetime.now(UTC).strftime("%Y-%m-%d")


def decide_completions(
    events: list[dict],
    habits: list[dict],
    policy: dict | None = None,
    *,
    today: date_type | None = None,
) -> list[CompletionDecision]:
    """The whole bridge decision, pure: events × habits → occurrence completions.

    Applies the time-window gate (``max_event_age_days`` around *today*) and
    dedups to ONE completion per (habit_id, event_date) — two walk events on
    the same day complete the daily-walk habit once.
    """
    p = {**DEFAULT_POLICY, **(policy or {})}
    today = today or datetime.now(UTC).date()
    max_age = int(p["max_event_age_days"])

    out: list[CompletionDecision] = []
    seen: set[tuple[str, str]] = set()
    for event in events:
        ev_date = event_date_str(event)
        try:
            age_days = abs((today - date_type.fromisoformat(ev_date)).days)
        except ValueError:
            continue
        if age_days > max_age:
            continue
        for habit in habits:
            decision = match_event_to_habit(event, habit, p)
            if not decision.matched:
                continue
            habit_id = str(habit.get("habit_id") or "")
            key = (habit_id, ev_date)
            if not habit_id or key in seen:
                continue
            seen.add(key)
            out.append(
                CompletionDecision(
                    habit_id=habit_id,
                    habit_title=str(habit.get("title") or ""),
                    event_id=str(event.get("event_id") or ""),
                    event_title=str(event.get("title") or ""),
                    event_category=str(event.get("category") or ""),
                    event_date=ev_date,
                    occurrence_id=f"occ_{habit_id}_{ev_date}",
                    confidence=decision.confidence,
                    reason=decision.reason,
                )
            )
    return out


# ── promoted-policy loading (mirrors app/delivery/gate.py) ───────────────────

_POLICY_CACHE: dict[str, dict] = {}


def active_policy(project_root: Path | None = None) -> dict:
    """The shipping policy: the promoted ``policy:event_habit_bridge``
    definition from ``.oac/promoted/`` when one exists, else DEFAULT_POLICY.
    Cached per-process; degrades to the baseline on any load error.
    """
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    key = str(root)
    if key not in _POLICY_CACHE:
        policy = dict(DEFAULT_POLICY)
        try:
            from src.improvement.snapshot import load_promoted_definition

            promoted = load_promoted_definition(COMPONENT_ID, root)
            if promoted:
                policy = {**DEFAULT_POLICY, **promoted}
        except Exception:  # noqa: BLE001 — degrade to baseline, never crash the cron
            pass
        _POLICY_CACHE[key] = policy
    return _POLICY_CACHE[key]


def _clear_policy_cache() -> None:
    """Test/CLI hook: force the next active_policy() call to re-read disk."""
    _POLICY_CACHE.clear()


# ── DB runner (the only impure part — plain plumbing around the policy) ──────


async def _get_cursor() -> int:
    from app.db.repos.agent_notes import AgentNotesRepo

    note = await AgentNotesRepo().get(STATE_KEY)
    if note and note.get("note_value"):
        val = note["note_value"]
        if isinstance(val, dict):
            return int(val.get("last_event_id", 0))
        return int(val) if str(val).isdigit() else 0
    return 0


async def _set_cursor(event_id: int) -> None:
    from app.db.repos.agent_notes import AgentNotesRepo

    await AgentNotesRepo().set(STATE_KEY, {"last_event_id": event_id}, expires_hours=8760)


async def run_bridge(policy: dict | None = None) -> dict[str, int]:
    """One cron tick: fetch new events since the cursor, apply the policy,
    complete matching habit occurrences through the habits repo.
    """
    from app.db.repos.events import EventsRepo
    from app.db.repos.habits import HabitsRepo

    policy = policy if policy is not None else active_policy()

    cursor = await _get_cursor()
    events = await EventsRepo().list_since_id(cursor, limit=200)
    if not events:
        logger.info("No new events since cursor %d", cursor)
        return {"events": 0, "completions": 0, "skipped": 0}

    max_id = max(int(e["id"]) for e in events)
    habits_repo = HabitsRepo()
    habits = await habits_repo.list(status="active")
    if not habits:
        logger.info("No active habits to match against")
        await _set_cursor(max_id)
        return {"events": len(events), "completions": 0, "skipped": 0}

    completions = 0
    skipped = 0
    for decision in decide_completions(events, habits, policy):
        # Ensure the occurrence row exists (idempotent: ON CONFLICT DO NOTHING).
        await habits_repo.create_occurrence(decision.occurrence_id, decision.habit_id, decision.event_date)

        occs = await habits_repo.get_occurrences(decision.habit_id, limit=5)
        already_done = any(
            o.get("occurrence_id") == decision.occurrence_id and o.get("status") == "completed" for o in occs
        )
        if already_done:
            skipped += 1
            continue

        result = await habits_repo.complete_occurrence(decision.occurrence_id, notes=decision.notes)
        if result:
            completions += 1
            logger.info(
                "Completed habit '%s' for %s (%s, confidence=%.2f)",
                decision.habit_title,
                decision.event_date,
                decision.reason,
                decision.confidence,
            )

    await _set_cursor(max_id)
    logger.info(
        "Bridge complete: %d completions, %d already done, %d events processed",
        completions,
        skipped,
        len(events),
    )
    return {"events": len(events), "completions": completions, "skipped": skipped}
