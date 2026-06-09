"""Proactive delivery-quality gate — a PURE, autoloop-optimisable policy.

Motivation (real v3 data): user engagement collapsed ~6x while bot volume
grew. The corpus shows three concrete failure classes this gate kills at
the single chokepoint every Twily send flows through (send_message.py):

  (a) near-duplicate proactive messages — 736 techtree "no new commits"
      variants (~250 from 10 templates); an error-fallback line ("*taps
      horn nervously* something got tangled in my spell routing") sent
      verbatim 30 times;
  (b) internal checker jargon leaked to chat — "All 6 checks passed"
      (×110), "Global cooldown is active", "idle_during_block fired";
  (c) raw error leaks — "[Render Error] f-string: expecting '}'" (×14),
      a "$(cat <<'REPORT'" heredoc, tracebacks.

Design contract: :func:`evaluate_message` is a pure function of
(text, recent, policy) with NO I/O, so the framework's autoresearch loop
(`src.improvement.autoresearch`) can probe it directly against the frozen
real-corpus cases in `app.delivery.gate_probes` and tune the numeric
knobs. Winners are promoted as component_id ``policy:delivery_gate`` into
`.oac/promoted/`, where :func:`active_policy` picks them up — the same
snapshot machinery the agent registry uses (`apply_promoted_to_tree`).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

# The repo root that holds `.oac/promoted/` (same resolution as
# app/agents/registry.py: backend/app/delivery/gate.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[3]

COMPONENT_ID = "policy:delivery_gate"

# The baseline policy. JSON-able by construction: this exact dict is the
# autoloop's baseline_definition, and mutated copies of it are what get
# promoted. Keep every value a plain str/int/float/list.
DEFAULT_POLICY: dict = {
    # How many recent Twily messages the dedup check compares against.
    "dedup_lookback": 8,
    # Normalized SequenceMatcher ratio at/above which a message is a duplicate.
    "dedup_similarity": 0.78,
    # "Nothing happened" notifications — the techtree/no-op class. A message
    # whose point is that there is no content has no business being sent.
    "noop_patterns": [
        r"\bno new commits\b",
        r"\bnothing new (?:to report|to share|since|happening|on the)\b",
        r"\bno (?:new )?updates? (?:since|today|tonight|yet|to (?:report|share)|right now)\b",
        r"\bnothing to report\b",
        r"\ball quiet on the\b",
    ],
    # Internal machinery / raw errors that must never reach chat.
    "leak_patterns": [
        r"\[Render Error",
        r"Traceback \(most recent",
        r"\$\(cat <<",
        r"<<'?(?:REPORT|EOF)'?\b",
        r"\ball \d+ checks passed\b",
        r"\bglobal cooldown\b",
        r"idle_during_block",
        r"PARSE_ERR",
        r"f-string: expecting",
        r"\bcheck(?:er)? (?:returned|exited) (?:with )?(?:code|status)\b",
        r"\bNoneType\b",
    ],
    # Anything shorter than this (stripped) is noise.
    "min_chars": 3,
}


class GateDecision(BaseModel):
    """The gate's verdict for one outbound message."""

    deliver: bool
    reason: str  # "ok" | "duplicate" | "noop" | "leak" | "too_short"
    matched: str = ""  # the pattern (leak/noop) or recent message (duplicate) that fired


@lru_cache(maxsize=512)
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _normalize(text: str) -> str:
    """Lowercase + collapse all whitespace runs — dedup compares content,
    not formatting (the corpus dupes differ by emoji spacing/newlines)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_message(
    text: str,
    recent: list[str],
    policy: dict | None = None,
) -> GateDecision:
    """Decide whether `text` should be delivered. PURE — no I/O.

    `recent` is the list of recent Twily messages, most-recent-first
    (the order ChatMessagesRepo.get_recent returns). Check order:

      1. leak  — internal jargon / raw errors (case-insensitive regex)
      2. noop  — "nothing happened" notifications
      3. too_short — below min_chars after strip
      4. duplicate — normalized SequenceMatcher ratio >= dedup_similarity
         against each of the last `dedup_lookback` recents

    Pure + dict-parameterised = directly probeable by the autoloop.
    """
    p = {**DEFAULT_POLICY, **(policy or {})}

    for pattern in p.get("leak_patterns", []):
        if _rx(pattern).search(text):
            return GateDecision(deliver=False, reason="leak", matched=pattern)

    for pattern in p.get("noop_patterns", []):
        if _rx(pattern).search(text):
            return GateDecision(deliver=False, reason="noop", matched=pattern)

    if len(text.strip()) < int(p.get("min_chars", 3)):
        return GateDecision(deliver=False, reason="too_short", matched="")

    norm = _normalize(text)
    if norm:
        threshold = float(p.get("dedup_similarity", 0.78))
        lookback = int(p.get("dedup_lookback", 8))
        for prev in recent[:lookback]:
            prev_norm = _normalize(str(prev))
            if not prev_norm:
                continue
            if SequenceMatcher(None, prev_norm, norm).ratio() >= threshold:
                return GateDecision(
                    deliver=False, reason="duplicate", matched=str(prev),
                )

    return GateDecision(deliver=True, reason="ok")


# ---- promoted-policy resolution ---------------------------------------

# Per-process cache: send_message runs as a short-lived subprocess, but the
# bot/scheduler processes also import this — don't re-read the snapshot on
# every send. Keyed by resolved project root so tests can redirect it.
_POLICY_CACHE: dict[str, dict] = {}


def active_policy(project_root: Path | None = None) -> dict:
    """The shipping policy: the promoted ``policy:delivery_gate`` definition
    from `.oac/promoted/` when one exists, else DEFAULT_POLICY.

    Uses the same framework loader the agent registry uses
    (`src.improvement.snapshot.load_promoted_definition`), merged over the
    defaults so a promoted snapshot from an older policy shape still
    carries every key. Cached per-process; falls back to DEFAULT_POLICY on
    any load error (the gate must never block delivery machinery).
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
        except Exception:  # noqa: BLE001 — degrade to baseline, never crash a send
            pass
        _POLICY_CACHE[key] = policy
    return _POLICY_CACHE[key]


def _clear_policy_cache() -> None:
    """Test/CLI hook: force the next active_policy() call to re-read disk."""
    _POLICY_CACHE.clear()
