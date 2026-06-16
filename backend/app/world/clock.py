"""In-world time. Twily's life runs on its own clock, advanced per turn — not
wall-clock — so a quiet night can pass in a few background ticks and a busy
afternoon can take many. Pure functions; no I/O."""

from __future__ import annotations

MINUTES_PER_DAY = 24 * 60

_PHASES = [
    (0, "deep night"),
    (5, "early morning"),
    (8, "morning"),
    (12, "midday"),
    (14, "afternoon"),
    (18, "evening"),
    (22, "night"),
]


def hm(clock_minutes: int) -> tuple[int, int]:
    """(hour, minute) within the current day."""
    m = clock_minutes % MINUTES_PER_DAY
    return m // 60, m % 60


def clock_label(clock_minutes: int) -> str:
    h, m = hm(clock_minutes)
    return f"{h:02d}:{m:02d}"


def day_phase(clock_minutes: int) -> str:
    h, _ = hm(clock_minutes)
    phase = _PHASES[0][1]
    for start, name in _PHASES:
        if h >= start:
            phase = name
    return phase


def advance(clock_minutes: int, day_count: int, minutes: int) -> tuple[int, int]:
    """Advance the clock; roll the day counter over midnight. Returns
    (clock_minutes, day_count)."""
    minutes = max(0, int(minutes))
    total = clock_minutes + minutes
    days_passed = total // MINUTES_PER_DAY
    return total % MINUTES_PER_DAY, day_count + days_passed


def is_sleeping_hours(clock_minutes: int) -> bool:
    h, _ = hm(clock_minutes)
    return h >= 23 or h < 6
