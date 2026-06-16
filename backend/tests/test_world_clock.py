"""In-world clock math (pure)."""

from __future__ import annotations

from app.world import clock


def test_clock_label_and_phase():
    assert clock.clock_label(8 * 60 + 35) == "08:35"
    assert clock.clock_label(0) == "00:00"
    assert clock.day_phase(8 * 60) == "morning"
    assert clock.day_phase(13 * 60) == "midday"
    assert clock.day_phase(20 * 60) == "evening"
    assert clock.day_phase(2 * 60) == "deep night"


def test_advance_rolls_day_over_midnight():
    # 23:30 + 60 min -> 00:30 next day
    clk, day = clock.advance(23 * 60 + 30, 3, 60)
    assert clk == 30
    assert day == 4


def test_advance_no_rollover_same_day():
    clk, day = clock.advance(10 * 60, 1, 45)
    assert clk == 10 * 60 + 45
    assert day == 1


def test_advance_clamps_negative_minutes():
    clk, day = clock.advance(10 * 60, 1, -30)
    assert clk == 10 * 60
    assert day == 1


def test_sleeping_hours():
    assert clock.is_sleeping_hours(2 * 60)
    assert clock.is_sleeping_hours(23 * 60 + 30)
    assert not clock.is_sleeping_hours(12 * 60)
