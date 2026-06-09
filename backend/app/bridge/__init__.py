"""Event → habit bridge — tunable matching policy + cron runner.

The decision logic (which detected life event auto-completes which habit) is a
pure function over a JSON-able policy dict, optimisable offline via the
framework's `src.improvement.autoresearch` loop (see `event_habit_probes`).
"""
