"""Visual Report Generator — matplotlib-based data visualization tool."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from textwrap import shorten
from typing import Any

from src import ScriptTool
from pydantic import BaseModel, Field

CHARTS_DIR = Path("data/charts")

# ── Design tokens ──
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
CARD_BG = "#16213e"
ROW_ALT = "#1e2a45"  # alternating row background
GREEN = "#4ade80"
RED = "#f87171"
AMBER = "#fbbf24"
BLUE = "#60a5fa"
PURPLE = "#a78bfa"
GRAY = "#6b7280"
MUTED = "#9ca3af"
DIM = "#3b3b5c"  # barely-visible grid lines

PRIORITY_COLORS = {"high": RED, "medium": AMBER, "low": BLUE}
PRIORITY_BADGES = {"high": "H", "medium": "M", "low": "L"}

LEVEL_NAMES = {1: "Lifelong", 2: "Yearly", 3: "Quarterly", 4: "Monthly", 5: "Weekly", 6: "Daily"}

TEMPLATES = {
    "todo_board": "Kanban-style todo overview: overdue, today, upcoming, no-deadline columns",
    "goal_progress": "Horizontal progress bars for active goals grouped by level",
    "habit_tracker": "Weekly habit grid with completion status and streaks",
    "priority_matrix": "Eisenhower matrix scatter plot (importance vs immediacy)",
    "daily_summary": "Day overview card with todos, habits, events summary",
    "campaign_status": "Active nudge campaign dashboard with tactics and effectiveness",
    "events_timeline": "Life events timeline scatter (filterable by --category)",
    "custom": "Agent-provided JSON data rendered as table, bar, line, or cards",
}

# Target Telegram display: images are viewed on phone screens, so we need
# larger fonts and higher DPI to stay readable after compression.
DPI = 150
FONT_TITLE = 24
FONT_SECTION = 18
FONT_BODY = 14
FONT_SMALL = 12
FONT_BADGE = 14


class Input(BaseModel):
    command: str = Field(description="render | list-templates")
    template: str = Field(default="", description="Template name")
    send: bool = Field(default=False, description="Auto-send via Telegram")
    caption: str = Field(default="", description="Telegram caption")
    data: str = Field(default="", description="JSON string for custom template")
    category: str = Field(default="", description="Event category filter for events_timeline")
    days: int = Field(default=7, description="Lookback period for time-based templates")
    limit: int = Field(default=20, description="Max items to show")


class Output(BaseModel):
    success: bool = True
    image_path: str = ""
    sent: bool = False
    templates: list[dict[str, str]] = Field(default_factory=list)
    error: str = ""


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(BG_COLOR)
    if ax is not None:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=FONT_SMALL)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRAY)


def _save_figure(fig, plt, template_name: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"report_{template_name}_{ts}.png"
    path = CHARTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# ── Template renderers ──


async def _render_todo_board(inp: Input) -> tuple[str, Any]:
    from app.db.repos.todos import TodosRepo

    repo = TodosRepo()
    overdue = await repo.get_overdue()
    today = await repo.get_today()
    upcoming = await repo.get_upcoming(days=inp.days)
    no_deadline = await repo.get_no_deadline()

    # Deduplicate across columns
    seen_ids = {t["todo_id"] for t in overdue}
    today = [t for t in today if t["todo_id"] not in seen_ids]
    seen_ids |= {t["todo_id"] for t in today}
    upcoming = [t for t in upcoming if t["todo_id"] not in seen_ids]
    seen_ids |= {t["todo_id"] for t in upcoming}
    no_deadline = [t for t in no_deadline if t["todo_id"] not in seen_ids]

    # Cap per column to keep image manageable
    cap = min(inp.limit, 10)
    overdue = overdue[:cap]
    today = today[:cap]
    upcoming = upcoming[:cap]
    no_deadline = no_deadline[:cap]

    plt = _setup_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    columns = [
        ("Overdue", overdue, RED),
        ("Today", today, AMBER),
        ("Upcoming", upcoming, BLUE),
        ("No Deadline", no_deadline, GRAY),
    ]
    # Only show columns that have items (or always show Today)
    active_cols = [c for c in columns if c[1] or c[0] == "Today"]

    n_cols = len(active_cols)
    max_items = max((len(c[1]) for c in active_cols), default=1)
    card_h = 1.15
    card_gap = 0.2
    col_width = 4.8
    fig_width = n_cols * col_width + 1.2
    fig_height = max(5.0, 2.5 + max_items * (card_h + card_gap))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _apply_dark_theme(fig, ax)
    ax.set_xlim(0, n_cols * col_width)
    ax.set_ylim(0, max_items * (card_h + card_gap) + 2)
    ax.axis("off")

    header_y = max_items * (card_h + card_gap) + 1.2

    for col_idx, (label, items, color) in enumerate(active_cols):
        cx = col_idx * col_width
        # Column header
        ax.text(
            cx + col_width / 2,
            header_y,
            f"{label} ({len(items)})",
            ha="center",
            va="center",
            fontsize=FONT_SECTION,
            fontweight="bold",
            color=color,
        )
        ax.plot(
            [cx + 0.15, cx + col_width - 0.15],
            [header_y - 0.35, header_y - 0.35],
            color=color,
            linewidth=2.5,
        )

        for i, todo in enumerate(items):
            y = header_y - 0.7 - i * (card_h + card_gap)
            card = FancyBboxPatch(
                (cx + 0.15, y - card_h / 2),
                col_width - 0.3,
                card_h,
                boxstyle="round,pad=0.05",
                facecolor=CARD_BG,
                edgecolor=color,
                linewidth=1,
            )
            ax.add_patch(card)

            raw_title = todo.get("title", "")
            pri = todo.get("priority", "medium")
            badge_color = PRIORITY_COLORS.get(pri, GRAY)
            badge_text = PRIORITY_BADGES.get(pri, "?")

            # Wrap title into up to 2 lines (reserve space for badge)
            max_chars = 24
            if len(raw_title) <= max_chars:
                line1 = raw_title
                line2 = ""
            else:
                # Break at word boundary near max_chars
                brk = raw_title.rfind(" ", 0, max_chars)
                if brk < 10:
                    brk = max_chars
                line1 = raw_title[:brk]
                rest = raw_title[brk:].strip()
                line2 = shorten(rest, width=max_chars, placeholder="...") if rest else ""

            # Title line 1
            ty = y + 0.18 if line2 else y + 0.08
            ax.text(
                cx + 0.4,
                ty,
                line1,
                fontsize=FONT_BODY,
                color=TEXT_COLOR,
                va="center",
            )
            # Title line 2
            if line2:
                ax.text(
                    cx + 0.4,
                    ty - 0.32,
                    line2,
                    fontsize=FONT_SMALL,
                    color=MUTED,
                    va="center",
                )
            # Priority badge (right-aligned with margin)
            ax.text(
                cx + col_width - 0.5,
                y + 0.08,
                badge_text,
                fontsize=FONT_BADGE,
                fontweight="bold",
                color=badge_color,
                ha="right",
                va="center",
            )
            # Deadline
            dl = todo.get("deadline")
            if dl:
                dl_str = dl.strftime("%m/%d") if isinstance(dl, date | datetime) else str(dl)[:5]
                ax.text(cx + 0.4, y - 0.35, dl_str, fontsize=FONT_SMALL, color=MUTED, va="center")

    total = len(overdue) + len(today) + len(upcoming) + len(no_deadline)
    fig.suptitle(f"Todo Board  ({total} tasks)", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    return _save_figure(fig, plt, "todo_board"), plt


async def _render_goal_progress(inp: Input) -> tuple[str, Any]:
    from app.db.repos.goals import GoalsRepo

    repo = GoalsRepo()
    goals = await repo.list_active(limit=inp.limit)

    plt = _setup_matplotlib()

    if not goals:
        fig, ax = plt.subplots(figsize=(10, 3))
        _apply_dark_theme(fig, ax)
        ax.text(0.5, 0.5, "No active goals", ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
        ax.axis("off")
        return _save_figure(fig, plt, "goal_progress"), plt

    # Group by level
    by_level: dict[int, list] = {}
    for g in goals:
        lv = g.get("level", 6)
        by_level.setdefault(lv, []).append(g)

    # Flatten into display order
    ordered: list[tuple] = []
    for lv in sorted(by_level.keys()):
        ordered.append(("header", LEVEL_NAMES.get(lv, f"Level {lv}")))
        for g in by_level[lv]:
            meta = g.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            progress = meta.get("progress", 0)
            ordered.append(("goal", g, progress))

    n_rows = len(ordered)
    row_h = 0.8
    fig_height = max(5, n_rows * row_h + 2.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    _apply_dark_theme(fig, ax)
    ax.set_xlim(-1, 101)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    from matplotlib.patches import Rectangle

    for i, item in enumerate(ordered):
        if item[0] == "header":
            ax.text(0, i, item[1], fontsize=FONT_SECTION, fontweight="bold", color=PURPLE, va="center")
        else:
            _, goal, progress = item
            title = shorten(goal.get("title", ""), width=55, placeholder="...")
            # Alternating row bg
            if (i % 2) == 0:
                ax.add_patch(Rectangle((-1, i - 0.4), 102, 0.8, facecolor=ROW_ALT, edgecolor="none"))
            # Background bar (brighter so 0% is visible)
            ax.barh(i, 100, height=0.55, color="#2a3a5e", left=0, edgecolor=GRAY, linewidth=0.3)
            # Progress fill
            bar_color = GREEN if progress >= 50 else AMBER if progress >= 25 else RED
            if progress > 0:
                ax.barh(i, progress, height=0.55, color=bar_color, left=0, alpha=0.85)
            # Labels
            ax.text(1, i, title, fontsize=FONT_BODY, color=TEXT_COLOR, va="center")
            pct_color = bar_color if progress > 0 else MUTED
            ax.text(
                99, i, f"{progress}%", fontsize=FONT_BODY, fontweight="bold", color=pct_color, va="center", ha="right"
            )

    fig.suptitle("Goal Progress", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)
    return _save_figure(fig, plt, "goal_progress"), plt


async def _render_habit_tracker(inp: Input) -> tuple[str, Any]:
    from app.db.repos.habits import HabitsRepo

    repo = HabitsRepo()
    habits = await repo.list(status="active", limit=inp.limit)

    plt = _setup_matplotlib()

    if not habits:
        fig, ax = plt.subplots(figsize=(10, 3))
        _apply_dark_theme(fig, ax)
        ax.text(0.5, 0.5, "No active habits", ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
        ax.axis("off")
        return _save_figure(fig, plt, "habit_tracker"), plt

    today = date.today()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    day_labels = [d.strftime("%a\n%m/%d") for d in days]

    # Get occurrences for each habit
    grid: list[dict[str, Any]] = []
    for habit in habits:
        occs = await repo.get_occurrences(habit["habit_id"], limit=30)
        occ_by_date = {}
        for o in occs:
            sd = o.get("scheduled_date")
            if isinstance(sd, date):
                occ_by_date[sd] = o.get("status", "pending")
            elif sd:
                occ_by_date[date.fromisoformat(str(sd))] = o.get("status", "pending")

        statuses = [occ_by_date.get(d) for d in days]

        # Streak: consecutive completed ending at most recent
        streak = 0
        for o in sorted(occs, key=lambda x: x.get("scheduled_date", ""), reverse=True):
            if o.get("status") == "completed":
                streak += 1
            else:
                break

        grid.append(
            {
                "title": shorten(habit.get("title", ""), width=36, placeholder="..."),
                "importance": habit.get("importance_level", 3),
                "statuses": statuses,
                "streak": streak,
            }
        )

    n_habits = len(grid)
    # Use a proper table layout with matplotlib's table or manual grid
    row_h = 0.55
    name_col_w = 5.0  # data units for name column
    day_col_w = 1.0
    streak_col_w = 1.2
    total_w = name_col_w + 7 * day_col_w + streak_col_w
    fig_height = max(4.0, (n_habits + 1) * row_h + 2.0)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    _apply_dark_theme(fig, ax)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, (n_habits + 1) * row_h + 0.5)
    ax.invert_yaxis()
    ax.axis("off")

    # Header row
    header_y = 0.25
    ax.text(
        name_col_w / 2, header_y, "Habit", fontsize=FONT_BODY, fontweight="bold", color=MUTED, ha="center", va="center"
    )
    for j, label in enumerate(day_labels):
        x = name_col_w + j * day_col_w + day_col_w / 2
        ax.text(x, header_y, label, fontsize=FONT_SMALL, color=MUTED, ha="center", va="center")
    ax.text(
        name_col_w + 7 * day_col_w + streak_col_w / 2,
        header_y,
        "Streak",
        fontsize=FONT_BODY,
        fontweight="bold",
        color=MUTED,
        ha="center",
        va="center",
    )

    # Separator
    sep_y = row_h * 0.9
    ax.plot([0.1, total_w - 0.1], [sep_y, sep_y], color=GRAY, linewidth=0.8, alpha=0.5)

    status_symbols = {
        "completed": ("\u2713", GREEN),
        "skipped": ("\u2717", RED),
        "pending": ("\u25cf", AMBER),
    }
    default_symbol = ("\u2014", GRAY)

    from matplotlib.patches import Rectangle

    for i, row in enumerate(grid):
        y = (i + 1) * row_h + 0.25

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(
                Rectangle((0, y - row_h / 2 + 0.02), total_w, row_h - 0.04, facecolor=ROW_ALT, edgecolor="none")
            )

        # Habit name (left-aligned)
        ax.text(0.2, y, row["title"], fontsize=FONT_BODY, color=TEXT_COLOR, ha="left", va="center")

        # Day cells
        for j, status in enumerate(row["statuses"]):
            cx = name_col_w + j * day_col_w + day_col_w / 2
            sym, color = status_symbols.get(status, default_symbol) if status else default_symbol
            ax.text(cx, y, sym, fontsize=FONT_SECTION, color=color, ha="center", va="center", fontweight="bold")

        # Streak
        sx = name_col_w + 7 * day_col_w + streak_col_w / 2
        streak_color = GREEN if row["streak"] >= 3 else AMBER if row["streak"] >= 1 else GRAY
        ax.text(
            sx,
            y,
            str(row["streak"]),
            fontsize=FONT_SECTION,
            color=streak_color,
            ha="center",
            va="center",
            fontweight="bold",
        )

    fig.suptitle("Habit Tracker (Last 7 Days)", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)
    return _save_figure(fig, plt, "habit_tracker"), plt


async def _render_priority_matrix(inp: Input) -> tuple[str, Any]:
    from app.db.repos.priorities import PrioritiesRepo

    repo = PrioritiesRepo()
    priorities = await repo.list(status="active", limit=inp.limit)

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 9))
    _apply_dark_theme(fig, ax)

    if not priorities:
        ax.text(0.5, 0.5, "No active priorities", ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
        ax.axis("off")
        return _save_figure(fig, plt, "priority_matrix"), plt

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.axhline(y=5, color=GRAY, linestyle="--", alpha=0.5, linewidth=1.5)
    ax.axvline(x=5, color=GRAY, linestyle="--", alpha=0.5, linewidth=1.5)

    # Quadrant background tints
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((5, 5), 5.5, 5.5, facecolor=GREEN, alpha=0.06))  # Do First
    ax.add_patch(Rectangle((-0.5, 5), 5.5, 5.5, facecolor=BLUE, alpha=0.06))  # Schedule
    ax.add_patch(Rectangle((5, -0.5), 5.5, 5.5, facecolor=AMBER, alpha=0.06))  # Delegate
    ax.add_patch(Rectangle((-0.5, -0.5), 5.5, 5.5, facecolor=RED, alpha=0.06))  # Eliminate

    # Quadrant labels — large, clearly visible
    ax.text(
        2.5, 9.7, "SCHEDULE", fontsize=FONT_SECTION, color=BLUE, ha="center", va="center", alpha=0.8, fontweight="bold"
    )
    ax.text(
        7.5, 9.7, "DO FIRST", fontsize=FONT_SECTION, color=GREEN, ha="center", va="center", alpha=0.9, fontweight="bold"
    )
    ax.text(
        2.5, 0.3, "ELIMINATE", fontsize=FONT_SECTION, color=RED, ha="center", va="center", alpha=0.8, fontweight="bold"
    )
    ax.text(
        7.5, 0.3, "DELEGATE", fontsize=FONT_SECTION, color=AMBER, ha="center", va="center", alpha=0.8, fontweight="bold"
    )

    # Collect positions to offset overlapping labels
    label_positions: list[tuple[float, float]] = []

    for p in priorities:
        raw_imm = float(p.get("immediacy", 5))
        raw_imp = float(p.get("importance", 5))
        # DB stores 0-1 scale; chart uses 0-10.  Auto-detect and rescale.
        imm = raw_imm * 10 if raw_imm <= 1.0 else raw_imm
        imp = raw_imp * 10 if raw_imp <= 1.0 else raw_imp
        real_imp_raw = p.get("real_importance")
        real_imp = float(real_imp_raw) if real_imp_raw is not None else None
        if real_imp is not None and real_imp <= 1.0:
            real_imp = real_imp * 10
        gap = abs(imp - real_imp) if real_imp is not None else 0
        size = max(200, gap * 40 + 200)

        color = RED if real_imp is not None and real_imp < imp * 0.7 else GREEN
        ax.scatter(imm, imp, s=size, c=color, alpha=0.7, edgecolors="white", linewidth=1.5, zorder=3)

        title = shorten(p.get("title", ""), width=35, placeholder="...")

        # Alternate label directions to avoid overlap
        directions = [
            (14, 18),  # up-right
            (-14, 18),  # up-left
            (14, -18),  # down-right
            (-14, -18),  # down-left
            (20, 40),  # far up-right
            (-20, 40),  # far up-left
        ]
        # Pick direction that's furthest from existing labels
        best_dir = directions[len(label_positions) % len(directions)]
        offset_x, offset_y = best_dir

        # Additional offset if close to another label
        for lx, ly in label_positions:
            if abs(imm - lx) < 2.5 and abs(imp - ly) < 2.0:
                offset_y += 22 if offset_y > 0 else -22
        label_positions.append((imm, imp))

        ax.annotate(
            title,
            (imm, imp),
            fontsize=FONT_BODY,
            color=TEXT_COLOR,
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": GRAY, "alpha": 0.5, "linewidth": 0.8},
            bbox={"boxstyle": "round,pad=0.3", "facecolor": CARD_BG, "edgecolor": GRAY, "alpha": 0.9, "linewidth": 0.5},
        )

    ax.set_xlabel("Immediacy", fontsize=FONT_SECTION, color=TEXT_COLOR)
    ax.set_ylabel("Importance", fontsize=FONT_SECTION, color=TEXT_COLOR)
    fig.suptitle("Priority Matrix (Eisenhower)", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    return _save_figure(fig, plt, "priority_matrix"), plt


async def _render_daily_summary(inp: Input) -> tuple[str, Any]:
    from app.db.repos.events import EventsRepo
    from app.db.repos.habits import HabitsRepo
    from app.db.repos.todos import TodosRepo

    today = date.today()
    today_str = today.isoformat()

    todos_repo = TodosRepo()
    habits_repo = HabitsRepo()
    events_repo = EventsRepo()

    completed_todos = await todos_repo.list(status="completed", date=today_str, limit=50)
    pending_todos = await todos_repo.get_today()
    overdue_todos = await todos_repo.get_overdue()
    habits_due = await habits_repo.get_due_today()
    all_habits = await habits_repo.list(status="active")
    event_summary = await events_repo.get_daily_summary(today_str)

    # Count completed habits for today
    habits_completed = 0
    habits_total = len(all_habits)
    for habit in all_habits:
        occs = await habits_repo.get_occurrences(habit["habit_id"], limit=1)
        for o in occs:
            sd = o.get("scheduled_date")
            if sd and (sd == today or str(sd) == today_str):
                if o.get("status") == "completed":
                    habits_completed += 1
                break

    plt = _setup_matplotlib()
    from matplotlib.patches import FancyBboxPatch

    # Compact layout sized to content
    n_event_rows = min(len(event_summary), 6) if event_summary else 1
    fig_height = max(6, 4.5 + n_event_rows * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    _apply_dark_theme(fig, ax)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, fig_height)

    top = fig_height - 0.3

    # Header
    day_name = today.strftime("%A, %B %d")
    ax.text(5, top, day_name, fontsize=18, fontweight="bold", color=TEXT_COLOR, ha="center", va="top")
    top -= 0.5
    ax.plot([0.5, 9.5], [top, top], color=PURPLE, linewidth=2.5)
    top -= 0.3

    # ── Left column ──
    lx = 0.5
    ly = top

    ax.text(lx, ly, "Todos", fontsize=FONT_SECTION, fontweight="bold", color=BLUE)
    ly -= 0.45
    ax.text(lx + 0.2, ly, f"\u2713 Completed: {len(completed_todos)}", fontsize=FONT_BODY, color=GREEN)
    ly -= 0.38
    ax.text(lx + 0.2, ly, f"\u25cb Pending: {len(pending_todos)}", fontsize=FONT_BODY, color=AMBER)
    ly -= 0.38
    ax.text(lx + 0.2, ly, f"\u25cf Overdue: {len(overdue_todos)}", fontsize=FONT_BODY, color=RED)
    ly -= 0.55

    ax.text(lx, ly, "Habits", fontsize=FONT_SECTION, fontweight="bold", color=BLUE)
    ly -= 0.45
    habit_color = GREEN if habits_completed == habits_total and habits_total > 0 else AMBER
    ax.text(lx + 0.2, ly, f"Done: {habits_completed}/{habits_total}", fontsize=FONT_BODY, color=habit_color)
    ly -= 0.38
    due_color = AMBER if habits_due else GREEN
    ax.text(lx + 0.2, ly, f"Due now: {len(habits_due)}", fontsize=FONT_BODY, color=due_color)
    ly -= 0.55

    ax.text(lx, ly, "Events Today", fontsize=FONT_SECTION, fontweight="bold", color=BLUE)
    ly -= 0.45
    if event_summary:
        for ev in event_summary[:6]:
            cat = ev.get("category", "?")
            cnt = ev.get("count", 0)
            ax.text(lx + 0.2, ly, f"{cat}: {cnt}", fontsize=FONT_BODY, color=TEXT_COLOR)
            ly -= 0.35
    else:
        ax.text(lx + 0.2, ly, "No events recorded", fontsize=FONT_BODY, color=MUTED)

    # ── Right column: Scoreboard ──
    rx = 6.0
    ry = top

    # Scoreboard card
    total_tasks = len(completed_todos) + len(pending_todos)
    completion_rate = (len(completed_todos) / total_tasks * 100) if total_tasks > 0 else 0
    habit_rate = (habits_completed / habits_total * 100) if habits_total > 0 else 0
    total_events = sum(ev.get("count", 0) for ev in event_summary) if event_summary else 0
    prod_score = int(completion_rate * 0.4 + habit_rate * 0.4 + min(total_events * 5, 20))

    card = FancyBboxPatch(
        (rx - 0.2, ry - 3.2),
        4.0,
        3.2,
        boxstyle="round,pad=0.1",
        facecolor=CARD_BG,
        edgecolor=PURPLE,
        linewidth=1.5,
    )
    ax.add_patch(card)

    ax.text(rx + 1.8, ry - 0.3, "Scoreboard", fontsize=FONT_SECTION, fontweight="bold", color=PURPLE, ha="center")

    ry -= 0.85
    ax.text(
        rx + 0.15,
        ry,
        f"Tasks:  {completion_rate:.0f}%",
        fontsize=FONT_BODY,
        color=GREEN if completion_rate >= 70 else AMBER,
    )
    # Mini bar
    bar_x, bar_y, bar_w = rx + 2.2, ry + 0.02, 1.4
    ax.plot([bar_x, bar_x + bar_w], [bar_y, bar_y], color=GRAY, linewidth=6, solid_capstyle="round")
    if completion_rate > 0:
        ax.plot(
            [bar_x, bar_x + bar_w * completion_rate / 100],
            [bar_y, bar_y],
            color=GREEN if completion_rate >= 70 else AMBER,
            linewidth=6,
            solid_capstyle="round",
        )

    ry -= 0.55
    ax.text(rx + 0.15, ry, f"Habits: {habit_rate:.0f}%", fontsize=FONT_BODY, color=GREEN if habit_rate >= 70 else AMBER)
    bar_y = ry + 0.02
    ax.plot([bar_x, bar_x + bar_w], [bar_y, bar_y], color=GRAY, linewidth=6, solid_capstyle="round")
    if habit_rate > 0:
        ax.plot(
            [bar_x, bar_x + bar_w * habit_rate / 100],
            [bar_y, bar_y],
            color=GREEN if habit_rate >= 70 else AMBER,
            linewidth=6,
            solid_capstyle="round",
        )

    ry -= 0.55
    ax.text(rx + 0.15, ry, f"Events: {total_events}", fontsize=FONT_BODY, color=TEXT_COLOR)

    ry -= 0.7
    prod_color = GREEN if prod_score >= 70 else AMBER if prod_score >= 40 else RED
    ax.text(rx + 1.8, ry, f"{prod_score}/100", fontsize=22, fontweight="bold", color=prod_color, ha="center")

    fig.suptitle("Daily Summary", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.99)
    return _save_figure(fig, plt, "daily_summary"), plt


async def _render_campaign_status(inp: Input) -> tuple[str, Any]:
    from app.db.repos.nudge_campaigns import NudgeCampaignsRepo

    repo = NudgeCampaignsRepo()
    campaigns = await repo.get_active(limit=inp.limit)

    plt = _setup_matplotlib()

    if not campaigns:
        fig, ax = plt.subplots(figsize=(10, 3))
        _apply_dark_theme(fig, ax)
        ax.text(0.5, 0.5, "No active campaigns", ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
        ax.axis("off")
        return _save_figure(fig, plt, "campaign_status"), plt

    from matplotlib.patches import FancyBboxPatch

    n = len(campaigns)
    cols = min(2, n)  # 2 columns max for readability
    rows = (n + cols - 1) // cols
    card_w = 7.0
    card_h = 4.0
    gap = 0.6
    fig_width = cols * (card_w + gap) + 1.0
    fig_height = max(6, rows * (card_h + gap) + 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    _apply_dark_theme(fig, ax)
    ax.axis("off")
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.invert_yaxis()

    for idx, camp in enumerate(campaigns):
        col = idx % cols
        row = idx // cols
        x = col * (card_w + gap) + 0.3
        y = row * (card_h + gap) + 1.2

        resp = camp.get("responsiveness_score", 0.5) or 0.5
        border_color = GREEN if resp >= 0.6 else AMBER if resp >= 0.3 else RED

        card = FancyBboxPatch(
            (x, y),
            card_w,
            card_h,
            boxstyle="round,pad=0.08",
            facecolor=CARD_BG,
            edgecolor=border_color,
            linewidth=1.5,
        )
        ax.add_patch(card)

        raw_title = camp.get("target_title", "")
        tactic = camp.get("current_tactic", "?")
        esc = camp.get("escalation_level", 1)
        nudges = camp.get("total_nudges", 0)

        tx = x + 0.3
        ty = y + 0.45
        # Wrap title to 2 lines if needed
        max_w = 28
        if len(raw_title) <= max_w:
            ax.text(tx, ty, raw_title, fontsize=FONT_BODY, fontweight="bold", color=TEXT_COLOR)
        else:
            brk = raw_title.rfind(" ", 0, max_w)
            if brk < 8:
                brk = max_w
            ax.text(tx, ty, raw_title[:brk], fontsize=FONT_BODY, fontweight="bold", color=TEXT_COLOR)
            rest = raw_title[brk:].strip()
            if rest:
                ty += 0.35
                ax.text(
                    tx,
                    ty,
                    shorten(rest, width=max_w, placeholder="..."),
                    fontsize=FONT_SMALL,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
        ty += 0.55
        ax.text(tx, ty, f"Tactic: {tactic}", fontsize=FONT_BODY, color=MUTED)
        ty += 0.55
        esc_bar = "\u2588" * esc + "\u2591" * (5 - esc)
        ax.text(tx, ty, f"Escalation: {esc_bar}  ({esc}/5)", fontsize=FONT_BODY, color=border_color)
        ty += 0.55
        ax.text(tx, ty, f"Nudges: {nudges}   Score: {resp:.0%}", fontsize=FONT_BODY, color=TEXT_COLOR)

        # Effectiveness bar
        eff = (camp.get("nudges_effective", 0) / nudges) if nudges > 0 else 0
        bar_y = ty + 0.4
        bar_left = tx
        bar_right = x + card_w - 0.3
        bar_len = bar_right - bar_left
        ax.plot([bar_left, bar_right], [bar_y, bar_y], color=GRAY, linewidth=8, solid_capstyle="round")
        if eff > 0:
            ax.plot(
                [bar_left, bar_left + bar_len * eff], [bar_y, bar_y], color=GREEN, linewidth=8, solid_capstyle="round"
            )

    fig.suptitle(f"Nudge Campaigns ({n} active)", fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)
    return _save_figure(fig, plt, "campaign_status"), plt


async def _render_events_timeline(inp: Input) -> tuple[str, Any]:
    from app.db.repos.events import EventsRepo

    repo = EventsRepo()

    if inp.category:
        events = await repo.list_by_category(inp.category, days=inp.days)
    else:
        events = await repo.list_recent(limit=inp.limit)

    plt = _setup_matplotlib()

    if not events:
        fig, ax = plt.subplots(figsize=(10, 3))
        _apply_dark_theme(fig, ax)
        msg = f"No events for category '{inp.category}'" if inp.category else "No recent events"
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
        ax.axis("off")
        return _save_figure(fig, plt, "events_timeline"), plt

    from zoneinfo import ZoneInfo

    from app.settings import get_settings

    tz = ZoneInfo(get_settings().user_timezone)

    def to_local(dt):
        if dt is None:
            return None
        if isinstance(dt, date) and not isinstance(dt, datetime):
            return datetime.combine(dt, datetime.min.time(), tzinfo=tz)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(tz)

    import matplotlib.dates as mdates

    categories = sorted({e.get("category", "other") for e in events})
    palette = [GREEN, RED, AMBER, BLUE, PURPLE, "#f472b6", "#34d399", "#fb923c"]
    cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    cat_y = {cat: i for i, cat in enumerate(categories)}

    weight_events = [e for e in events if e.get("category") == "weight" and e.get("value")]

    # Give each category enough vertical space for jitter
    n_cats = len(categories)
    fig_height = max(5, n_cats * 1.4 + 3.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    _apply_dark_theme(fig, ax)

    # Draw horizontal bands per category for readability
    from matplotlib.patches import Rectangle

    for i, _cat in enumerate(categories):
        if i % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (ax.get_xlim()[0] if ax.get_xlim()[0] else 0, i - 0.45),
                    1,
                    0.9,
                    facecolor=ROW_ALT,
                    edgecolor="none",
                    transform=ax.get_yaxis_transform(),
                )
            )

    # For single-category views with many events, add vertical jitter to avoid stacking
    import random

    rng = random.Random(42)  # deterministic jitter

    for e in events:
        cat = e.get("category", "other")
        if cat == "weight":
            continue
        dt = to_local(e.get("occurred_at") or e.get("date"))
        if dt is None:
            continue
        y_base = cat_y.get(cat, 0)

        # Add jitter when single category (dots would overlap on same line)
        jitter = rng.uniform(-0.3, 0.3) if n_cats <= 2 else 0
        y = y_base + jitter

        size = 60
        val = e.get("value")
        if val:
            with contextlib.suppress(ValueError, TypeError):
                size = max(40, min(180, float(val) * 5))

        ax.scatter(dt, y, s=size, c=cat_colors.get(cat, GRAY), alpha=0.7, edgecolors="white", linewidth=0.5, zorder=3)

    # For medication: use a legend instead of per-dot annotations (which overlap)
    if inp.category == "medication" and events:
        # Build legend of unique medications
        med_types: dict[str, int] = {}
        for e in events:
            label = e.get("subcategory") or e.get("title", "unknown")
            v = e.get("value", "")
            u = e.get("unit", "")
            key = f"{label} {v}{u}".strip()
            med_types[key] = med_types.get(key, 0) + 1
        # Show as text box instead of per-dot labels
        legend_lines = [f"{name} ({cnt}x)" for name, cnt in sorted(med_types.items(), key=lambda x: -x[1])[:8]]
        legend_text = "\n".join(legend_lines)
        ax.text(
            0.98,
            0.97,
            legend_text,
            transform=ax.transAxes,
            fontsize=FONT_SMALL,
            color=TEXT_COLOR,
            va="top",
            ha="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": CARD_BG, "edgecolor": GRAY, "alpha": 0.9},
        )

    # Weight overlay on twin axis
    if weight_events and "weight" in cat_y:
        w_dates, w_vals = [], []
        for we in sorted(weight_events, key=lambda x: x.get("occurred_at") or x.get("date") or ""):
            dt = to_local(we.get("occurred_at") or we.get("date"))
            if dt:
                try:
                    w_dates.append(dt)
                    w_vals.append(float(we["value"]))
                except (ValueError, TypeError):
                    pass
        if w_dates:
            ax2 = ax.twinx()
            ax2.plot(w_dates, w_vals, "o-", color=BLUE, linewidth=2, markersize=5, alpha=0.8)
            ax2.set_ylabel("Weight (kg)", fontsize=FONT_BODY, color=BLUE)
            ax2.tick_params(colors=BLUE, labelsize=FONT_SMALL)
            ax2.spines["right"].set_color(BLUE)

    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(categories, fontsize=FONT_BODY, color=TEXT_COLOR)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", labelsize=FONT_SMALL)
    fig.autofmt_xdate()

    # Add horizontal grid lines between categories
    for i in range(n_cats - 1):
        ax.axhline(y=i + 0.5, color=DIM, linewidth=0.5, alpha=0.5)

    title = f"Events: {inp.category}" if inp.category else f"Events Timeline (last {inp.days}d)"
    fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    return _save_figure(fig, plt, "events_timeline"), plt


async def _render_custom(inp: Input) -> tuple[str, Any]:
    if not inp.data:
        raise ValueError("--data JSON string is required for custom template")

    data = json.loads(inp.data)
    title = data.get("title", "Custom Report")
    chart_type = data.get("type", "table")
    values = data.get("values", [])
    labels = data.get("labels", [])
    rows = data.get("rows", [])

    plt = _setup_matplotlib()

    if chart_type == "bar":
        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_theme(fig, ax)
        colors = [GREEN, BLUE, AMBER, RED, PURPLE] * 10
        bars = ax.bar(labels[: len(values)], values, color=colors[: len(values)], edgecolor="none")
        # Value labels on bars
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=FONT_BODY,
                color=TEXT_COLOR,
            )
        ax.tick_params(axis="x", labelsize=FONT_BODY)
        ax.tick_params(axis="y", labelsize=FONT_SMALL)
        fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    elif chart_type == "line":
        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_dark_theme(fig, ax)
        ax.plot(labels[: len(values)], values, "o-", color=GREEN, linewidth=2.5, markersize=7)
        ax.tick_params(axis="x", labelsize=FONT_BODY)
        ax.tick_params(axis="y", labelsize=FONT_SMALL)
        fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)
        fig.autofmt_xdate()

    elif chart_type == "cards":
        n = len(rows)
        fig_height = max(3, n * 0.7 + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        _apply_dark_theme(fig, ax)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n + 1)
        ax.invert_yaxis()

        from matplotlib.patches import FancyBboxPatch

        for i, row in enumerate(rows):
            card = FancyBboxPatch(
                (0.2, i + 0.1),
                9.6,
                0.55,
                boxstyle="round,pad=0.04",
                facecolor=CARD_BG,
                edgecolor=GRAY,
                linewidth=0.8,
            )
            ax.add_patch(card)
            text = "  |  ".join(f"{k}: {v}" for k, v in row.items()) if isinstance(row, dict) else str(row)
            ax.text(
                0.5,
                i + 0.38,
                shorten(text, width=100, placeholder="..."),
                fontsize=FONT_BODY,
                color=TEXT_COLOR,
                va="center",
            )

        fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    else:  # table
        n = len(rows)
        if not rows:
            fig, ax = plt.subplots(figsize=(10, 3))
            _apply_dark_theme(fig, ax)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=FONT_TITLE, color=MUTED)
            ax.axis("off")
        else:
            if isinstance(rows[0], dict):
                headers = list(rows[0].keys())
                cell_text = [[str(r.get(h, "")) for h in headers] for r in rows]
            else:
                headers = labels or [f"Col {i}" for i in range(len(rows[0]))]
                cell_text = [[str(c) for c in r] for r in rows]

            fig_height = max(3, n * 0.5 + 2)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            _apply_dark_theme(fig, ax)
            ax.axis("off")

            table = ax.table(
                cellText=cell_text,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(FONT_BODY)
            table.scale(1, 1.5)
            for key, cell in table.get_celld().items():
                cell.set_edgecolor(GRAY)
                if key[0] == 0:
                    cell.set_facecolor(PURPLE)
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    cell.set_facecolor(CARD_BG if key[0] % 2 == 1 else ROW_ALT)
                    cell.set_text_props(color=TEXT_COLOR)

            fig.suptitle(title, fontsize=FONT_TITLE, fontweight="bold", color=TEXT_COLOR, y=0.98)

    return _save_figure(fig, plt, "custom"), plt


TEMPLATE_RENDERERS = {
    "todo_board": _render_todo_board,
    "goal_progress": _render_goal_progress,
    "habit_tracker": _render_habit_tracker,
    "priority_matrix": _render_priority_matrix,
    "daily_summary": _render_daily_summary,
    "campaign_status": _render_campaign_status,
    "events_timeline": _render_events_timeline,
    "custom": _render_custom,
}


class VisualReportTool(ScriptTool[Input, Output]):
    name = "visual_report"
    description = "Generate visual data reports as images"

    def execute(self, inp: Input) -> Output:
        if inp.command == "list-templates":
            return Output(
                success=True,
                templates=[{"name": k, "description": v} for k, v in TEMPLATES.items()],
            )

        if inp.command == "render":
            if not inp.template:
                return Output(success=False, error="--template is required for render command")
            if inp.template not in TEMPLATE_RENDERERS:
                return Output(
                    success=False,
                    error=f"Unknown template: {inp.template}. Available: {', '.join(TEMPLATES)}",
                )
            return asyncio.run(self._render(inp))

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _render(self, inp: Input) -> Output:
        try:
            renderer = TEMPLATE_RENDERERS[inp.template]
            image_path, _ = await renderer(inp)
        except Exception as e:
            return Output(success=False, error=f"Render failed: {e}")

        sent = False
        if inp.send:
            try:
                from app.tools.telegram.send_image import SendImageTool  # TODO(v4-port): app.tools.telegram (send_image) not yet ported to v4. Function-local import guarded by try/except, so the render path still works; sending degrades to an error result until the telegram tool area is ported.

                result = await SendImageTool()._send(image_path, inp.caption)
                sent = result.success
                if not result.success:
                    return Output(success=True, image_path=image_path, sent=False, error=f"Send failed: {result.error}")
            except Exception as e:
                return Output(success=True, image_path=image_path, sent=False, error=f"Send failed: {e}")

        return Output(success=True, image_path=image_path, sent=sent)


if __name__ == "__main__":
    VisualReportTool.run()
