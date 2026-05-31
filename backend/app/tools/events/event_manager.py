"""Event Manager — CRUD for life events with categories and plotting."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import date, datetime
from pathlib import Path

from src import ScriptTool
from pydantic import BaseModel, Field

VALID_CATEGORIES = {
    "medication": {"unit": "mg", "examples": "mph, concerta, atenza"},
    "walk": {"unit": "min", "examples": ""},
    "sick": {"unit": "", "examples": "nausea, fever"},
    "pain": {"unit": "/10", "examples": "head, back, stomach"},
    "weight": {"unit": "kg", "examples": ""},
    "purchase": {"unit": "pln", "examples": "item name"},
    "workout": {"unit": "min", "examples": "cardio, gym"},
    "shower": {"unit": "", "examples": ""},
    "travel": {"unit": "", "examples": "warsaw_to_wroclaw, wroclaw_to_warsaw"},
    "exercise": {"unit": "reps", "examples": "bench press, squat"},
    "eating": {"unit": "", "examples": "pizza, salad"},
    "drinking": {"unit": "ml", "examples": "water, coffee, tea"},
    "late_activity": {"unit": "hour", "examples": ""},
}


class Input(BaseModel):
    command: str = Field(
        default="create|get|update|list|list-recent|list-by-category|delete|get-state|update-state|daily-summary|plot",
    )
    today_only: bool = Field(default=False, description="Filter to today's events only")
    event_id: str = Field(default="", description="Event ID")
    category: str = Field(default="", description="Event category")
    subcategory: str = Field(default="", description="Event subcategory")
    title: str = Field(default="", description="Event title/description")
    value: str = Field(default="", description="Numeric or text value")
    unit: str = Field(default="", description="Unit of measurement")
    source: str = Field(default="manual", description="Source: manual or extracted")
    source_message_id: int = Field(default=0, description="Source chat message ID")
    date_from: str = Field(default="", description="Start date filter (YYYY-MM-DD)")
    date_to: str = Field(default="", description="End date filter (YYYY-MM-DD)")
    date: str = Field(default="", description="Date (YYYY-MM-DD)")
    days: int = Field(default=30, description="Number of days for lookback")
    limit: int = Field(default=50, description="Result limit")
    last_processed_message_id: int = Field(default=0, description="Last processed message ID for extraction state")
    occurred_at: str = Field(default="", description="When the event occurred (ISO datetime)")
    quantity: float | None = Field(default=None, description="Numeric quantity (e.g. 36 for 36mg)")
    cost: float | None = Field(default=None, description="Monetary cost")
    currency: str = Field(default="", description="Currency code (pln, eur, usd)")
    duration_minutes: int | None = Field(default=None, description="Duration in minutes")
    metadata_json: str = Field(default="", description="JSON string for extra metadata")


class Output(BaseModel):
    success: bool = True
    already_exists: bool = False
    event: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)
    summary: list[dict] = Field(default_factory=list)
    state: dict = Field(default_factory=dict)
    count: int = 0
    chart_path: str = ""
    error: str = ""


class EventManagerTool(ScriptTool[Input, Output]):
    name = "event_manager"
    description = "Track life events with categories, values, and plotting"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from app.db.repos.events import EventsRepo

        repo = EventsRepo()

        if inp.command == "create":
            if not inp.category:
                return Output(success=False, error="Category is required")
            if inp.category not in VALID_CATEGORIES:
                return Output(
                    success=False,
                    error=f"Invalid category: {inp.category}. Valid: {', '.join(sorted(VALID_CATEGORIES))}",
                )
            if not inp.title:
                return Output(success=False, error="Title is required")

            # Normalize subcategory to lowercase to prevent MPH/mph/Mph duplicates
            normalized_sub = inp.subcategory.lower().strip() if inp.subcategory else None

            # Dedup: skip if this message already has an event in this category
            if inp.source_message_id and await repo.exists_for_message(inp.source_message_id, inp.category):
                return Output(success=True, already_exists=True)

            now = datetime.now()
            if inp.occurred_at:
                occurred = datetime.fromisoformat(inp.occurred_at)
            else:
                occurred = now

            # Dedup: skip if a similar event exists within 60 min (catches manual+extracted duplicates)
            if await repo.exists_similar(
                inp.category,
                occurred,
                subcategory=normalized_sub,
                value=inp.value or None,
                window_minutes=60,
            ):
                return Output(success=True, already_exists=True)

            event_date = date.fromisoformat(inp.date) if inp.date else occurred.date()
            hash_input = f"{inp.category}{inp.title}{occurred.isoformat()}{inp.source_message_id}"
            short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            eid = f"evt_{event_date.strftime('%Y%m%d')}_{short_hash}"

            # Parse extra metadata JSON
            extra_metadata: dict = {}
            if inp.metadata_json:
                try:
                    import json

                    extra_metadata = json.loads(inp.metadata_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            event = await repo.create(
                event_id=eid,
                category=inp.category,
                title=inp.title,
                occurred_at=occurred,
                date=event_date,
                subcategory=normalized_sub,
                value=inp.value or None,
                unit=inp.unit or None,
                source=inp.source,
                source_message_id=inp.source_message_id or None,
                metadata=extra_metadata or None,
                quantity=inp.quantity,
                cost=inp.cost,
                currency=inp.currency or None,
                duration_minutes=inp.duration_minutes,
            )

            # Cache the event artifact
            try:
                from app.db.repos.context_cache import add_to_cache

                parts = [inp.title]
                if inp.value and inp.unit:
                    parts.append(f"{inp.value}{inp.unit}")
                elif inp.value:
                    parts.append(inp.value)
                if inp.subcategory:
                    parts.append(f"({inp.subcategory})")
                await add_to_cache(
                    "event",
                    f"{inp.category}: {' '.join(parts)}",
                    entity_type="events",
                    entity_id=eid,
                    tags=["event", inp.category, f"date:{event_date.isoformat()}"],
                    source_agent="event_manager",
                )
            except Exception:
                pass

            # Auto-create todo for purchase events
            if inp.category == "purchase":
                try:
                    from app.db.repos.todos import TodosRepo

                    todo_id = f"todo_{event_date.strftime('%Y%m%d')}_{short_hash}"
                    cost_str = f" ({inp.cost} {inp.currency})" if inp.cost else ""
                    await TodosRepo().create(
                        todo_id=todo_id,
                        title=f"Complete purchase: {inp.title}{cost_str}",
                        date_str=event_date.isoformat(),
                        description=f"Auto-created from purchase event {eid}",
                        source="event_manager",
                    )
                except Exception:
                    pass  # Don't fail the event creation if todo fails

            return Output(success=True, event=event)

        if inp.command == "get":
            event = await repo.get(inp.event_id)
            if not event:
                return Output(success=False, error=f"Event not found: {inp.event_id}")
            return Output(success=True, event=event)

        if inp.command == "list":
            events = await repo.list(
                category=inp.category or None,
                date_from=inp.date_from or None,
                date_to=inp.date_to or None,
                source=inp.source if inp.source != "manual" else None,
                limit=inp.limit,
            )
            return Output(success=True, events=events, count=len(events))

        if inp.command == "list-recent":
            events = await repo.list_recent(limit=inp.limit or 20)
            # Filter to today only if requested
            if inp.today_only:
                today = date.today().isoformat()
                events = [e for e in events if str(e.get("date", ""))[:10] == today]
            # Add date_relative field to help distinguish today vs yesterday
            today = date.today().isoformat()
            from datetime import timedelta

            yesterday = (date.today() - timedelta(days=1)).isoformat()
            for event in events:
                event_date = str(event.get("date", ""))[:10]
                if event_date == today:
                    event["date_relative"] = "today"
                elif event_date == yesterday:
                    event["date_relative"] = "yesterday"
                else:
                    event["date_relative"] = f"on {event_date}"
            return Output(success=True, events=events, count=len(events))

        if inp.command == "list-by-category":
            if not inp.category:
                return Output(success=False, error="Category is required")
            events = await repo.list_by_category(inp.category, days=inp.days)
            return Output(success=True, events=events, count=len(events))

        if inp.command == "update":
            if not inp.event_id:
                return Output(success=False, error="event_id is required for update")
            occurred = datetime.fromisoformat(inp.occurred_at) if inp.occurred_at else None
            updated = await repo.update(
                inp.event_id,
                category=inp.category or None,
                subcategory=inp.subcategory.lower().strip() if inp.subcategory else None,
                title=inp.title or None,
                value=inp.value or None,
                unit=inp.unit or None,
                occurred_at=occurred,
                quantity=inp.quantity,
                cost=inp.cost,
                currency=inp.currency or None,
                duration_minutes=inp.duration_minutes,
            )
            if not updated:
                return Output(success=False, error=f"Event not found: {inp.event_id}")
            return Output(success=True, event=updated)

        if inp.command == "delete":
            ok = await repo.delete(inp.event_id)
            if ok:
                # Invalidate context cache entry for this event
                try:
                    from app.db.repos.context_cache import ContextCacheRepo

                    cache_repo = ContextCacheRepo()
                    await cache_repo.delete(inp.event_id)
                except Exception:
                    pass
            return Output(success=ok, error="" if ok else f"Event not found: {inp.event_id}")

        if inp.command == "get-state":
            state = await repo.get_extraction_state()
            return Output(success=True, state=state)

        if inp.command == "update-state":
            state = await repo.update_extraction_state(inp.last_processed_message_id)
            return Output(success=True, state=state)

        if inp.command == "daily-summary":
            target_date = inp.date or date.today().isoformat()
            summary = await repo.get_daily_summary(target_date)
            return Output(success=True, summary=summary, count=len(summary))

        if inp.command == "plot":
            return await self._plot(inp, repo)

        return Output(success=False, error=f"Unknown command: {inp.command}")

    async def _plot(self, inp: Input, repo: object) -> Output:
        if not inp.category:
            return Output(success=False, error="Category is required for plotting")

        from app.db.repos.events import EventsRepo

        assert isinstance(repo, EventsRepo)
        events = await repo.list_by_category(inp.category, days=inp.days)
        if not events:
            return Output(success=False, error=f"No events found for category: {inp.category}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except ImportError:
            return Output(success=False, error="matplotlib not installed")

        from zoneinfo import ZoneInfo

        from app.settings import get_settings

        tz = ZoneInfo(get_settings().user_timezone)
        tz_label = get_settings().user_timezone

        # Convert occurred_at to local timezone for display
        def to_local(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                from zoneinfo import ZoneInfo as ZI

                dt = dt.replace(tzinfo=ZI("UTC"))
            return dt.astimezone(tz)

        chart_dir = Path("data/charts")
        chart_dir.mkdir(parents=True, exist_ok=True)
        filename = f"events_{inp.category}_{inp.days}d.png"
        chart_path = chart_dir / filename

        fig, ax = plt.subplots(figsize=(12, 5))

        if inp.category == "weight":
            dates = []
            values = []
            for e in events:
                if e.get("value"):
                    try:
                        values.append(float(e["value"]))
                        dates.append(to_local(e["occurred_at"]))
                    except (ValueError, TypeError):
                        pass
            if dates:
                ax.plot(dates, values, "o-", color="#2196F3", linewidth=2, markersize=6)
                ax.set_ylabel("kg")
                ax.set_title(f"Weight — last {inp.days} days ({tz_label})")
            else:
                ax.text(0.5, 0.5, "No weight data with values", ha="center", va="center", transform=ax.transAxes)

        elif inp.category == "medication":
            dates = []
            labels = []
            for e in events:
                dates.append(to_local(e["occurred_at"]))
                label = e.get("subcategory") or e.get("title", "")
                val = e.get("value", "")
                unit = e.get("unit", "")
                labels.append(f"{label} {val}{unit}".strip())
            if dates:
                ax.scatter(dates, range(len(dates)), c="#4CAF50", s=80, zorder=3)
                for i, lbl in enumerate(labels):
                    ax.annotate(lbl, (dates[i], i), fontsize=8, xytext=(5, 0), textcoords="offset points")
                ax.set_yticks([])
                ax.set_title(f"Medication intake — last {inp.days} days ({tz_label})")
            else:
                ax.text(0.5, 0.5, "No medication data", ha="center", va="center", transform=ax.transAxes)

        elif inp.category == "late_activity":
            hours: dict[int, int] = {h: 0 for h in range(6)}
            for e in events:
                if e.get("value"):
                    try:
                        h = int(float(e["value"]))
                        if 0 <= h <= 5:
                            hours[h] += 1
                    except (ValueError, TypeError):
                        pass
            ax.bar(list(hours.keys()), list(hours.values()), color="#FF9800")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Count")
            ax.set_title(f"Late night activity — last {inp.days} days ({tz_label})")
            ax.set_xticks(range(6))

        else:
            # Generic: event frequency per day
            day_counts: dict[str, int] = {}
            for e in events:
                d = str(e.get("date", ""))[:10]
                day_counts[d] = day_counts.get(d, 0) + 1
            if day_counts:
                sorted_days = sorted(day_counts.keys())
                counts = [day_counts[d] for d in sorted_days]
                ax.bar(sorted_days, counts, color="#9C27B0")
                ax.set_ylabel("Count")
                ax.set_title(f"{inp.category} events — last {inp.days} days ({tz_label})")
                if len(sorted_days) > 10:
                    ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return Output(success=True, chart_path=str(chart_path))


if __name__ == "__main__":
    EventManagerTool.run()
