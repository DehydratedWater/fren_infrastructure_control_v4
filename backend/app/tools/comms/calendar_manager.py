"""Calendar Manager — list, create, update, delete events and check availability."""

from __future__ import annotations

import asyncio

from src import ScriptTool
from pydantic import BaseModel, Field

# NOTE(v4-port): `google.auth` / `googleapiclient` are runtime dependencies
# imported lazily inside _dispatch so the tool module can be imported without
# them installed (mirrors how gmail_manager defers its google imports).


class Input(BaseModel):
    command: str = Field(
        description="list-calendars|list-events|get-event|create-event|update-event|delete-event|check-availability|check-auth"
    )
    event_id: str = Field(default="", description="Calendar event ID")
    calendar_id: str = Field(default="", description="Calendar ID (empty = all calendars for reads)")
    summary: str = Field(default="", description="Event title/summary")
    description: str = Field(default="", description="Event description")
    location: str = Field(default="", description="Event location")
    start: str = Field(default="", description="Start time (ISO 8601 or YYYY-MM-DD for all-day)")
    end: str = Field(default="", description="End time (ISO 8601 or YYYY-MM-DD for all-day)")
    time_min: str = Field(default="", description="Filter: events starting after (ISO 8601)")
    time_max: str = Field(default="", description="Filter: events starting before (ISO 8601)")
    query: str = Field(default="", description="Text search for events")
    attendees: str = Field(default="", description="Comma-separated attendee emails")
    recurrence: str = Field(default="", description="RRULE string for recurring events")
    max_results: int = Field(default=25, description="Max results to return")
    all_day: bool = Field(default=False, description="Create as all-day event")


class Output(BaseModel):
    success: bool = True
    event: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)
    calendars: list[dict] = Field(default_factory=list)
    availability: dict = Field(default_factory=dict)
    count: int = 0
    error: str = ""


class CalendarManagerTool(ScriptTool[Input, Output]):
    name = "calendar_manager"
    description = "Manage Google Calendar events and check availability"

    def execute(self, inp: Input) -> Output:
        return asyncio.run(self._dispatch(inp))

    async def _dispatch(self, inp: Input) -> Output:
        from google.auth.exceptions import RefreshError
        from googleapiclient.errors import HttpError

        from app.services import calendar_client
        from app.services.google_auth import check_auth

        if inp.command == "check-auth":
            result = check_auth()
            return Output(success=result.get("authenticated", False), event=result)

        try:
            if inp.command == "list-calendars":
                cals = calendar_client.list_calendars()
                return Output(success=True, calendars=cals, count=len(cals))

            if inp.command == "list-events":
                events = calendar_client.list_events(
                    calendar_id=inp.calendar_id,
                    time_min=inp.time_min,
                    time_max=inp.time_max,
                    max_results=inp.max_results,
                    query=inp.query,
                )
                return Output(success=True, events=events, count=len(events))

            if inp.command == "get-event":
                if not inp.event_id:
                    return Output(success=False, error="event_id required")
                event = calendar_client.get_event(inp.event_id, calendar_id=inp.calendar_id)
                return Output(success=True, event=event)

            if inp.command == "create-event":
                if not inp.summary or not inp.start or not inp.end:
                    return Output(success=False, error="summary, start, and end required")
                attendee_list = [a.strip() for a in inp.attendees.split(",") if a.strip()] if inp.attendees else None
                recurrence_list = [inp.recurrence] if inp.recurrence else None
                event = calendar_client.create_event(
                    summary=inp.summary,
                    start=inp.start,
                    end=inp.end,
                    description=inp.description,
                    location=inp.location,
                    attendees=attendee_list,
                    recurrence=recurrence_list,
                    all_day=inp.all_day,
                )
                return Output(success=True, event=event)

            if inp.command == "update-event":
                if not inp.event_id:
                    return Output(success=False, error="event_id required")
                fields = {}
                if inp.summary:
                    fields["summary"] = inp.summary
                if inp.description:
                    fields["description"] = inp.description
                if inp.location:
                    fields["location"] = inp.location
                if inp.start:
                    fields["start"] = inp.start
                if inp.end:
                    fields["end"] = inp.end
                event = calendar_client.update_event(inp.event_id, **fields)
                return Output(success=True, event=event)

            if inp.command == "delete-event":
                if not inp.event_id:
                    return Output(success=False, error="event_id required")
                calendar_client.delete_event(inp.event_id)
                return Output(success=True)

            if inp.command == "check-availability":
                if not inp.time_min or not inp.time_max:
                    return Output(success=False, error="time_min and time_max required")
                avail = calendar_client.check_availability(inp.time_min, inp.time_max)
                return Output(success=True, availability=avail)

        except FileNotFoundError:
            return Output(
                success=False,
                error="Google Calendar not authenticated. Run: uv run scripts/google_oauth_setup.py",
            )
        except RefreshError:
            return Output(
                success=False,
                error="Google Calendar token expired or revoked. Re-run: uv run scripts/google_oauth_setup.py",
            )
        except HttpError as e:
            return Output(
                success=False,
                error=f"Google Calendar API error: {e.status_code} {e.reason}",
            )
        except Exception as e:
            return Output(success=False, error=f"Calendar operation failed: {e}")

        return Output(success=False, error=f"Unknown command: {inp.command}")


if __name__ == "__main__":
    CalendarManagerTool.run()
