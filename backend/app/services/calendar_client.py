"""Google Calendar API wrapper — sync methods, use asyncio.to_thread from tools."""

from __future__ import annotations

from googleapiclient.discovery import build

from app.settings import get_settings
from app.services.google_auth import get_credentials


def _service():
    return build("calendar", "v3", credentials=get_credentials())


def _format_event(event: dict) -> dict:
    """Format a Calendar event into a clean dict."""
    return {
        "id": event.get("id", ""),
        "summary": event.get("summary", ""),
        "description": event.get("description", ""),
        "location": event.get("location", ""),
        "start": event.get("start", {}),
        "end": event.get("end", {}),
        "status": event.get("status", ""),
        "htmlLink": event.get("htmlLink", ""),
        "creator": event.get("creator", {}),
        "organizer": event.get("organizer", {}),
        "attendees": event.get("attendees", []),
        "recurrence": event.get("recurrence", []),
        "created": event.get("created", ""),
        "updated": event.get("updated", ""),
    }


# ── Public API ──


def list_calendars() -> list[dict]:
    svc = _service()
    resp = svc.calendarList().list().execute()
    return [
        {
            "id": c["id"],
            "summary": c.get("summary", ""),
            "primary": c.get("primary", False),
            "accessRole": c.get("accessRole", ""),
        }
        for c in resp.get("items", [])
    ]


def list_events(
    calendar_id: str = "",
    time_min: str = "",
    time_max: str = "",
    max_results: int = 25,
    query: str = "",
) -> list[dict]:
    """List events. Empty calendar_id reads from all visible calendars."""
    svc = _service()

    if not calendar_id:
        # Read from all visible calendars
        calendars = list_calendars()
        all_events = []
        for cal in calendars:
            try:
                events = _list_single_calendar(svc, cal["id"], time_min, time_max, max_results, query)
                all_events.extend(events)
            except Exception:
                continue  # Skip calendars that error
        # Sort by start time
        all_events.sort(key=lambda e: e.get("start", {}).get("dateTime", e.get("start", {}).get("date", "")))
        return all_events[:max_results]

    return _list_single_calendar(svc, calendar_id, time_min, time_max, max_results, query)


def _list_single_calendar(
    svc, calendar_id: str, time_min: str, time_max: str, max_results: int, query: str
) -> list[dict]:
    kwargs: dict = {
        "calendarId": calendar_id,
        "maxResults": max_results,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if time_min:
        kwargs["timeMin"] = time_min
    if time_max:
        kwargs["timeMax"] = time_max
    if query:
        kwargs["q"] = query

    resp = svc.events().list(**kwargs).execute()
    return [_format_event(e) for e in resp.get("items", [])]


def get_event(event_id: str, calendar_id: str = "") -> dict:
    svc = _service()
    cal_id = calendar_id or get_settings().google_calendar_id
    event = svc.events().get(calendarId=cal_id, eventId=event_id).execute()
    return _format_event(event)


def create_event(
    summary: str,
    start: str,
    end: str,
    description: str = "",
    location: str = "",
    attendees: list[str] | None = None,
    recurrence: list[str] | None = None,
    all_day: bool = False,
) -> dict:
    """Create event on Twily's own calendar (settings.google_calendar_id)."""
    svc = _service()
    cal_id = get_settings().google_calendar_id

    body: dict = {"summary": summary}

    if all_day:
        body["start"] = {"date": start}
        body["end"] = {"date": end}
    else:
        body["start"] = {"dateTime": start}
        body["end"] = {"dateTime": end}

    if description:
        body["description"] = description
    if location:
        body["location"] = location
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    if recurrence:
        body["recurrence"] = recurrence

    event = svc.events().insert(calendarId=cal_id, body=body).execute()
    return _format_event(event)


def update_event(event_id: str, **fields) -> dict:
    """Update event on Twily's own calendar only."""
    svc = _service()
    cal_id = get_settings().google_calendar_id

    event = svc.events().get(calendarId=cal_id, eventId=event_id).execute()

    for key, value in fields.items():
        if key in ("start", "end") and isinstance(value, str):
            if len(value) <= 10:  # date only
                event[key] = {"date": value}
            else:
                event[key] = {"dateTime": value}
        else:
            event[key] = value

    updated = svc.events().update(calendarId=cal_id, eventId=event_id, body=event).execute()
    return _format_event(updated)


def delete_event(event_id: str) -> bool:
    """Delete event from Twily's own calendar only."""
    svc = _service()
    cal_id = get_settings().google_calendar_id
    svc.events().delete(calendarId=cal_id, eventId=event_id).execute()
    return True


def check_availability(time_min: str, time_max: str) -> dict:
    """Check freebusy data across all visible calendars."""
    svc = _service()
    calendars = list_calendars()
    cal_ids = [{"id": c["id"]} for c in calendars]

    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "items": cal_ids,
    }

    resp = svc.freebusy().query(body=body).execute()
    busy_periods = {}
    for cal_id, data in resp.get("calendars", {}).items():
        periods = data.get("busy", [])
        if periods:
            busy_periods[cal_id] = periods

    return {
        "time_min": time_min,
        "time_max": time_max,
        "busy": busy_periods,
        "calendars_checked": len(cal_ids),
    }
