"""
MYCONEX Google Calendar Ingester
----------------------------------
Fetches upcoming events from Google Calendar and makes them available to
Buzlock for context injection and morning briefing.

Auth: OAuth2 via a credentials.json downloaded from Google Cloud Console.
On first run, opens a browser for consent; token is cached at
~/.myconex/gcal_token.json for subsequent runs.

Env vars:
  GCAL_CREDENTIALS_FILE  — path to credentials.json (default: ~/.myconex/gcal_credentials.json)
  GCAL_TOKEN_FILE        — token cache path (default: ~/.myconex/gcal_token.json)
  GCAL_LOOKAHEAD_HOURS   — how far ahead to fetch events (default: 24)
  GCAL_MAX_EVENTS        — max events to return (default: 10)
  GCAL_CALENDARS         — comma-separated calendar IDs (default: "primary")

Usage:
    from integrations.calendar_ingester import get_upcoming_events, get_events_context
    events = await get_upcoming_events()
    context_str = get_events_context(events)
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BASE               = Path.home() / ".myconex"
_CREDS_FILE         = Path(os.getenv("GCAL_CREDENTIALS_FILE", str(_BASE / "gcal_credentials.json")))
_TOKEN_FILE         = Path(os.getenv("GCAL_TOKEN_FILE",       str(_BASE / "gcal_token.json")))
GCAL_LOOKAHEAD      = int(os.getenv("GCAL_LOOKAHEAD_HOURS", "24"))
GCAL_MAX_EVENTS     = int(os.getenv("GCAL_MAX_EVENTS",       "10"))
GCAL_CALENDARS      = [c.strip() for c in os.getenv("GCAL_CALENDARS", "primary").split(",") if c.strip()]

_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _get_credentials():
    """Return valid Google OAuth2 credentials, refreshing if needed."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if _TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_FILE), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDS_FILE.exists():
                raise FileNotFoundError(
                    f"Google Calendar credentials not found at {_CREDS_FILE}. "
                    "Download credentials.json from Google Cloud Console and place it there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_FILE), _SCOPES)
            creds = flow.run_local_server(port=0)
        _BASE.mkdir(parents=True, exist_ok=True)
        _TOKEN_FILE.write_text(creds.to_json())

    return creds


def _fetch_events_sync(lookahead_hours: int, max_events: int) -> list[dict[str, Any]]:
    """Synchronous fetch — run in asyncio.to_thread."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(hours=lookahead_hours)).isoformat()

    events: list[dict] = []
    for cal_id in GCAL_CALENDARS:
        try:
            result = service.events().list(
                calendarId=cal_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_events,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            for item in result.get("items", []):
                start = item.get("start", {})
                end   = item.get("end",   {})
                events.append({
                    "id":          item.get("id", ""),
                    "summary":     item.get("summary", "(no title)"),
                    "description": (item.get("description") or "")[:300],
                    "location":    item.get("location", ""),
                    "start":       start.get("dateTime") or start.get("date", ""),
                    "end":         end.get("dateTime")   or end.get("date",   ""),
                    "calendar":    cal_id,
                    "attendees":   [a.get("email", "") for a in item.get("attendees", [])[:5]],
                    "meet_link":   item.get("hangoutLink", ""),
                })
        except Exception as exc:
            logger.warning("[calendar] failed to fetch from %s: %s", cal_id, exc)

    events.sort(key=lambda e: e["start"])
    return events[:max_events]


async def get_upcoming_events(
    lookahead_hours: int = GCAL_LOOKAHEAD,
    max_events: int = GCAL_MAX_EVENTS,
) -> list[dict[str, Any]]:
    """Async wrapper — fetches upcoming calendar events."""
    try:
        return await asyncio.to_thread(_fetch_events_sync, lookahead_hours, max_events)
    except FileNotFoundError as exc:
        logger.info("[calendar] %s", exc)
        return []
    except Exception as exc:
        logger.warning("[calendar] get_upcoming_events failed: %s", exc)
        return []


def format_event_time(event: dict) -> str:
    """Return a human-readable time string for an event."""
    start = event.get("start", "")
    if not start:
        return ""
    try:
        if "T" in start:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            local = dt.astimezone()
            return local.strftime("%-I:%M %p")
        else:
            return start  # all-day event — just the date
    except Exception:
        return start


def get_events_context(events: list[dict]) -> str:
    """
    Format events as a context block for injection into the system prompt.
    Returns empty string if no events.
    """
    if not events:
        return ""
    lines = ["[Upcoming calendar events (today/tomorrow):]"]
    for e in events:
        time_str = format_event_time(e)
        loc = f" @ {e['location'][:40]}" if e.get("location") else ""
        meet = " 🎥 Meet" if e.get("meet_link") else ""
        lines.append(f"• {time_str}  **{e['summary']}**{loc}{meet}")
        if e.get("description"):
            lines.append(f"  _{e['description'][:120]}_")
    return "\n".join(lines)
