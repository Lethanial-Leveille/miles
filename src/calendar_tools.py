import os
import datetime
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from tools import Permission, tool

TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'token.json')

def _get_calendar_service():
    """Private helper to authenticate and build the Calendar API client."""
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"{TOKEN_FILE} not found. Please run scripts/google_auth.py first.")
    
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, ['https://www.googleapis.com/auth/calendar'])
    service = build('calendar', 'v3', credentials=creds)
    return service

@tool(
    name="get_upcoming_events",
    description="Fetches the next 10 upcoming events from Lethanial's primary calendar.",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_upcoming_events():
    service = _get_calendar_service()
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    
    events_result = service.events().list(
        calendarId='primary', timeMin=now,
        maxResults=10, singleEvents=True,
        orderBy='startTime').execute()
        
    events = events_result.get('items', [])
    
    if not events:
        return "No upcoming events found."
        
    result = []
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'Untitled')
        result.append(f"{start}: {summary}")
        
    return "\n".join(result)

@tool(
    name="check_calendar_freebusy",
    description=(
        "Checks Lethanial's primary calendar for busy blocks across a given time range. "
        "Useful for determining available gaps before scheduling an event."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "time_min": {
                "type": "string",
                "description": "Start of the time range in ISO 8601 format (e.g., 2026-09-13T10:00:00Z)"
            },
            "time_max": {
                "type": "string",
                "description": "End of the time range in ISO 8601 format (e.g., 2026-09-13T18:00:00Z)"
            }
        },
        "required": ["time_min", "time_max"]
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def check_calendar_freebusy(time_min, time_max):
    service = _get_calendar_service()
    body = {
        "timeMin": time_min,
        "timeMax": time_max,
        "timeZone": "UTC",
        "items": [{"id": "primary"}]
    }
    
    eventsResult = service.freebusy().query(body=body).execute()
    calendars = eventsResult.get("calendars", {})
    primary = calendars.get("primary", {})
    busy = primary.get("busy", [])
    
    return {"busy_blocks": busy}

