"""Every calendar the Google token can see, and whether it is selected.

calendar_tools reads only selected calendars, so this answers "why is
that event missing" without guessing. Run from the repo root:
    python3 scripts/list_calendars.py
"""

import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def check_calendars():
    print("Authenticating...")
    creds = Credentials.from_authorized_user_file('data/token.json')
    service = build('calendar', 'v3', credentials=creds)
    
    print("Fetching calendar list...\n")
    calendars = service.calendarList().list(showHidden=True).execute()
    
    for cal in calendars.get('items', []):
        name = cal.get('summary', 'Unknown')
        cal_id = cal.get('id', 'Unknown')
        selected = cal.get('selected', False)
        print(f"Name: {name}")
        print(f"  ID: {cal_id}")
        print(f"  Selected in API: {selected}\n")

if __name__ == '__main__':
    check_calendars()

