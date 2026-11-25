#!/usr/bin/env python3
"""Mark club events he attends, so they count as busy (see src/calendar_roles.py).

    python3 scripts/calendar_roles.py attend "Design Team Meeting" [--day YYYY-MM-DD] [--instance]
    python3 scripts/calendar_roles.py list
    python3 scripts/calendar_roles.py drop 3

attend reads the calendar (read only) to find the event: the first one whose
title contains the words, from today forward unless --day is given. It marks
the whole repeating series by default, or only that occurrence with --instance.
Only events on club calendars can be marked. list and drop touch the table
only.
"""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import calendar_roles  # noqa: E402
import calendar_tools  # noqa: E402
from database import add_attending, attending_rows, drop_attending, init_db  # noqa: E402

init_db()
args = sys.argv[1:]
command = args[0] if args else ""
if command == "attend" and len(args) >= 2:
    title = args[1]
    day = datetime.date.today()
    if "--day" in args:
        day = datetime.date.fromisoformat(args[args.index("--day") + 1])
    start = datetime.datetime.combine(day, datetime.time(0))
    window = calendar_tools.events_for_app(start, start + datetime.timedelta(days=60 if "--day" not in args else 1))
    found = calendar_roles.find_attendable(window["events"], title)
    if not found:
        print(f"No club event with {title!r} in its title found.")
        sys.exit(1)
    event = found[0]
    series_id = event["id"] if "--instance" in args else event["series_id"]
    new = add_attending(event["calendar_id"], series_id, event["title"])
    print(("Marked" if new else "Already marked"), f"{event['title']} ({event['calendar']}), "
          + ("that occurrence" if "--instance" in args else "every repeat"))
elif command == "list":
    for row_id, calendar_id, series_id, title, created in attending_rows():
        print(f"{row_id}  {title}  [{calendar_id}]  series {series_id}  since {created[:10]}")
elif command == "drop" and len(args) == 2:
    print("Dropped." if drop_attending(int(args[1])) else "No such id.")
else:
    print(__doc__)
    sys.exit(1)
