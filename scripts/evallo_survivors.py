"""Read only: which Evallo events survive the filter, and why.

For a date window (default Nov 23 to Dec 7 2026, both days included), writes one
line per surviving Evallo event to a mode 600 file: its evalloSessionId (or
"none"), its Google `updated` time, and the C4 row that kept it. The Mac script
compares the ids with Evallo's own list for him. A survivor with row 4 is a
failure; the others should be rows 1, 2, 3 or 5.

Run from the repo root, on the Pi:
    python3 scripts/evallo_survivors.py [FIRST_DAY LAST_DAY] [--out FILE]

It reads the calendars (events.list only) and the verified set, and writes
nothing but the output file. It prints no event titles and no secrets. The state
is read as if this process had accepted a push, because it is a separate process
from the server and has no flag of its own; if no status row exists it says so
and writes nothing.
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import evallo_filter  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "evallo_survivors.txt")


def survivor_lines(events, state, now):
    """One 'id updated row' line per Evallo event the filter keeps."""
    lines = []
    for event in events:
        if not evallo_filter.is_evallo(event):
            continue
        kept, row = evallo_filter.decide(event, state, now)
        if kept:
            lines.append(f"{evallo_filter.session_id(event) or 'none'} "
                         f"{event.get('updated') or 'none'} row {row}")
    return lines


def write_private(path, lines):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write("\n".join(lines) + ("\n" if lines else ""))
    os.chmod(path, 0o600)


def raw_events(first, last_exclusive):
    """Every selected calendar's raw events in the window, unfiltered."""
    import calendar_tools
    service = calendar_tools._service()
    start = datetime.datetime.combine(first, datetime.time(0), evallo_filter.NY)
    end = datetime.datetime.combine(last_exclusive, datetime.time(0), evallo_filter.NY)
    events = []
    for cid in calendar_tools._selected(service):
        token = None
        while True:
            page = service.events().list(
                calendarId=cid, timeMin=start.isoformat(), timeMax=end.isoformat(),
                singleEvents=True, maxResults=250,
                **({"pageToken": token} if token else {})).execute()
            events += page.get("items", [])
            token = page.get("nextPageToken")
            if not token:
                break
    return events


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("first", nargs="?", default="2026-11-23")
    parser.add_argument("last", nargs="?", default="2026-12-07")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    first = datetime.date.fromisoformat(args.first)
    last = datetime.date.fromisoformat(args.last)
    state = evallo_filter.load_state()
    if not state.readable or state.fetched_at is None:
        print("No readable Evallo status row; nothing written.")
        return 1
    state.accepted = True
    lines = survivor_lines(raw_events(first, last + datetime.timedelta(days=1)),
                           state, evallo_filter.now_utc())
    write_private(args.out, lines)
    print(f"{len(lines)} surviving Evallo events written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
