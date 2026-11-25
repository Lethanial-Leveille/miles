"""Read only: which Evallo events survive the filter, and why.

For a date window (default Nov 23 to Dec 7 2026, both days included), writes one
line per surviving Evallo event to a mode 600 file: its evalloSessionId (or
"none"), its Google `updated` time, and the C4 row that kept it. The Mac script
compares the ids with Evallo's own list for him. A survivor with row 4 is a
failure; the others should be rows 1, 2, 3 or 5.

Run from the repo root, on the Pi:
    python3 scripts/evallo_survivors.py [FIRST_DAY LAST_DAY] [--out FILE]

It reads the calendars (events.list only) and the verified set, and writes
nothing but the output file. It prints no event titles and no secrets.

The state is the real one: the `active` flag in the database, as the server and
the voice process see it. The file's first line is
`outcome=<active, stale, off or unreadable> window=<first>..<last> fetched_at=<iso or none>`:
the filter's outcome, the export window asked for (both days included), and the
`fetched_at` of the snapshot the filter used, so the Mac compares the same
window and knows which snapshot it is checking. When the outcome is not active
the filter was not applied, so the file lists every Evallo event and must not be
read as a verification. The file is
written to a unique mode 600 temporary file in the same directory and renamed
into place, so a reader never sees half a file and an existing file is never
truncated first (a symlink at the path is replaced, never followed).
"""

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import evallo_filter  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "evallo_survivors.txt")


def _updated_text(event):
    """The Google `updated` time as a normalized UTC stamp; never the raw string."""
    raw = event.get("updated")
    if raw is None or raw == "":
        return "none"
    moment = evallo_filter.parse_moment(raw)
    return evallo_filter.iso_z(moment) if moment is not None else "unparseable"


def header_line(outcome, first, last, state):
    """The first line of the file: outcome=, then the window and the snapshot."""
    fetched = state.fetched_raw if getattr(state, "fetched_raw", None) else "none"
    return f"outcome={outcome} window={first.isoformat()}..{last.isoformat()} fetched_at={fetched}"


def survivor_lines(events, state, now):
    """One 'id updated row' line per Evallo event the filter keeps."""
    lines = []
    for event in events:
        if not evallo_filter.is_evallo(event):
            continue
        kept, row = evallo_filter.decide(event, state, now)
        if kept:
            lines.append(f"{evallo_filter.session_id(event) or 'none'} "
                         f"{_updated_text(event)} row {row}")
    return lines


def write_private(path, lines):
    """Atomically: an exclusively created, unique, mode 600 temporary file in
    the same directory, then os.replace over the destination."""
    directory = os.path.dirname(os.path.abspath(path))
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    tmp = fd = None
    for _ in range(10):
        tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}."
                                      f"{os.urandom(6).hex()}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
            break
        except FileExistsError:
            continue
    if fd is None:
        raise OSError("could not create a temporary file")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write("\n".join(lines) + ("\n" if lines else ""))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    now = evallo_filter.now_utc()
    outcome = evallo_filter.status(state, now)["outcome"]
    lines = survivor_lines(raw_events(first, last + datetime.timedelta(days=1)), state, now)
    write_private(args.out, [header_line(outcome, first, last, state)] + lines)
    print(f"outcome={outcome}: {len(lines)} Evallo events listed in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
