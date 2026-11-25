"""Which Evallo events on his calendar are his, decided on the Pi.

Evallo's two way sync writes the whole organization's tutoring sessions onto
his primary calendar. The Mac reads his own sessions from Evallo's API and
pushes their ids (PUT /agent/evallo/sessions); every calendar read drops an
Evallo event that is not his. The one rule is keep() below, written out as a
table in docs/EVALLO_FILTER_CONTRACT.md (C4). Every uncertainty errs toward
keeping the event: busy, never free.

The filter is on only when there is a status row, the state was read, and
`active` is 1 in the database (C2). miles-server clears it at startup, so a restart
turns it off until the Mac's next push is accepted, and a redeploy or a rollback
can never apply an old set.
"""

import contextlib
import contextvars
import datetime
import re
import sqlite3
import unicodedata
from zoneinfo import ZoneInfo

import database

NY = ZoneInfo("America/New_York")
UTC = datetime.timezone.utc

EVALLO_KEYS = ("evalloEvent", "evalloSessionId", "orgId")
SESSION_ID = re.compile(r"^[0-9a-f]{24}\Z")
# Evallo stamps the push time, Google stamps `updated`; the two clocks may
# disagree, so an event updated this close to the snapshot, or later, may have
# been booked after it and is kept (row 3).
CUTOFF_MARGIN = datetime.timedelta(minutes=15)
FRESH_FOR = datetime.timedelta(hours=24)
TUTOR_NAME = "Lethanial Leveille"
# <student> <> <tutor> | ...   and   <student> / <tutor>
_NAME_RULE = re.compile(
    r"^[^<>/]+?\s*(?:<>|/)\s*" + re.escape(TUTOR_NAME.casefold()) + r"\s*(?:\||$)")

OUTCOMES = ("active", "stale", "off", "unreadable")
_RANK = {"active": 0, "stale": 1, "off": 2, "unreadable": 3}

def clear_active():
    """miles-server's startup: the filter stays off until a push is accepted
    after this start. Only the server calls this; voice and the review never
    write the flag. A failure is logged and never stops the server."""
    try:
        conn = sqlite3.connect(database.DB_PATH, timeout=10)
        try:
            conn.execute("UPDATE evallo_verification SET active = 0")
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        print(f"Evallo filter: could not clear active at startup ({type(exc).__name__}).",
              flush=True)


def now_utc():
    return datetime.datetime.now(UTC)


class Conflict(Exception):
    """A push refused by the ordering rule. reason is older_snapshot or
    conflicting_snapshot."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def iso_z(moment):
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_moment(text):
    """An aware datetime from an ISO string, or None. A naive string is not a
    moment here."""
    if not isinstance(text, str) or not text:
        return None
    try:
        moment = datetime.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


# ── state ──

class State:
    """What one read of the database found: the ids and the status row together
    (one transaction), whether it could be read, and whether a push has
    been accepted since miles-server started (the `active` column)."""

    def __init__(self, readable=True, ids=frozenset(), fetched_at=None, window=None,
                 fetched_raw=None, accepted=False):
        self.readable = readable
        self.ids = ids
        self.fetched_at = fetched_at          # aware datetime or None
        self.fetched_raw = fetched_raw        # as stored
        self.window = window                  # (start date, end date) or None
        self.accepted = accepted

    @property
    def on(self):
        return self.readable and self.fetched_at is not None and self.accepted

    def outcome(self, now):
        """unreadable, then off, then the freshness check."""
        if not self.readable:
            return "unreadable"
        if self.fetched_at is None or not self.accepted:
            return "off"
        if now - self.fetched_at > FRESH_FOR:
            return "stale"
        return "active"


def load_state():
    """The verified ids and the status row from one transaction. A failure is
    logged without detail beyond its type, and gives an unreadable state: every
    event is kept."""
    conn = None
    try:
        conn = sqlite3.connect(database.DB_PATH, timeout=5, isolation_level=None)
        conn.execute("BEGIN")
        row = conn.execute("SELECT fetched_at, window_start, window_end, count, active "
                           "FROM evallo_verification WHERE id = 1").fetchone()
        ids = frozenset(r[0] for r in conn.execute("SELECT session_id FROM evallo_verified_sessions"))
        conn.execute("COMMIT")
        if row is None:
            return State()
        if row[3] != len(ids):
            raise ValueError("stored count does not match the stored ids")
        fetched = parse_moment(row[0])
        window = (datetime.date.fromisoformat(row[1]), datetime.date.fromisoformat(row[2]))
        if fetched is None:
            raise ValueError("stored fetched_at does not parse")
        return State(ids=ids, fetched_at=fetched, window=window, fetched_raw=row[0],
                     accepted=row[4] == 1)
    except Exception as exc:
        print(f"Evallo filter state unreadable ({type(exc).__name__}); keeping every event.",
              flush=True)
        return State(readable=False)
    finally:
        if conn is not None:
            conn.close()


def status(state, now=None):
    now = now or now_utc()
    return {"outcome": state.outcome(now), "verified_at": state.fetched_raw}


# ── the rule ──

def private_props(event):
    extended = event.get("extendedProperties") if isinstance(event, dict) else None
    props = extended.get("private") if isinstance(extended, dict) else None
    return props if isinstance(props, dict) else {}


def is_evallo(event):
    props = private_props(event)
    return any(key in props for key in EVALLO_KEYS)


def session_id(event):
    value = private_props(event).get("evalloSessionId")
    return value if isinstance(value, str) and SESSION_ID.match(value) else None


def _zone(name):
    """The ZoneInfo for a Google timeZone name, or None when absent or unknown."""
    if not isinstance(name, str) or not name:
        return None
    try:
        return ZoneInfo(name)
    except Exception:
        return None


def place_in_time(event):
    """A copy of the event whose start and end carry real offsets. Google can
    send a dateTime with no offset and a separate timeZone; read as local time
    it lands hours off. An offset in the dateTime always wins; with no offset
    and no usable zone the field is left as it came (the old behavior), and
    start_date calls that start unusable."""
    placed = dict(event)
    changed = False
    zones = {}
    for key in ("start", "end"):
        part = event.get(key)
        if not isinstance(part, dict) or not part.get("dateTime"):
            continue
        try:
            moment = datetime.datetime.fromisoformat(str(part["dateTime"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if moment.tzinfo is not None:
            continue
        zone = _zone(part.get("timeZone")) or zones.get("start")
        if zone is None:
            continue
        zones[key] = zone
        placed[key] = {**part, "dateTime": moment.replace(tzinfo=zone).isoformat()}
        changed = True
    return placed if changed else event


def start_date(event):
    """The New York date of the event's start, or None when unusable."""
    start = event.get("start") if isinstance(event, dict) else None
    if not isinstance(start, dict):
        return None
    try:
        if start.get("dateTime"):
            moment = datetime.datetime.fromisoformat(str(start["dateTime"]).replace("Z", "+00:00"))
            if moment.tzinfo is None:
                # No offset in the string: the event's own zone. An unknown or
                # missing zone makes the start unusable; New York is never guessed.
                zone = _zone(start.get("timeZone"))
                if zone is None:
                    return None
                moment = moment.replace(tzinfo=zone)
            return moment.astimezone(NY).date()
        if start.get("date"):
            return datetime.date.fromisoformat(str(start["date"]))
    except (ValueError, TypeError):
        return None
    return None


def _squash(text):
    """Any run of whitespace, a non breaking space included, as one space, the
    way evallo_matching reads a title."""
    text = unicodedata.normalize("NFKD", text.replace("\u00a0", " "))
    return re.sub(r"\s+", " ", text).strip().casefold()


def has_tutor_name(event):
    title = event.get("summary") if isinstance(event, dict) else None
    return isinstance(title, str) and bool(_NAME_RULE.match(_squash(title)))


def decide(event, state, now):
    """(keep, row). row is the C4 row that decided, 0 when the event is not
    Evallo's or the filter is off. Never raises: an event that cannot be
    evaluated is kept (busy, never free), and the failure is logged without any
    of the event's contents."""
    try:
        return _decide(event, state, now)
    except Exception as exc:
        print(f"Evallo filter: could not evaluate one event ({type(exc).__name__}); "
              "keeping it.", flush=True)
        return True, 0


def _decide(event, state, now):
    if not state.on or not is_evallo(event):
        return True, 0
    sid = session_id(event)
    if sid is None:
        return True, 1
    if sid in state.ids:
        return True, 2
    updated = parse_moment(event.get("updated"))
    if updated is None or updated > state.fetched_at - CUTOFF_MARGIN:
        return True, 3
    fresh = now - state.fetched_at <= FRESH_FOR
    day = start_date(event)
    if fresh and day is not None and state.window[0] <= day < state.window[1]:
        return False, 4
    if has_tutor_name(event):
        return True, 5
    return False, 6


def keep(event, state, now):
    return decide(event, state, now)[0]


ANONYMOUS_TITLE = "Evallo session (unverified)"
# Everything an anonymous block keeps, besides the reduced attendee list and the
# Evallo private properties: when it is, whether it blocks time, which series.
_ANONYMOUS_KEEPS = ("id", "start", "end", "transparency", "status", "recurringEventId",
                    "originalStartTime")


def anonymize(event):
    """An uncertain event as a bare busy block: no title, description, location,
    attendees, organizer, conference data or names. The Evallo private
    properties stay (the write guard needs them), and the attendee list holds
    only the owner's own response, so a declined event is still not busy."""
    block = {key: event[key] for key in _ANONYMOUS_KEEPS if key in event}
    block["summary"] = ANONYMOUS_TITLE
    private = private_props(event)
    block["extendedProperties"] = {"private": {k: private[k] for k in EVALLO_KEYS if k in private}}
    attendees = event.get("attendees")
    if isinstance(attendees, list):
        for who in attendees:
            if isinstance(who, dict) and who.get("self"):
                block["attendees"] = [{"self": True, "responseStatus": who.get("responseStatus")}]
                break
    return block


def view(event, state, now):
    """(event as MILES may see it, or None when dropped; C4 row). A kept event
    that is only uncertain (row 1 or 3) and does not carry his full name comes
    back anonymized. Never raises."""
    kept, row = decide(event, state, now)
    if not kept:
        return None, row
    try:
        event = place_in_time(event)
    except Exception:
        pass
    if row in (1, 3):
        try:
            if not has_tutor_name(event):
                return anonymize(event), row
        except Exception as exc:
            print(f"Evallo filter: could not anonymize one event ({type(exc).__name__}); "
                  "returning a bare block.", flush=True)
            return {"id": event.get("id"), "start": event.get("start"), "end": event.get("end"),
                    "summary": ANONYMOUS_TITLE,
                    "extendedProperties": {"private": {}}}, row
    return event, row


# ── what he is told ──

def note(outcome):
    """The one generated line for an outcome dict, or None when active."""
    if not outcome or outcome["outcome"] == "active":
        return None
    if outcome["outcome"] == "unreadable":
        return "Evallo sessions could not be checked"
    moment = parse_moment(outcome.get("verified_at"))
    if moment is None:
        return "Evallo sessions never verified"
    local = moment.astimezone(NY)
    return f"Evallo sessions not verified since {local.strftime('%a %b')} {local.day}, " \
           f"{local.strftime('%-I:%M %p')}"


def worst(outcomes):
    """The most cautious outcome among several (unreadable, off, stale)."""
    found = [o for o in outcomes if o]
    return max(found, key=lambda o: _RANK[o["outcome"]]) if found else None


_collector = contextvars.ContextVar("evallo_outcomes", default=None)


@contextlib.contextmanager
def collecting():
    """Collects the outcome of every calendar read made inside the block, so a
    caller that builds an answer from several reads can say what state its
    events came from, without a read of its own."""
    seen = []
    outer = _collector.get()
    token = _collector.set(seen)
    try:
        yield seen
    finally:
        _collector.reset(token)
        if outer is not None:
            outer.extend(seen)       # an enclosing collector sees these reads too


def record(outcome):
    seen = _collector.get()
    if seen is not None:
        seen.append(outcome)


# ── storing a push ──

def store_snapshot(fetched_at, window_start, window_end, session_ids, now=None):
    """Replace the set and the status row in one BEGIN IMMEDIATE transaction.
    Returns (count, fetched_at text, pushed_at text). Raises Conflict. Sets the
    active flag in the same transaction as the replacement."""
    now = now or now_utc()
    fetched_at = fetched_at.replace(microsecond=0)    # whole seconds, compared and stored
    fetched_text, pushed_text = iso_z(fetched_at), iso_z(now)
    ids = sorted(set(session_ids))
    conn = sqlite3.connect(database.DB_PATH, timeout=10, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT fetched_at, pushed_at, window_start, window_end, count "
                               "FROM evallo_verification WHERE id = 1").fetchone()
            if row is not None:
                stored = parse_moment(row[0])
                if stored is not None and fetched_at < stored:
                    raise Conflict("older_snapshot")
                if stored is not None and fetched_at == stored:
                    same_ids = {r[0] for r in conn.execute(
                        "SELECT session_id FROM evallo_verified_sessions")} == set(ids)
                    if row[2] == window_start.isoformat() and row[3] == window_end.isoformat() and same_ids:
                        conn.execute("UPDATE evallo_verification SET active = 1 WHERE id = 1")
                        conn.execute("COMMIT")
                        return row[4], row[0], row[1]
                    raise Conflict("conflicting_snapshot")
            conn.execute("DELETE FROM evallo_verified_sessions")
            conn.executemany("INSERT INTO evallo_verified_sessions (session_id) VALUES (?)",
                             [(i,) for i in ids])
            conn.execute("INSERT OR REPLACE INTO evallo_verification "
                         "(id, fetched_at, pushed_at, window_start, window_end, count, active) "
                         "VALUES (1, ?, ?, ?, ?, ?, 1)",
                         (fetched_text, pushed_text, window_start.isoformat(),
                          window_end.isoformat(), len(ids)))
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return len(ids), fetched_text, pushed_text
