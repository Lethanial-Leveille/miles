"""Approvals: what must be tapped in Nova before anything outside the Pi happens.

An agent (the Mac) proposes an action: a text message, an Evallo edit. It is
stored here with the exact words and the chat it is for, pushed to the phone,
and does nothing until Lethanial taps Approve on that one request.

Rules this module enforces, in code, so no caller can forget them:

  * One request, one tap. There is no batch approve and nothing approves itself.
    The routes that approve take a full login; the Mac's token is scope "agent"
    and is refused there (server.py).
  * A request expires 2 hours after it was proposed, approved or not. An
    approval that no agent claimed in time is expired, never run later. An
    expired request does nothing.
  * A claimed request is handed out once. If the agent dies mid-send it stays
    "claimed" and is NOT retried: a text that may have gone out must not go out
    twice. It shows up in the event log for a human to look at.
  * Every step (proposed, approved, skipped, expired, claimed, done, failed) is a
    row in approval_events with a timestamp, the exact text and the chat.

pending_action.py is the voice-turn gate and lives in memory for two minutes;
it cannot hold a request across a restart, be tapped from a phone or be claimed
by another machine, so it is not used here.
"""

import json
import os
import re
import sqlite3

from runtime_profile import current, headless
from datetime import datetime, timedelta, timezone

import database

TTL = timedelta(hours=2)
# Stage D sends texts; Stage E adds the reschedule. Sessions are on one fixed
# Zoom link, so nothing adds meeting links to Evallo any more.
KINDS = {"imessage", "evallo_reschedule"}
MAX_TEXT_CHARS = 400

PENDING, APPROVED, SKIPPED, EXPIRED, CLAIMED, DONE, FAILED = (
    "pending", "approved", "skipped", "expired", "claimed", "done", "failed")


class ApprovalError(Exception):
    """A refusal with an HTTP-ish code: 404 unknown, 409 already decided, 410 expired."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _now(now=None):
    return now or datetime.now(timezone.utc)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _conn():
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


_URL = re.compile(r"https?://\S+|\b[\w.-]+\.(?:com|org|net|io|me|ly|us|edu)/\S*", re.I)
# His fixed UF Zoom meeting, pasted by hand into data/zoom.json as {"link": "..."}.
# The Mac has its own copy; the two must match exactly or the Pi refuses the text.
ZOOM_PATH = str(current().root / "missing-zoom.json") if headless() else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zoom.json")
# Query: pwd and/or omn (Zoom's meeting-instance number), nothing else.
_Q = r"(pwd=[A-Za-z0-9_-]+(\.\d+)?|omn=\d+)"
_ZOOM = re.compile(r"^https://([a-z0-9-]+\.)?zoom\.us/(j/\d{9,11}|my/[A-Za-z0-9_-]+)"
                   rf"(\?{_Q}(&{_Q})?)?$")


def zoom_link():
    """The one link a text may carry, or None (then no link is allowed at all).
    Read on every check, so a changed file needs no restart."""
    try:
        with open(ZOOM_PATH) as f:
            link = (json.load(f).get("link") or "").strip()
    except (OSError, ValueError, AttributeError):
        return None
    return link if _ZOOM.match(link) else None


def check_text(text):
    """Texts go to students and parents, many of them minors: short, and no link
    except his one Zoom link from data/zoom.json. Raises ValueError with the reason."""
    text = (text or "").strip()
    if not text:
        raise ValueError("The message is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"The message is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.")
    urls = _URL.findall(text)
    if urls:
        allowed = zoom_link()
        for url in urls:
            if allowed is None:
                raise ValueError("No links are allowed in a message: the Pi has no Zoom link in "
                                 f"data/zoom.json, so {url!r} was refused.")
            if url.rstrip(".,!?)") != allowed:
                raise ValueError(f"The only link allowed in a message is the Zoom link, not {url!r}.")
    return text


# A reminder is several iMessages from one approval (the words, then the Zoom
# link alone so Messages shows its preview card). text is the parts joined by
# PART_SEP: the card shows it, and the Mac sends the parts only if they join
# back to exactly the text that was approved.
PART_SEP = "\n\n"
MAX_PARTS = 3


def check_parts(text, parts):
    """Each part must pass check_text on its own, and the parts must join to
    exactly text. Raises ValueError with the reason."""
    if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS \
            or not all(isinstance(p, str) for p in parts):
        raise ValueError(f"A reminder is 1 to {MAX_PARTS} messages.")
    for p in parts:
        if check_text(p) != p:
            raise ValueError("Each message must have no leading or trailing spaces.")
    if PART_SEP.join(parts) != text:
        raise ValueError("The messages do not join to the text shown for approval.")


def _event(conn, row, event, detail=None, now=None):
    conn.execute(
        "INSERT INTO approval_events (at, approval_id, event, kind, chat_name, text, detail) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_iso(_now(now)), row["id"], event, row["kind"], row["chat_name"], row["text"], detail))


def _public(row):
    d = dict(row)
    d["payload"] = json.loads(d["payload"]) if d.get("payload") else {}
    return d


def propose(kind, summary, text, chat_name, payload=None, dedupe_key=None, now=None):
    """Store a request. Returns (row, created). A repeat of a dedupe_key returns
    the existing row, whatever became of it, so a Skip, an expiry or a rerun of
    the nightly job never re-asks about the same thing."""
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}.")
    if kind == "imessage":
        text = check_text(text)
        if (payload or {}).get("parts") is not None:
            check_parts(text, payload["parts"])
    now = _now(now)
    conn = _conn()
    try:
        if dedupe_key:
            old = conn.execute("SELECT * FROM approvals WHERE dedupe_key = ?", (dedupe_key,)).fetchone()
            if old:
                return _public(old), False
        cur = conn.execute(
            "INSERT INTO approvals (kind, summary, text, chat_name, payload, dedupe_key, status, "
            "created_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (kind, summary, text, chat_name, json.dumps(payload or {}), dedupe_key,
             _iso(now), _iso(now + TTL)))
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (cur.lastrowid,)).fetchone()
        _event(conn, row, "proposed", now=now)
        conn.commit()
        return _public(row), True
    finally:
        conn.close()


def _expire_due(conn, now):
    """Anything past its time that has not been claimed is expired for good."""
    rows = conn.execute(
        "SELECT * FROM approvals WHERE status IN ('pending', 'approved') AND expires_at <= ?",
        (_iso(now),)).fetchall()
    for row in rows:
        conn.execute("UPDATE approvals SET status = 'expired' WHERE id = ?", (row["id"],))
        _event(conn, row, "expired", detail=f"was {row['status']}", now=now)
    return len(rows)


def expire_due(now=None):
    conn = _conn()
    try:
        n = _expire_due(conn, _now(now))
        conn.commit()
        return n
    finally:
        conn.close()


def get(approval_id, now=None):
    conn = _conn()
    try:
        _expire_due(conn, _now(now))
        conn.commit()
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise ApprovalError(404, "No such request.")
        return _public(row)
    finally:
        conn.close()


def list_pending(now=None):
    conn = _conn()
    try:
        _expire_due(conn, _now(now))
        conn.commit()
        rows = conn.execute(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY id").fetchall()
        return [_public(r) for r in rows]
    finally:
        conn.close()


def decide(approval_id, approve, now=None):
    """Approve or skip ONE request. Expired or already decided requests refuse."""
    now = _now(now)
    conn = _conn()
    try:
        _expire_due(conn, now)
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise ApprovalError(404, "No such request.")
        if row["status"] == EXPIRED:
            conn.commit()
            raise ApprovalError(410, "That request expired, so nothing was done.")
        if row["status"] != PENDING:
            raise ApprovalError(409, f"That request is already {row['status']}.")
        new = APPROVED if approve else SKIPPED
        conn.execute("UPDATE approvals SET status = ?, decided_at = ? WHERE id = ?",
                     (new, _iso(now), approval_id))
        _event(conn, row, new, now=now)
        conn.commit()
        return _public(conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone())
    finally:
        conn.close()


def claim_approved(kinds=None, now=None):
    """Approved, unexpired requests, each handed out exactly once."""
    now = _now(now)
    conn = _conn()
    try:
        _expire_due(conn, now)
        rows = conn.execute(
            "SELECT * FROM approvals WHERE status = 'approved' ORDER BY id").fetchall()
        out = []
        for row in rows:
            if kinds and row["kind"] not in kinds:
                continue
            conn.execute("UPDATE approvals SET status = 'claimed', claimed_at = ? WHERE id = ?",
                         (_iso(now), row["id"]))
            _event(conn, row, "claimed", now=now)
            out.append(_public(row))
        conn.commit()
        return out
    finally:
        conn.close()


def report(approval_id, ok, result="", now=None):
    now = _now(now)
    conn = _conn()
    try:
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None:
            raise ApprovalError(404, "No such request.")
        if row["status"] != CLAIMED:
            raise ApprovalError(409, f"That request is {row['status']}, not claimed.")
        new = DONE if ok else FAILED
        conn.execute("UPDATE approvals SET status = ?, finished_at = ?, result = ? WHERE id = ?",
                     (new, _iso(now), result[:1000], approval_id))
        _event(conn, row, new, detail=result[:1000], now=now)
        conn.commit()
        return new
    finally:
        conn.close()


def events(limit=100):
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM approval_events ORDER BY id DESC LIMIT ?", (limit,))]
    finally:
        conn.close()


# ── Which chat each student's texts go to (names only; ids stay on the Mac) ──

def set_contacts(mapping, now=None):
    """Replace the table with what the Mac's data/contacts.json maps. Only the
    chat's display name is kept, so a tool here can show him where a text would
    go. The Mac alone knows the chat id and never sends it."""
    stamp = _iso(_now(now))
    conn = _conn()
    try:
        conn.execute("DELETE FROM imessage_contacts")
        for first, chat in mapping.items():
            conn.execute("INSERT INTO imessage_contacts (first_name, chat_name, updated_at) "
                         "VALUES (?, ?, ?)", (first.strip().lower(), chat.strip(), stamp))
        conn.commit()
    finally:
        conn.close()


def chat_name_for(first_name):
    conn = _conn()
    try:
        row = conn.execute("SELECT chat_name FROM imessage_contacts WHERE first_name = ?",
                           (first_name.strip().lower(),)).fetchone()
        return row["chat_name"] if row else None
    finally:
        conn.close()


# ── Telling the phone ──

APPROVAL_CATEGORY = "MILES_APPROVAL"


def notify_new(row):
    """Push one request to the phone with Approve and Skip on the lock screen.
    The buttons are the app's notification category; tapping one opens the app
    on that request (it never approves from the lock screen blind)."""
    import push
    if row["kind"] == "imessage":
        body = f"To {row['chat_name']}: {row['text']}"
    else:
        body = row["summary"]
    return push.notify("Approve?", body, kind="approval",
                       data={"kind": "approval", "approval_id": row["id"],
                             "expires_at": row["expires_at"]},
                       category=APPROVAL_CATEGORY, collapse_id=f"approval-{row['id']}")
