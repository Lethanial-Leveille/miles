"""The Mac watchdog, and login-expired pushes for every saved session.

The Mac agent checks in on every run and every 30 s poll (heartbeat) and
reports each scheduled job (job_report). check(), run every five minutes by
miles-agent-watchdog.timer, pushes when:

  * there has been no heartbeat since 7 PM by 7:15 PM (the 7 PM drafts), or
  * a scheduled job did not report success within 30 minutes of its time.

One alert per job per day, with three buttons in Nova:

  Retry when awake   the Mac runs the missed job the next time it checks in
  Remind me at 7 AM  the same push again at 7 AM
  Skip tonight       nothing more about it

A job that reports success later on its own (launchd runs it on wake) closes
its alert. Texts are still only proposed, never sent, and the Mac never drafts
one for a session that has already started.

login_expired() is the one way any part of the system says a saved login
(UF Zoom, Evallo, Google) stopped working: one push per service per day, with
the exact command that fixes it and the job that failed.
"""

import sqlite3
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import database

TZ = ZoneInfo("America/New_York")

# job -> (when it runs, Eastern; what to call it in a push)
JOBS = {
    "draft": (time(19, 0), "The 7 PM reminder drafts"),
    "summaries": (time(21, 0), "The 9 PM Zoom summary grab"),
}
GRACE = timedelta(minutes=30)
HEARTBEAT_BY = time(19, 15)       # and since HEARTBEAT_SINCE
HEARTBEAT_SINCE = time(19, 0)
REMIND_AT = time(7, 0)
DECISIONS = ("retry", "remind", "skip")
CATEGORY = "MILES_WATCHDOG"

MAC = "cd ~/PersonalProjects/miles-evallo-mac && .venv-evallo/bin/python"
SERVICES = {
    "zoom": ("UF Zoom", f"{MAC} scripts/zoom_login.py"),
    "google": ("Google", f"{MAC} scripts/gcal_login.py --client-secret "
                         "~/PersonalProjects/client_secret.json"),
    "evallo": ("Evallo", f"{MAC} scripts/evallo_login.py && scp data/evallo_state.json "
                         "theycallmelee@100.99.248.127:~/miles/data/"),
}


def _conn():
    conn = sqlite3.connect(database.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _now(now=None):
    return (now or datetime.now(TZ)).astimezone(TZ)


def _iso(dt):
    return dt.astimezone(TZ).isoformat(timespec="seconds")


def _at(day, t):
    return datetime.combine(day, t, tzinfo=TZ)


def _push(title, body, data=None, category=None, collapse_id=None, kind="watchdog"):
    # kind, level and detail ride in the push: a plain tap opens Nova's error
    # screen with the whole message.
    data = {"kind": kind, "level": "error", "detail": body, **(data or {})}
    try:
        import push
        return push.notify(title, body, kind=kind, data=data, category=category,
                           collapse_id=collapse_id)
    except Exception as exc:
        print(f"watchdog: push failed ({type(exc).__name__}: {exc}): {body}", flush=True)
        return 0


# ── What the Mac sends ──

def heartbeat(source, now=None):
    """Record a check in. Returns the jobs he asked to retry that the Mac has
    not been handed yet, each handed out once: [{"job", "day"}]."""
    now = _now(now)
    conn = _conn()
    try:
        conn.execute("INSERT INTO agent_heartbeats (source, at) VALUES (?, ?) "
                     "ON CONFLICT(source) DO UPDATE SET at = excluded.at", (source, _iso(now)))
        rows = conn.execute(
            "SELECT id, job, day FROM watchdog_alerts WHERE decision = 'retry' "
            "AND retry_sent_at IS NULL AND resolved_at IS NULL").fetchall()
        for r in rows:
            conn.execute("UPDATE watchdog_alerts SET retry_sent_at = ? WHERE id = ?",
                         (_iso(now), r["id"]))
        conn.commit()
        return [{"job": r["job"], "day": r["day"]} for r in rows]
    finally:
        conn.close()


def last_heartbeat(conn):
    row = conn.execute("SELECT MAX(at) AS at FROM agent_heartbeats").fetchone()
    return datetime.fromisoformat(row["at"]) if row and row["at"] else None


def job_report(job, day, ok, detail="", now=None):
    """A scheduled job finished (ok) or could not (not ok, with why). Success
    closes that day's alert, whoever caused the run."""
    if job not in JOBS:
        raise ValueError(f"Unknown job {job!r}.")
    datetime.strptime(day, "%Y-%m-%d")
    now = _now(now)
    conn = _conn()
    try:
        conn.execute(
            "INSERT INTO agent_jobs (job, day, reported_at, ok, detail) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(job, day) DO UPDATE SET reported_at = excluded.reported_at, "
            "ok = MAX(agent_jobs.ok, excluded.ok), detail = excluded.detail",
            (job, day, _iso(now), int(bool(ok)), (detail or "")[:2000]))
        if ok:
            conn.execute("UPDATE watchdog_alerts SET resolved_at = ? "
                         "WHERE job = ? AND day = ? AND resolved_at IS NULL",
                         (_iso(now), job, day))
        conn.commit()
    finally:
        conn.close()


# ── The five minute check ──

def _job_done(conn, job, day):
    row = conn.execute("SELECT ok FROM agent_jobs WHERE job = ? AND day = ?",
                       (job, day.isoformat())).fetchone()
    return bool(row and row["ok"])


def _why_late(conn, job, day, now):
    """Why this job needs an alert now, or None."""
    if _job_done(conn, job, day):
        return None
    run_at, _ = JOBS[job]
    if job == "draft" and now >= _at(day, HEARTBEAT_BY):
        beat = last_heartbeat(conn)
        if beat is None or beat < _at(day, HEARTBEAT_SINCE):
            return "The Mac has not checked in since 7 PM (asleep or off?)."
    if now >= _at(day, run_at) + GRACE:
        row = conn.execute("SELECT detail FROM agent_jobs WHERE job = ? AND day = ?",
                           (job, day.isoformat())).fetchone()
        if row:
            return f"It did not finish: {row['detail'] or 'no reason given'}"
        return "It did not report in within 30 minutes."
    return None


def check(now=None):
    """Raise new alerts and repeat due 7 AM reminders. Returns what it pushed."""
    now = _now(now)
    pushed = []
    conn = _conn()
    try:
        for job in JOBS:
            # Today's run, and yesterday's when it is past midnight and that one
            # never got its alert (a Pi that was down at the time).
            for day in (now.date(), now.date() - timedelta(days=1)):
                if day < now.date() and now >= _at(now.date(), REMIND_AT):
                    continue
                if conn.execute("SELECT 1 FROM watchdog_alerts WHERE job = ? AND day = ?",
                                (job, day.isoformat())).fetchone():
                    continue
                why = _why_late(conn, job, day, now)
                if why is None:
                    continue
                cur = conn.execute(
                    "INSERT INTO watchdog_alerts (job, day, reason, created_at) VALUES (?, ?, ?, ?)",
                    (job, day.isoformat(), why, _iso(now)))
                conn.commit()
                pushed.append(_alert_push(conn, cur.lastrowid))
        for row in conn.execute(
                "SELECT id FROM watchdog_alerts WHERE decision = 'remind' AND remind_at <= ? "
                "AND resolved_at IS NULL", (_iso(now),)).fetchall():
            conn.execute("UPDATE watchdog_alerts SET decision = NULL, remind_at = NULL, "
                         "reminded_at = ? WHERE id = ?", (_iso(now), row["id"]))
            conn.commit()
            pushed.append(_alert_push(conn, row["id"]))
    finally:
        conn.close()
    return pushed


def _alert_push(conn, alert_id):
    a = conn.execute("SELECT * FROM watchdog_alerts WHERE id = ?", (alert_id,)).fetchone()
    _, label = JOBS[a["job"]]
    day = datetime.strptime(a["day"], "%Y-%m-%d").strftime("%a %b %-d")
    body = f"{label} for {day} did not run. {a['reason']}"
    _push("Mac missed a job", body, data={"watchdog_id": a["id"]}, category=CATEGORY,
          collapse_id=f"watchdog-{a['id']}")
    conn.execute("UPDATE watchdog_alerts SET pushed_at = ? WHERE id = ?",
                 (_iso(_now()), alert_id))
    conn.commit()
    return body


# ── What he taps ──

class AlertError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def next_remind_time(now):
    at = _at(now.date(), REMIND_AT)
    return at if at > now else at + timedelta(days=1)


def decide(alert_id, decision, now=None):
    if decision not in DECISIONS:
        raise AlertError(422, f"Unknown decision {decision!r}.")
    now = _now(now)
    conn = _conn()
    try:
        a = conn.execute("SELECT * FROM watchdog_alerts WHERE id = ?", (alert_id,)).fetchone()
        if a is None:
            raise AlertError(404, "No such alert.")
        if a["resolved_at"]:
            return {"status": "already_done", "message": "That job has since run on its own."}
        remind_at = _iso(next_remind_time(now)) if decision == "remind" else None
        conn.execute("UPDATE watchdog_alerts SET decision = ?, decided_at = ?, remind_at = ?, "
                     "retry_sent_at = NULL, resolved_at = ? WHERE id = ?",
                     (decision, _iso(now), remind_at,
                      _iso(now) if decision == "skip" else None, alert_id))
        conn.commit()
    finally:
        conn.close()
    return {"status": decision, "message": {
        "retry": "The Mac will run it the next time it checks in.",
        "remind": "I'll ask again at 7 AM.",
        "skip": "Skipped. Nothing more about it.",
    }[decision]}


def open_alerts():
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM watchdog_alerts WHERE resolved_at IS NULL ORDER BY id DESC LIMIT 20")]
    finally:
        conn.close()


# ── Saved logins ──

def login_expired(service, job, now=None):
    """Push "<Service> login expired" with the command that fixes it, at most
    once per service per day. Returns whether this call pushed."""
    if service not in SERVICES:
        raise ValueError(f"Unknown service {service!r}.")
    name, command = SERVICES[service]
    now = _now(now)
    conn = _conn()
    try:
        try:
            conn.execute("INSERT INTO login_alerts (service, day, job, at) VALUES (?, ?, ?, ?)",
                         (service, now.date().isoformat(), (job or "")[:200], _iso(now)))
            conn.commit()
        except sqlite3.IntegrityError:
            return False
    finally:
        conn.close()
    _push(f"{name} login expired",
          f"{name} login expired (password change?). On the Mac run: {command}  "
          f"Failed job: {job}.", collapse_id=f"login-{service}", kind="login")
    return True
