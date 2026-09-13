"""Oura ring data, as READ tools.

Health data is READ in kind and private in content, so every tool here carries
min_tier hokage.

Every field says its unit. The first sleep tool returned Oura's contributor
scores under the names total_sleep and deep_sleep. Those are ratings out of a
hundred, not durations, and Nova read total_sleep 100 as a hundred minutes and
told Lethanial he had slept an hour and forty on a night of ten hours forty
three. A number with no unit is a number the model will assign one to.

Arithmetic happens here, not in the model. The heart rate tool summarizes its
samples in code and returns a handful of numbers, rather than handing a few
thousand characters of raw readings to a model that would average them by eye.

Failures raise. The executor in brain.py turns an exception into an is_error
result, so Nova says the ring is unreachable instead of reading an HTTP body.
"""

import datetime
import fcntl
import json
import os
import statistics
from contextlib import contextmanager

from requests_oauthlib import OAuth2Session

import config  # noqa: F401  loads .env, which holds the client id and secret
from tools import Permission, tool

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TOKEN_FILE = os.path.join(_DATA_DIR, "oura_token.json")
_LOCK_FILE = TOKEN_FILE + ".lock"
_API = "https://api.ouraring.com/v2/usercollection"
_TOKEN_URL = "https://api.ouraring.com/oauth/token"
_TIMEOUT_S = 10

# Recent enough that a daytime question is about today, long enough to hold a
# handful of the ring's five minute readings.
_HEART_RATE_WINDOW_HOURS = 3


@contextmanager
def _token_lock():
    """Held around every request, across processes.

    miles-voice and miles-server are separate processes sharing one token file.
    If both refreshed at once, the second would send a refresh token the first
    had already spent, and a provider that rotates refresh tokens rejects a
    spent one, which means running oura_auth.py again by hand. With the lock the
    second process waits, then reads the token the first one saved."""
    with open(_LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _save_token(token):
    """Write then rename, so a crash mid write leaves the old token in place
    rather than half a file neither service can parse."""
    tmp = TOKEN_FILE + ".tmp"
    # Owner only. The token is a standing grant to his health data.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(token, f)
    os.replace(tmp, TOKEN_FILE)


def _get(path, **params):
    """GET one endpoint and return its data list."""
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError("no Oura token; run scripts/oura_auth.py")

    # Read at call time rather than import time, so a rotated secret in .env
    # takes effect on restart without anything caching the old one.
    client_id = os.environ.get("OURA_CLIENT_ID")
    client_secret = os.environ.get("OURA_CLIENT_SECRET")

    with _token_lock():
        # Read inside the lock, so a token another process just refreshed is
        # the one used.
        with open(TOKEN_FILE) as f:
            token = json.load(f)
        session = OAuth2Session(
            client_id,
            token=token,
            auto_refresh_url=_TOKEN_URL,
            auto_refresh_kwargs={"client_id": client_id, "client_secret": client_secret},
            token_updater=_save_token,
        )
        response = session.get(f"{_API}/{path}", params=params, timeout=_TIMEOUT_S)

    response.raise_for_status()
    return response.json().get("data", [])


def _day_window(today=None):
    """The last few days, asked for explicitly.

    Leaving the range off leans on whatever Oura defaults to, which made
    "latest" depend on an undocumented choice. end_date is tomorrow because a
    night that ended this morning is filed under today."""
    today = today or datetime.date.today()
    return {
        "start_date": (today - datetime.timedelta(days=2)).isoformat(),
        "end_date": (today + datetime.timedelta(days=1)).isoformat(),
    }


def _latest_day(items):
    return max(items, key=lambda item: item.get("day", "")) if items else None


def duration_words(seconds):
    """"10 hours 43 minutes". Words carry the unit all the way to speech, where
    "10h 43m" asks the model to expand an abbreviation and a bare number asks it
    to guess."""
    if not isinstance(seconds, (int, float)):
        return None
    hours, minutes = divmod(int(seconds) // 60, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'' if hours == 1 else 's'}")
    if minutes or not hours:
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")
    return " ".join(parts)


def summarize_sleep(periods, daily):
    """The most recent main sleep, with durations from the sleep periods and the
    score from daily_sleep. Naps are skipped when a main sleep exists, because
    "how did I sleep" means the night."""
    nights = [p for p in periods if p.get("type") == "long_sleep"] or periods
    if not nights:
        return None
    night = max(nights, key=lambda p: p.get("bedtime_end", ""))
    score = next((d.get("score") for d in daily if d.get("day") == night.get("day")), None)
    return {
        "date": night.get("day"),
        "sleep_score_out_of_100": score,
        "total_sleep": duration_words(night.get("total_sleep_duration")),
        "time_in_bed": duration_words(night.get("time_in_bed")),
        "deep_sleep": duration_words(night.get("deep_sleep_duration")),
        "rem_sleep": duration_words(night.get("rem_sleep_duration")),
        "efficiency_percent": night.get("efficiency"),
    }


def summarize_heart_rate(samples, now):
    readings = [s for s in samples if isinstance(s.get("bpm"), (int, float))]
    if not readings:
        return None
    bpms = [s["bpm"] for s in readings]
    last = max(readings, key=lambda s: s.get("timestamp", ""))
    taken = datetime.datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
    return {
        "latest_bpm": last["bpm"],
        "latest_reading_minutes_ago": round((now - taken).total_seconds() / 60),
        "average_bpm": round(statistics.mean(bpms)),
        "lowest_bpm": min(bpms),
        "highest_bpm": max(bpms),
        "readings": len(bpms),
        "window_hours": _HEART_RATE_WINDOW_HOURS,
    }


_NO_INPUT = {"type": "object", "properties": {}, "required": []}


@tool(
    name="get_oura_readiness",
    description=(
        "Lethanial's latest readiness score from his Oura ring, with the HRV "
        "balance and recovery index behind it. Call this when he asks how "
        "recovered or ready he is, whether he should train hard today, or about "
        "his readiness or HRV. Every value is a score out of one hundred, not a "
        "measurement."
    ),
    input_schema=_NO_INPUT,
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_oura_readiness():
    latest = _latest_day(_get("daily_readiness", **_day_window()))
    if latest is None:
        return "No readiness data from the last three days."
    contributors = latest.get("contributors", {})
    return {
        "date": latest.get("day"),
        "readiness_score_out_of_100": latest.get("score"),
        "hrv_balance_score_out_of_100": contributors.get("hrv_balance"),
        "recovery_index_score_out_of_100": contributors.get("recovery_index"),
    }


@tool(
    name="get_oura_sleep",
    description=(
        "Lethanial's most recent night of sleep from his Oura ring: total sleep, "
        "time in bed, deep and REM sleep as durations, efficiency as a percent, "
        "and the sleep score out of one hundred. Call this when he asks how he "
        "slept, how long he slept, or about his deep or REM sleep."
    ),
    input_schema=_NO_INPUT,
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_oura_sleep():
    window = _day_window()
    summary = summarize_sleep(_get("sleep", **window), _get("daily_sleep", **window))
    return summary or "No sleep recorded in the last three days."


@tool(
    name="get_oura_heartrate",
    description=(
        "Lethanial's recent heart rate from his Oura ring over the last few "
        "hours: the latest reading and how long ago it was taken, plus the "
        "average, lowest and highest. Call this when he asks what his heart "
        "rate is or has been."
    ),
    input_schema=_NO_INPUT,
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_oura_heartrate():
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(hours=_HEART_RATE_WINDOW_HOURS)
    samples = _get("heartrate",
                   start_datetime=start.isoformat(timespec="seconds"),
                   end_datetime=now.isoformat(timespec="seconds"))
    summary = summarize_heart_rate(samples, now)
    return summary or (f"No heart rate readings in the last "
                       f"{_HEART_RATE_WINDOW_HOURS} hours. The ring may be off or charging.")


@tool(
    name="get_oura_activity",
    description=(
        "Lethanial's activity for the latest day from his Oura ring: steps, "
        "active calories, and the activity score out of one hundred. Call this "
        "when he asks how active he has been, his step count, or calories burned."
    ),
    input_schema=_NO_INPUT,
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_oura_activity():
    latest = _latest_day(_get("daily_activity", **_day_window()))
    if latest is None:
        return "No activity data from the last three days."
    return {
        "date": latest.get("day"),
        "activity_score_out_of_100": latest.get("score"),
        "steps": latest.get("steps"),
        "active_calories": latest.get("active_calories"),
    }
