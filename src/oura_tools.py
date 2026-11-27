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

from runtime_profile import refuse_live

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
    refuse_live("Oura")
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


def wake_from_periods(periods, now):
    """The end of his most recent main sleep that has already ended, as a naive
    local time, or None. Naps do not start a new day, so only long sleeps count
    when there are any."""
    mains = [p for p in periods if p.get("type") == "long_sleep"] or \
            [p for p in periods if p.get("type") == "sleep"]
    ends = []
    for p in mains:
        end = p.get("bedtime_end")
        if not end:
            continue
        t = datetime.datetime.fromisoformat(end)
        if t.tzinfo:
            t = t.astimezone().replace(tzinfo=None)
        if t <= now:
            ends.append(t)
    return max(ends) if ends else None


def last_wake(now=None):
    """When he last woke up, from the ring, or None if the ring cannot say."""
    now = now or datetime.datetime.now()
    try:
        return wake_from_periods(_get("sleep", **_day_window(now.date())), now)
    except Exception as exc:
        print(f"Oura wake time unavailable: {exc}", flush=True)
        return None


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


# Nov 21 2026: "how did I sleep" came back as efficiency 76%, deep 1:46, REM
# 1:41, score 75. A person asked the same question says roughly how long you
# slept, when you dropped off and woke, and whether that is normal for you. So
# the summary leads with that, in rounded clock times and plain durations, and
# keeps the instrument readings under if_he_asks_for_detail.

# Nights to compare against. Two weeks is enough for a usual that a single bad
# night does not move, short enough to follow a semester changing his hours.
USUAL_NIGHTS = 14
_MIN_NIGHTS_FOR_USUAL = 3
# Differences smaller than these are noise, and "eleven minutes more than
# usual" is the kind of precision nobody asked for.
_SAME_DURATION_MIN = 30
_SAME_CLOCK_MIN = 45
_AWAKE_IN_BED_WORTH_SAYING_MIN = 45
_NAP_WORTH_SAYING_MIN = 15


def _local(stamp):
    t = datetime.datetime.fromisoformat(stamp)
    return t.astimezone().replace(tzinfo=None) if t.tzinfo else t


def about_duration(seconds):
    """"about 7 and a half hours": to the nearest quarter hour, as said aloud."""
    if not isinstance(seconds, (int, float)):
        return None
    quarters = round(seconds / 900)
    hours, q = divmod(quarters, 4)
    if hours == 0:
        minutes = max(5, round(seconds / 300) * 5)
        return f"about {minutes} minutes"
    if hours == 1:
        return {0: "about an hour", 1: "a little over an hour",
                2: "about an hour and a half", 3: "almost 2 hours"}[q]
    return {0: f"about {hours} hours",
            1: f"a little over {hours} hours",
            2: f"about {hours} and a half hours",
            3: f"almost {hours + 1} hours"}[q]


def clock(t):
    """Rounded to five minutes, which is as exact as "around" means."""
    t = t + datetime.timedelta(minutes=2, seconds=30)
    t = t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)
    return t.strftime("%-I:%M %p").replace(":00 ", " ")


def _compare_minutes(diff, same, more, less):
    if abs(diff) < same:
        return "about the same as usual"
    words = about_duration(abs(diff) * 60)
    return f"{words} {more if diff > 0 else less} than usual"


def _minutes_after_six_pm(t):
    """Bedtimes straddle midnight, so they are compared on an evening clock."""
    return (t.hour * 60 + t.minute - 18 * 60) % 1440


def _rating(score):
    if score is None:
        return None
    return ("great" if score >= 85 else "good" if score >= 75
            else "okay" if score >= 65 else "rough")


def summarize_sleep(periods, daily):
    """The most recent main sleep, said the way a person would say it.

    Naps are skipped when a main sleep exists, because "how did I sleep" means
    the night; naps after it are mentioned on their own."""
    mains = [p for p in periods if p.get("type") == "long_sleep"] or periods
    if not mains:
        return None
    night = max(mains, key=lambda p: p.get("bedtime_end", ""))
    score = next((d.get("score") for d in daily if d.get("day") == night.get("day")), None)

    in_bed = _local(night["bedtime_start"]) if night.get("bedtime_start") else None
    asleep = in_bed + datetime.timedelta(seconds=night.get("latency") or 0) if in_bed else None
    woke = _local(night["bedtime_end"]) if night.get("bedtime_end") else None
    total = night.get("total_sleep_duration")
    awake_s = (night.get("time_in_bed") or 0) - (total or 0)

    out = {
        "night": (f"{asleep.strftime('%A')} night" if asleep and asleep.hour >= 12
                  else f"{(asleep - datetime.timedelta(days=1)).strftime('%A')} night"
                  if asleep else night.get("day")),
        "slept": about_duration(total),
        "fell_asleep_around": clock(asleep) if asleep else None,
        "woke_up_around": clock(woke) if woke else None,
        "awake_in_bed": (about_duration(awake_s)
                         if awake_s >= _AWAKE_IN_BED_WORTH_SAYING_MIN * 60 else None),
        "overall": _rating(score),
        "sleep_compared_to_usual": None,
        "bedtime_compared_to_usual": None,
        "wake_compared_to_usual": None,
        "naps_since": [
            f"{about_duration(n.get('total_sleep_duration'))} around {clock(_local(n['bedtime_start']))}"
            for n in periods
            if n is not night and n.get("type") != "long_sleep" and n.get("bedtime_start")
            and woke and _local(n["bedtime_start"]) > woke
            and (n.get("total_sleep_duration") or 0) >= _NAP_WORTH_SAYING_MIN * 60
        ],
        "if_he_asks_for_detail": {
            "sleep_score_out_of_100": score,
            "efficiency_percent": night.get("efficiency"),
            "deep_sleep": duration_words(night.get("deep_sleep_duration")),
            "rem_sleep": duration_words(night.get("rem_sleep_duration")),
            "time_in_bed": duration_words(night.get("time_in_bed")),
            "lowest_heart_rate_bpm": night.get("lowest_heart_rate"),
            "average_hrv_ms": night.get("average_hrv"),
        },
    }

    before = [p for p in periods if p.get("type") == "long_sleep" and p is not night
              and p.get("bedtime_end", "") < night.get("bedtime_end", "")][-USUAL_NIGHTS:]
    if len(before) >= _MIN_NIGHTS_FOR_USUAL and asleep and woke and total:
        usual_total = statistics.median(p["total_sleep_duration"] for p in before
                                        if p.get("total_sleep_duration"))
        out["sleep_compared_to_usual"] = _compare_minutes(
            (total - usual_total) / 60, _SAME_DURATION_MIN, "more", "less")
        usual_asleep = statistics.median(
            _minutes_after_six_pm(_local(p["bedtime_start"])
                                  + datetime.timedelta(seconds=p.get("latency") or 0))
            for p in before)
        out["bedtime_compared_to_usual"] = _compare_minutes(
            _minutes_after_six_pm(asleep) - usual_asleep, _SAME_CLOCK_MIN, "later", "earlier")
        usual_woke = statistics.median(_minutes_after_six_pm(_local(p["bedtime_end"]))
                                       for p in before)
        out["wake_compared_to_usual"] = _compare_minutes(
            _minutes_after_six_pm(woke) - usual_woke, _SAME_CLOCK_MIN, "later", "earlier")
    return out


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
        "Lethanial's most recent night of sleep from his Oura ring, already put "
        "the way a friend would say it: roughly how long he slept, when he fell "
        "asleep and woke up, whether he lay awake a long time, naps since, a "
        "one word rating, and how the night compares to his usual over the "
        "last two weeks. Call this when he asks how he slept, how long he "
        "slept, when he went to bed or woke up, or about his deep or REM sleep. "
        "Answer like a person: lead with about how long and from when to when, "
        "then only what stands out, such as a much later bedtime than usual, a "
        "short night, or a long stretch awake. A null comparison means nothing "
        "notable or not enough history, so leave it out. The numbers under "
        "if_he_asks_for_detail (score, efficiency, deep, REM, heart rate, HRV) "
        "are only for when he asks for them by name or asks why."
    ),
    input_schema=_NO_INPUT,
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_oura_sleep():
    window = _day_window()
    # Two weeks of nights, so the last one can be compared with his usual
    history = dict(window, start_date=(datetime.date.today()
                                       - datetime.timedelta(days=USUAL_NIGHTS + 1)).isoformat())
    summary = summarize_sleep(_get("sleep", **history), _get("daily_sleep", **window))
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
