import time
import threading
import requests
import sqlite3
from datetime import datetime, timedelta
from config import DEFAULT_LOCATION, WEATHER_API_KEY, DB_PATH
from tools import Permission, tool
from parsing import words_for_number
from database import due_reminders, complete_reminder
import alerts


def _plural(amount, unit):
    """Singularize a unit for an amount of one.

    The tool hands units in already pluralized, because that is what the enum
    offers, so "1 minutes timer is up" was announced on every single one minute
    timer."""
    return unit[:-1] if amount == 1 and unit.endswith("s") else unit


def _spoken_amount(amount):
    """Spelled out when it can be, because Nova says this aloud."""
    try:
        return words_for_number(amount)
    except ValueError:
        return str(amount)


def _attributive(unit):
    """Singular, for a unit used as a modifier rather than as a quantity.

    "Timer set for five minutes" is a quantity and stays plural. "Your five
    minutes timer is up" is a modifier on `timer` and is simply wrong; English
    wants "your five minute timer". Two different grammatical roles, so two
    different helpers, rather than one that is right half the time."""
    return unit[:-1] if unit.endswith("s") else unit


# ── Weather ──

# Coordinates for a place name do not change, and geocoding was previously run
# on every single request. Caching for the life of the process removes one HTTP
# call per lookup, which roughly pays for the forecast call added below.
_GEOCODE_CACHE = {}

# OpenWeatherMap condition ids: 2xx thunderstorm, 3xx drizzle, 5xx rain,
# 6xx snow. 7xx is atmosphere (haze, fog), 800 clear, 80x cloud. So anything
# below 700 is falling out of the sky and anything at or above it is not.
_PRECIP_CEILING = 700

# Three hour steps, so four blocks is twelve hours. Enough to answer "is it
# going to rain later" without pretending to a precision the endpoint does not
# have: this cannot say "it stops in twenty minutes". Minute level precipitation
# is One Call 3.0, which is a separate signup with a card on file.
_FORECAST_BLOCKS = 4


def _geocode(location):
    """Resolve a place name to coordinates, cached per process."""
    if location in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[location]

    geo = requests.get(
        "http://api.openweathermap.org/geo/1.0/direct",
        params={"q": location, "limit": 1, "appid": WEATHER_API_KEY},
    )
    rows = geo.json()
    if not rows:
        return None

    coords = (rows[0]["lat"], rows[0]["lon"])
    _GEOCODE_CACHE[location] = coords
    return coords


def _is_precip(block):
    return block["weather"][0]["id"] < _PRECIP_CEILING


def _precip_outlook(lat, lon, raining_now):
    """When precipitation starts or stops in the next twelve hours.

    Returns a short phrase, or None when there is nothing to say. None is the
    common case and it matters: a tool that always returns a rain field invites
    Nova to mention rain on a clear day."""
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "appid": WEATHER_API_KEY,
                    "units": "imperial"},
        )
        blocks = r.json().get("list", [])[:_FORECAST_BLOCKS]
    except Exception:
        # A failed forecast must not fail the whole lookup. Current conditions
        # are still worth answering with.
        return None

    if not blocks:
        return None

    def when(block):
        return datetime.fromtimestamp(block["dt"]).strftime("%-I %p")

    if raining_now:
        for block in blocks:
            if not _is_precip(block):
                return f"easing off around {when(block)}"
        return "continuing through the next several hours"

    for block in blocks:
        if _is_precip(block):
            return f"rain likely around {when(block)}"
    return None


def fetch_weather(location=None):
    """Current conditions as structured data.

    Public, like set_timer and cancel_reminder, because local_intent calls it
    directly to answer a weather turn without Claude. The tool wrapper below is
    one of two callers, not the only one.

    Deliberately a dict rather than a sentence. The previous version returned a
    finished English paragraph carrying four facts, so Nova read the paragraph
    aloud, every time, whatever was asked. Telling her to be brief fought the
    data she was handed; changing the data is the lever that actually works."""
    location = location or DEFAULT_LOCATION

    coords = _geocode(location)
    if coords is None:
        return {"error": f"Could not find location: {location}"}

    lat, lon = coords
    r = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": WEATHER_API_KEY,
                "units": "imperial"},
    )
    d = r.json()

    raining_now = _is_precip(d)
    return {
        "location": location,
        "temp": round(d["main"]["temp"]),
        "feels_like": round(d["main"]["feels_like"]),
        "condition": d["weather"][0]["description"],
        "humidity": d["main"]["humidity"],
        "wind_mph": round(d["wind"]["speed"]),
        "precip": _precip_outlook(lat, lon, raining_now),
    }


@tool(
    name="get_weather",
    description=(
        "Current outdoor conditions, plus whether rain is starting or stopping "
        "in the next twelve hours. Call this whenever Lethanial asks about the "
        "weather, the temperature, whether it is going to rain, whether rain "
        "will stop, or whether he needs a jacket or an umbrella. "
        # Interpolated rather than written out, so the description and
        # DEFAULT_LOCATION can never disagree. Naming the city explicitly is
        # load bearing: Lethanial's memories mention more than one place he
        # lives, so "his home location" was genuinely ambiguous and the model
        # correctly refused to guess, asking him where every single time
        # instead of calling the tool.
        f"Location is optional. When he does not name one, omit the parameter "
        f"and {DEFAULT_LOCATION}, where he lives during the school year, is "
        f"used. Never ask him which place he means, even though his memories "
        f"mention more than one: an unqualified weather question always means "
        f"{DEFAULT_LOCATION}, and asking spends a whole turn of his to learn "
        f"something already known. "
        "Returns temperature and feels_like in Fahrenheit, a condition "
        "description, humidity as a percentage, wind_mph, and precip. "
        "Report only what was actually asked: temperature and the precip "
        "outlook are what he usually wants. Humidity and wind are there for "
        "when he asks for them specifically and should not be volunteered. "
        "A precip of null means nothing is coming, so say nothing about rain "
        "at all rather than saying it will not rain."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name only, no state or country, since "
                               "the geocoder works best that way. Omit this "
                               "entirely unless he named a different city; "
                               f"omitting it uses {DEFAULT_LOCATION}.",
            },
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
)
def get_weather_tool(location=None):
    return fetch_weather(location)


# ── Timer ──

def set_timer(duration_str):
    parts = duration_str.lower().strip().split()
    if len(parts) < 2:
        return "Could not parse timer duration."

    try:
        amount = int(parts[0])
    except ValueError:
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
            "forty five": 45, "fifty": 50, "sixty": 60,
        }
        amount = word_to_num.get(parts[0], 0)
        if amount == 0:
            return "Could not parse timer duration."

    unit = parts[1]
    if "hour" in unit:
        seconds = amount * 3600
    elif "min" in unit:
        seconds = amount * 60
    elif "sec" in unit:
        seconds = amount
    else:
        return f"Unknown time unit: {unit}"

    spoken_unit = _plural(amount, unit)

    # A row, fired by the poller, since Sep 16 2026. It was a thread sleeping in
    # whichever process set it, so a timer set from the app slept in
    # miles-server, whose alert queue nothing drains, and never went off; and a
    # restart dropped every running timer. The same bug reminders had until
    # Sep 6, with the same fix: the row is the only state.
    due = datetime.now() + timedelta(seconds=seconds)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO reminders (content, due_at, created_at, kind) VALUES (?, ?, ?, 'timer')",
        (f"{amount} {_attributive(unit)} timer", due.isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()
    return f"Timer set for {amount} {spoken_unit} ({seconds} seconds)."


def _timer_alert(content, late):
    """(text, summary) for a timer row, whose content reads like "10 minute timer".

    The amount is spelled out for speech, which is why it is parsed back rather
    than stored as a sentence."""
    amount, rest = content.split(" ", 1)
    spoken = f"{_spoken_amount(int(amount))} {rest}"
    if late:
        return (f"[calmly] Lethanial, your {spoken} went off while you were away.",
                f"the {content} went off while he was away")
    return (f"[calmly] Lethanial, your {spoken} is up.",
            f"the {content} just finished")


# ── Reminders ──

# How often the poller asks the table what is due. Twenty seconds was chosen
# for reminders, which are set to the minute. Timers moved into the same table
# on Sep 16 2026 and are set to the second, so a ten second timer could have
# rung thirty seconds in. Five keeps a timer within five seconds, and each pass
# is still one indexed query against a small local SQLite file.
REMINDER_POLL_S = 5

# Past this much lateness, the announcement says so. A reminder delivered four
# hours after it was due is still worth hearing, but presenting it as though it
# had just come due is a small lie that makes the clock look broken.
REMINDER_LATE_S = 3600


def set_reminder(content, due_time=None):
    """Save a reminder. Firing is the poller's job, not this function's.

    This used to spawn a threading.Thread that slept until the due time and
    fired from there. That made the thread the real state and the row merely a
    record of it: the row survived a restart and the thread did not, so every
    pending reminder was silently dropped on any deploy or crash, and
    Restart=always makes both routine. Nothing scanned the table at boot, so a
    reminder set for tomorrow morning simply never happened.

    It also meant a reminder set through the app fired inside the uvicorn
    process, where its alert queued into that process's alerts._pending and was
    delivered only if another chat message arrived inside the fifteen second
    fold window. Otherwise it was lost without even reaching alert_log.

    Now the row is the only state and poll_reminders is the only thing that
    fires, which fixes both: a restart re reads the table, and it does not
    matter which process wrote the row.

    A due time already in the past is stored and fires on the next pass rather
    than being quietly dropped. It is a bug when it happens, almost always the
    clock guidance in the prompt being ignored, and the whole argument in
    alerts.py is that a silent non delivery is the worst available outcome. So
    it announces, late, and says it is late."""
    if due_time:
        # Validated here so an unparseable string is refused at the point it can
        # still be corrected, rather than being stored and skipped forever by a
        # poller that cannot read it.
        try:
            datetime.fromisoformat(due_time)
        except (TypeError, ValueError):
            return (f"Could not read '{due_time}' as a date and time, so "
                    f"nothing was saved. Use YYYY-MM-DDTHH:MM:SS.")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (content, due_at, created_at) VALUES (?, ?, ?)",
        (content, due_time, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    if due_time and datetime.fromisoformat(due_time) <= datetime.now():
        return (f"Reminder saved: {content}. That time has already passed, so "
                f"it will announce now.")

    return f"Reminder saved: {content}" + (f" (due: {due_time})" if due_time else "")


def poll_reminders(now=None):
    """One pass over the table. Returns how many fired.

    Marks delivered BEFORE queueing the alert, and that order is deliberate.
    complete_reminder returns whether it actually changed a row, so the UPDATE
    doubles as a claim: two passes racing the same reminder cannot both win it.

    The two orderings fail differently. Firing first and completing second is at
    least once, and its failure mode is a reminder that announces every twenty
    seconds forever if the completion keeps failing, which is unusable. Claiming
    first is at most once, and its failure mode is losing one reminder if the
    process dies in the microseconds between the commit and the in memory
    append. The second failure is rarer and far less bad, so the loss window is
    accepted on purpose."""
    now = now or datetime.now()
    fired = 0

    for reminder_id, content, due_at, kind in due_reminders(now.isoformat()):
        if not complete_reminder(reminder_id):
            continue                    # another pass already claimed it

        try:
            late_seconds = (now - datetime.fromisoformat(due_at)).total_seconds()
        except (TypeError, ValueError):
            late_seconds = 0

        if kind == "timer":
            # Queued, never spoken from here, like every alert: the poller
            # cannot tell an open mic from an idle room. alerts.py explains.
            text, summary = _timer_alert(content, late_seconds > REMINDER_LATE_S)
            print(f"\n*** TIMER DONE: {content} ***", flush=True)
            alerts.fire(kind="timer", text=text, summary=summary)
            fired += 1
            continue

        if late_seconds > REMINDER_LATE_S:
            text = (f"[calmly] Lethanial, a reminder that came due while you "
                    f"were away. {content}.")
            summary = f"a reminder came due while he was away: {content}"
        else:
            text = f"[calmly] Lethanial, a reminder. {content}."
            summary = f"a reminder just came due: {content}"

        print(f"\n*** REMINDER: {content} ***", flush=True)
        alerts.fire(kind="reminder", text=text, summary=summary)
        fired += 1

    return fired


_poller_started = False


def start_reminder_poller(interval=REMINDER_POLL_S):
    """Start the one thread that fires reminders.

    Called from the voice loop only, never from the server. Both processes can
    create reminders, but exactly one may deliver them: a poller in each would
    race for the same rows, and the claim in poll_reminders would keep them
    correct while the alert still landed in whichever process won, which for the
    server is a queue nothing drains.

    Guarded against a second start rather than left to the caller, because two
    pollers in one process is a bug with no symptom other than reminders
    announcing twice."""
    global _poller_started
    if _poller_started:
        return False

    def loop():
        while True:
            try:
                poll_reminders()
            except Exception as exc:
                # A poller that dies takes every future reminder with it, and
                # silently, which is the failure this whole change exists to
                # remove. Log and keep the thread alive.
                print(f"Reminder poll failed ({type(exc).__name__}): {exc}",
                      flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    _poller_started = True
    return True


def cancel_reminder(content, kind=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if kind is None:
        c.execute(
            "DELETE FROM reminders WHERE content LIKE ? AND completed = 0",
            (f"%{content}%",)
        )
    else:
        c.execute(
            "DELETE FROM reminders WHERE content LIKE ? AND completed = 0 AND kind = ?",
            (f"%{content}%", kind)
        )
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        return f"Removed {deleted} reminder(s) matching '{content}'."
    return f"No active reminders found matching '{content}'."


# ── Tool registrations for the fire and forget actions ──
#
# All four carry returns_to_model=False. Nothing they return is worth a second
# Claude call: the confirmation Nova already said alongside the call is the
# answer, and a follow up round trip would add a second of latency to "set a
# timer for ten minutes" in exchange for rephrasing "Timer set."
#
# Each wraps the existing implementation rather than replacing it, so the
# parsing and threading that were already tested stay tested.

@tool(
    name="set_timer",
    description=(
        "Start a countdown timer that speaks aloud when it finishes. Call this "
        "when Lethanial asks to set a timer or to be told when some amount of "
        "time has passed. The timer announces itself, so the spoken "
        "confirmation is all that is needed from you."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "amount": {"type": "integer", "description": "How many units, e.g. 10"},
            "unit": {"type": "string", "enum": ["seconds", "minutes", "hours"]},
        },
        "required": ["amount", "unit"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
)
def set_timer_tool(amount, unit):
    # Sep 16 2026: asked to cancel a timer, with no tool that said it could,
    # Nova set a zero second timer instead, which rang at once. A timer of
    # nothing is never what he asked for, so it is refused out loud.
    if not isinstance(amount, int) or amount < 1:
        raise ValueError("a timer needs a length of at least one; to stop a "
                         "timer, use cancel_reminder")
    # Reuses the string parser rather than duplicating the threading, and a
    # structured amount plus unit always satisfies it, so the word number
    # fallback inside it is now dead weight the tag path still needs.
    return set_timer(f"{amount} {unit}")


@tool(
    name="set_reminder",
    description=(
        "Save a reminder, optionally with a time at which it speaks aloud. "
        "Call this when Lethanial asks to be reminded of something. Compute "
        "due from the clock supplied with his message, never from an example "
        "and never from a guess: a reminder dated in the past is saved and "
        "then never fires, which fails silently. Omit due when he gives no "
        "time, which saves the reminder without scheduling it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string",
                        "description": "What to remind him about, in his words"},
            "due": {"type": "string",
                    "description": "ISO 8601, YYYY-MM-DDTHH:MM:SS, computed "
                                   "from the supplied clock. A time in the "
                                   "past never fires."},
        },
        "required": ["content"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
)
def set_reminder_tool(content, due=None):
    return set_reminder(content, due)


@tool(
    name="cancel_reminder",
    description=(
        "Cancel running timers or saved reminders matching a phrase. Call "
        "this when Lethanial cancels, stops, removes, or says never mind about a "
        "timer or a reminder. A timer's saved text reads like '10 minute timer', "
        "so 'timer' matches every running timer and '10 minute' matches that "
        "one. For a reminder, match on its distinctive words rather than the "
        "whole sentence. Never set a new timer to cancel one."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string",
                        "description": "Words to match against saved reminders"},
        },
        "required": ["content"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
)
def cancel_reminder_tool(content):
    result = cancel_reminder(content)
    # Nothing matched is a failure, not a quiet success. Returned as a string it
    # reached the "Done." fallback; raised, it becomes an is_error result, which
    # always goes back to Nova so she can say what she could not find.
    if result.startswith("No active reminders"):
        raise LookupError(result)
    return result


@tool(
    name="ignore",
    description=(
        "Stay silent and end the turn. Call this when what you heard was not "
        "addressed to you at all: Lethanial talking to someone else in the "
        "room, a fragment of a conversation you are not part of, a television, "
        "or speech that only makes sense as part of an exchange you cannot "
        "see. The follow up window stays open after a turn, so it does pick up "
        "the room.\n\n"
        "Say NOTHING alongside this call. Not an explanation, not an offer to "
        "help, not a single word. Announcing that you are not part of a "
        "conversation is itself joining it, and it is worse than silence "
        "because he then has to wait through it.\n\n"
        "Writing a sentence like 'I'm not part of that conversation' WITHOUT "
        "calling this tool is the specific failure to avoid. If that sentence "
        "is what you were about to say, this tool is what you actually meant, "
        "and the sentence should not be said at all. There is no case where "
        "explaining your non participation out loud is the better answer.\n\n"
        "Do not use this merely because a request is unclear or you lack the "
        "information to answer. Asking him to repeat himself is right when he "
        "was talking to you. This is only for when he was not."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.CONTROL,
    returns_to_model=False,
)
def ignore_tool():
    # A state change, not work. brain.py reads the call itself; there is
    # nothing to execute and nothing to say.
    return ""


@tool(
    name="dismiss",
    description=(
        "End the conversation. Call this when Lethanial is signing off rather "
        "than asking for anything: thanks, that's all, I'm good, goodbye, "
        "alright I'm done, or anything that reads as closing rather than "
        "continuing. Judge the intent, not the words. Do not call it when a "
        "similar phrase sits inside a larger thought, such as later meaning "
        "afterward, or I'm good answering how he is. When in doubt do not call "
        "it: staying available costs him nothing and cutting him off mid "
        "thought does. Say a short natural goodbye alongside it, varied rather "
        "than a stock line."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.CONTROL,
    returns_to_model=False,
)
def dismiss_tool():
    """Executes nothing. The state transition happens in brain.py, which reads
    the call and exits the follow up loop.

    It is a tool rather than a bracket tag because it is a real state change,
    unlike an emotion cue, and because keeping it as a tag would have meant
    keeping the whole tag parser alive for exactly one case. As a tool it also
    lands in tool_call_log, which finally makes it measurable how often Nova
    ends a session while Lethanial is still talking."""
    return "dismissed"
