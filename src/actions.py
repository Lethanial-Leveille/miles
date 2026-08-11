import time
import threading
import requests
import sqlite3
from datetime import datetime
from config import DEFAULT_LOCATION, WEATHER_API_KEY, DB_PATH
from tools import Permission, tool
import alerts


def _plural(amount, unit):
    """Singularize a unit for an amount of one.

    The tool hands units in already pluralized, because that is what the enum
    offers, so "1 minutes timer is up" was announced on every single one minute
    timer."""
    return unit[:-1] if amount == 1 and unit.endswith("s") else unit


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


def _fetch_weather(location=None):
    """Current conditions as structured data.

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
    return _fetch_weather(location)


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

    def timer_thread():
        time.sleep(seconds)
        print(f"\n*** TIMER DONE: {amount} {spoken_unit} ***")
        # Queued, never spoken from here. A background thread cannot tell an
        # open mic from an idle room, and this used to fire straight into a
        # question being asked. alerts.py explains the mechanism.
        alerts.fire(
            kind="timer",
            text=f"[calmly] Lethanial, your {amount} {spoken_unit} timer is up.",
            summary=f"the {amount} {spoken_unit} timer just finished",
        )

    threading.Thread(target=timer_thread, daemon=True).start()
    return f"Timer set for {amount} {spoken_unit} ({seconds} seconds)."


# ── Reminders ──

def set_reminder(content, due_time=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO reminders (content, due_at, created_at) VALUES (?, ?, ?)",
        (content, due_time, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    if due_time:
        try:
            due_dt = datetime.fromisoformat(due_time)
            delay  = (due_dt - datetime.now()).total_seconds()
            if delay > 0:
                def reminder_thread():
                    time.sleep(delay)
                    print(f"\n*** REMINDER: {content} ***")
                    alerts.fire(
                        kind="reminder",
                        text=f"[calmly] Lethanial, a reminder. {content}.",
                        summary=f"a reminder just came due: {content}",
                    )
                    conn2 = sqlite3.connect(DB_PATH)
                    conn2.execute(
                        "UPDATE reminders SET completed = 1 WHERE content = ? AND due_at = ?",
                        (content, due_time)
                    )
                    conn2.commit()
                    conn2.close()

                threading.Thread(target=reminder_thread, daemon=True).start()
            else:
                return "That time has already passed. Reminder saved but won't trigger."
        except Exception:
            pass

    return f"Reminder saved: {content}" + (f" (due: {due_time})" if due_time else "")


def cancel_reminder(content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "DELETE FROM reminders WHERE content LIKE ? AND completed = 0",
        (f"%{content}%",)
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
        "Delete saved reminders matching a phrase. Call this when Lethanial "
        "cancels, removes, or says never mind about a reminder. Match on the "
        "distinctive words of the reminder rather than the whole sentence."
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
    return cancel_reminder(content)


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
