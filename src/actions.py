import time
import threading
import requests
import sqlite3
from datetime import datetime
from config import DEFAULT_LOCATION, WEATHER_API_KEY, DB_PATH
from tools import Permission, tool

# Filled in by voice_main.py at startup so timer/reminder alerts can speak aloud.
# The server never sets this, so alerts from API requests silently log only.
_speak_fn = None

def set_speak_fn(fn):
    global _speak_fn
    _speak_fn = fn


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
        "Current outdoor conditions for a place, plus whether rain is starting "
        "or stopping in the next twelve hours. Call this whenever Lethanial "
        "asks about the weather, the temperature, whether it is going to rain, "
        "whether rain will stop, or whether he needs a jacket or an umbrella. "
        "Omit the location to use his default. "
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
                "description": "City name only, no state or country. The "
                               "geocoder works best that way. Omit for the "
                               "default location.",
            },
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
)
def get_weather_tool(location=None):
    return _fetch_weather(location)


def get_weather(location=None):
    """Legacy bracket action path. Deleted in Phase 2 along with the tag system.

    Kept short on purpose rather than left as the old four fact paragraph, so
    the verbosity fix lands now instead of waiting on NATIVE_TOOLS."""
    try:
        data = _fetch_weather(location)
        if "error" in data:
            return data["error"]

        line = (f"{data['location']}: {data['temp']} degrees, feels like "
                f"{data['feels_like']}, {data['condition']}")
        if data["precip"]:
            line += f", {data['precip']}"
        return line + "."
    except Exception as e:
        return f"Weather lookup failed: {e}"


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

    def timer_thread():
        time.sleep(seconds)
        print(f"\n*** TIMER DONE: {amount} {unit} ***")
        alert = f"[calmly] Lethanial, your {amount} {unit} timer is up."
        if _speak_fn:
            _speak_fn(alert)
        print("Listening for 'hey nova'...")

    threading.Thread(target=timer_thread, daemon=True).start()
    return f"Timer set for {amount} {unit} ({seconds} seconds)."


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
                    alert = f"[calmly] Lethanial, a reminder. {content}."
                    if _speak_fn:
                        _speak_fn(alert)
                    conn2 = sqlite3.connect(DB_PATH)
                    conn2.execute(
                        "UPDATE reminders SET completed = 1 WHERE content = ? AND due_at = ?",
                        (content, due_time)
                    )
                    conn2.commit()
                    conn2.close()
                    print("Listening for 'hey nova'...")

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


# ── Dispatcher ──

def execute_actions(actions):
    results = []
    for action in actions:
        atype = action["type"]
        params = action["params"]

        if atype == "weather":
            loc = params.get("location") or params.get("value")
            results.append({"type": "weather", "data": get_weather(loc)})

        elif atype == "timer":
            duration = params.get("duration") or params.get("value", "")
            results.append({"type": "timer", "data": set_timer(duration)})

        elif atype == "reminder":
            content  = params.get("content") or params.get("value", "")
            due_time = params.get("due", None)
            results.append({"type": "reminder", "data": set_reminder(content, due_time)})

        elif atype == "cancel_reminder":
            content = params.get("content") or params.get("value", "")
            results.append({"type": "cancel_reminder", "data": cancel_reminder(content)})

    return results
