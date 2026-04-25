import time
import threading
import requests
import sqlite3
from datetime import datetime
from config import DEFAULT_LOCATION, WEATHER_API_KEY, DB_PATH

# Filled in by voice_main.py at startup so timer/reminder alerts can speak aloud.
# The server never sets this, so alerts from API requests silently log only.
_speak_fn = None

def set_speak_fn(fn):
    global _speak_fn
    _speak_fn = fn


# ── Weather ──

def get_weather(location=None):
    if not location:
        location = DEFAULT_LOCATION

    try:
        geo = requests.get(
            "http://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": WEATHER_API_KEY}
        )
        geo_data = geo.json()
        if not geo_data:
            return f"Could not find location: {location}"

        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

        weather = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "imperial"}
        )
        d = weather.json()

        temp       = d["main"]["temp"]
        feels_like = d["main"]["feels_like"]
        humidity   = d["main"]["humidity"]
        description = d["weather"][0]["description"]
        wind_speed = d["wind"]["speed"]

        return (
            f"Location: {location}. "
            f"Currently {description}, {temp} degrees F (feels like {feels_like} degrees F). "
            f"Humidity: {humidity} percent. Wind: {wind_speed} mph."
        )
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
