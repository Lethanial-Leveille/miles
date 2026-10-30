"""What his iPhone's Health app reports, as a READ tool.

The Pi cannot read HealthKit. The data lives on the phone and only the Nova app
may ask for it, so the app sends a copy whenever it opens (POST /health/sync)
and this reads the copy. That makes every answer here as old as the last time he
opened the app, and the result says so: a step count with no age on it reads as
a live reading, and "you've only done 800 steps" at four in the afternoon from a
sync made at nine is a wrong answer delivered with confidence.

Overlap with Oura is deliberate and labelled. The ring also reports steps and
active calories, from the ring. These are the phone's own, and the tool says
"iPhone" in the fields so Nova does not read one as the other. What the phone has
that nothing else does is body weight over time, and the workouts Apple records,
like a run with its distance. Lifting stays in Hevy.

Private health data, so min_tier hokage, like Oura and Hevy.

Arithmetic happens here. The average, the weight change and the staleness are
computed in code and handed over as numbers with units in the field names.
"""

from datetime import datetime, timedelta

from database import health_days_since, health_workouts_since, last_health_sync
from tools import Permission, tool

_DEFAULT_DAYS = 7
_MAX_DAYS = 30

# The app syncs when it opens, so a few hours old is normal and not worth a
# caveat; past this, "today" figures are likely well behind.
_STALE_AFTER_HOURS = 6

# Enough runs to hear a pattern in, few enough for one spoken answer.
_MAX_WORKOUTS = 6

_KINDS = {
    "running": "run", "walking": "walk", "hiking": "hike", "cycling": "bike ride",
    "swimming": "swim", "strength": "strength session", "other": "workout",
}


def _round(value, places=1):
    return None if value is None else round(value, places)


def summarize_health(days, workouts, synced_at, now, window_days):
    """The answer for the model, from stored rows. Pure, so it can be tested
    against a fixed clock.

    `days` are health_days rows oldest first, `workouts` newest first."""
    today = now.date().isoformat()
    today_row = next((d for d in days if d["day"] == today), None)

    # Today is partial by definition, so it stays out of the average. Averaging
    # a half finished day in would drag the figure down every morning.
    finished = [d for d in days if d["day"] < today and d["steps"] is not None]
    average_steps = (round(sum(d["steps"] for d in finished) / len(finished))
                     if finished else None)

    weights = [d for d in days if d["weight_lb"] is not None]
    weight = None
    if weights:
        latest, earliest = weights[-1], weights[0]
        weight = {
            "latest_pounds": _round(latest["weight_lb"]),
            "weighed_on": latest["day"],
            "change_pounds_since_first_weigh_in_in_window": (
                _round(latest["weight_lb"] - earliest["weight_lb"])
                if earliest is not latest else None),
            "first_weigh_in_in_window": earliest["day"] if earliest is not latest else None,
        }

    age_hours = (now - datetime.fromisoformat(synced_at)).total_seconds() / 3600
    result = {
        "source": "the iPhone's Health app, not Oura",
        "last_synced_hours_ago": _round(age_hours),
        "may_be_out_of_date": age_hours > _STALE_AFTER_HOURS,
        "window_days": window_days,
        "iphone_steps_today_so_far": today_row["steps"] if today_row else None,
        "iphone_active_calories_today_so_far": (
            _round(today_row["active_kcal"], 0) if today_row else None),
        "iphone_average_steps_per_finished_day": average_steps,
        "finished_days_averaged": len(finished),
        "body_weight": weight,
        "workouts_newest_first": [
            {
                "when": datetime.fromisoformat(w["started_at"]).strftime("%A %B %-d"),
                "kind": _KINDS.get(w["kind"], w["kind"]),
                "minutes": round(w["minutes"]),
                "miles": _round(w["distance_mi"], 2),
                "active_calories": _round(w["kcal"], 0),
            }
            for w in workouts[:_MAX_WORKOUTS]
        ],
    }
    return result


@tool(
    name="get_health_summary",
    description=(
        "Lethanial's numbers from his iPhone's Health app: steps and active "
        "calories so far today, his average steps on finished days, his body "
        "weight and how it has changed, and workouts Apple recorded such as "
        "runs and walks with their distance. Call this when he asks about his "
        "weight, how far he ran, how much he walked, or his steps on his "
        "phone. These come from the phone and are only as fresh as the last "
        "time he opened the Nova app: if may_be_out_of_date is true, say when "
        "the numbers are from instead of presenting them as live. Sleep, "
        "readiness and heart rate are Oura's, and lifting is Hevy's, so use "
        "those tools for them."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "days": {"type": "integer", "minimum": 1, "maximum": _MAX_DAYS,
                     "description": f"How many days back to look. Default {_DEFAULT_DAYS}."},
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_health_summary(days=_DEFAULT_DAYS):
    days = max(1, min(int(days), _MAX_DAYS))
    synced_at = last_health_sync()
    if synced_at is None:
        return ("No health data yet. The Nova app sends it when it opens, and it "
                "has not sent any. He needs to open the app and allow Health access.")

    now = datetime.now()
    since = now - timedelta(days=days)
    return summarize_health(
        health_days_since(since.date().isoformat()),
        health_workouts_since(since.replace(hour=0, minute=0, second=0).isoformat()),
        synced_at, now, days)
