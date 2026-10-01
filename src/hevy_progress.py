"""Raise the weight in his Hevy routines when a session earned it.

He asked on Oct 1 2026 not to have to think about adding weight. get_session_plan
already said when a jump was earned, but only when he asked, and the routine in
Hevy kept the old weight, so the number in front of him at the gym was the one
he had outgrown. This writes the new weight into the routine itself, a few
minutes after he finishes, so the next session opens with it filled in.

It only ever raises. A lighter off day, a stall, a lift he skipped: none of
those change the routine. Hevy's own "update routine" at the end of a workout
copies what he did, lower weights included, and he keeps the routine as is.

Only the working sets' weight of a lift that earned the jump changes. Reps,
rep ranges, warm ups, notes and every other lift are sent back exactly as they
were read, because an update replaces the whole routine. The routine is backed
up to data/hevy_backups before every write.

Run by scripts/hevy_progress.py on a timer. Each workout is judged once:
hevy_progressions records it, raise or not.
"""

import datetime
import json
import os

import requests

import hevy_tools
from database import hevy_workouts_seen, record_hevy_progression

_BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "hevy_backups")

# Only workouts this recent are judged. The table starts empty, and without a
# window the first run would raise weights for sessions from months ago that
# the current routines were never built from.
_WINDOW = datetime.timedelta(days=3)

_SET_KEYS = ("type", "weight_kg", "reps", "distance_meters", "duration_seconds",
             "custom_metric", "rep_range")


def _working(sets):
    return [s for s in sets if s.get("type") != "warmup"]


def raises_for(workout, routine):
    """[(exercise, from_lb, to_lb, template_id)] this workout earned in this routine.

    A lift the routine is already at or past the new weight for is skipped,
    so a workout judged after someone raised it by hand changes nothing."""
    planned = {e["exercise_template_id"]: e for e in routine.get("exercises", [])}
    raises = []
    for e in workout.get("exercises", []):
        in_routine = planned.get(e["exercise_template_id"])
        if in_routine is None:
            continue
        jump = hevy_tools.earned_jump(_working(e.get("sets", [])), e["title"],
                                      in_routine.get("sets", []))
        if jump is None:
            continue
        now_lb = hevy_tools.pounds(hevy_tools.working_weight(_working(in_routine["sets"])))
        if now_lb is not None and now_lb >= jump[1]:
            continue
        raises.append((e["title"], jump[0], jump[1], e["exercise_template_id"]))
    return raises


def raised_routine(routine, raises):
    """The routine as the API takes it back, with only the raised weights changed.

    Hevy rejects rep_range: null, so the key is left out where it is empty."""
    new_kg = {tid: to_lb * hevy_tools._KG_PER_LB for _, _, to_lb, tid in raises}
    exercises = []
    for e in routine["exercises"]:
        sets = []
        for s in e["sets"]:
            one = {k: s.get(k) for k in _SET_KEYS if s.get(k) is not None or k != "rep_range"}
            if e["exercise_template_id"] in new_kg and s.get("type") != "warmup":
                one["weight_kg"] = new_kg[e["exercise_template_id"]]
            sets.append(one)
        exercises.append({"exercise_template_id": e["exercise_template_id"],
                          "superset_id": e.get("superset_id"),
                          "rest_seconds": e.get("rest_seconds"),
                          "notes": e.get("notes"), "sets": sets})
    return {"routine": {"title": routine["title"], "notes": routine.get("notes"),
                        "exercises": exercises}}


def _backup(routine):
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = os.path.join(_BACKUP_DIR, f"progress_{routine['id']}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(routine, f, indent=1)
    return path


def _put(routine, body):
    response = requests.put(f"{hevy_tools._API}/routines/{routine['id']}",
                            headers={"api-key": os.environ["HEVY_API_KEY"]},
                            json=body, timeout=hevy_tools._TIMEOUT_S)
    response.raise_for_status()


def run(now=None):
    """Judge every recent unjudged workout started from a routine, oldest first.
    Returns [(routine title, exercise, from_lb, to_lb)] for the log."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    seen = hevy_workouts_seen()
    workouts = hevy_tools._get("workouts", page=1, pageSize=hevy_tools._PAGE_SIZE).get("workouts", [])
    fresh = sorted((w for w in workouts
                    if w.get("routine_id") and w["id"] not in seen and w.get("end_time")
                    and now - hevy_tools._local(w["end_time"]) <= _WINDOW),
                   key=lambda w: w["start_time"])

    done = []
    for workout in fresh:
        # Read fresh for each workout, so a second session of the same routine
        # is judged against the weight the first one raised it to.
        routines = {r["id"]: r for r in hevy_tools._all_pages("routines", "routines",
                                                                hevy_tools._NAME_SEARCH_PAGES)}
        routine = routines.get(workout["routine_id"])
        if routine is None:
            continue
        raises = raises_for(workout, routine)
        if raises:
            _backup(routine)
            _put(routine, raised_routine(routine, raises))
        record_hevy_progression(workout, routine, [r[:3] for r in raises])
        done += [(routine["title"], *r[:3]) for r in raises]
    return done
