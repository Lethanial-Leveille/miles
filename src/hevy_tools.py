"""Hevy workout logs, as READ tools.

Lifting logs are private, so every tool here carries min_tier hokage, the same
as Oura.

Hevy stores every weight in kilograms and Lethanial logs in pounds, so his 225
pound RDL arrives as 102.05840462301894. Converted back and rounded to the half
pound it is exactly what he typed. Left in kilograms, Nova would read out a
number he never lifted.

Every set becomes words here, "6 reps at 185 pounds", for the reason the Oura
module gives: a number with no unit is a number the model will assign one to.
Three shapes in his log need care. Older sets carry 0 for distance and
duration where nothing was logged, so zero means absent. A weighted pull up or
dip stores the weight added to his body, not the weight moved. Dead hangs and
handstand holds are time, not reps.

Records and top sets are chosen in code, not by the model reading rows.

Failures raise. The executor in brain.py turns an exception into an is_error
result, so Nova says Hevy is unreachable instead of reading an HTTP body.
"""

import datetime
import os
import re

import requests

import config  # noqa: F401  loads .env, which holds HEVY_API_KEY
from database import hevy_raises_after
from tools import Permission, tool

_API = "https://api.hevyapp.com/v1"
_TIMEOUT_S = 10

# Hevy refuses a pageSize above 10.
_PAGE_SIZE = 10

# How far back a lift's name is searched for. 40 workouts was all of them on
# Sep 28 2026, about five months.
_NAME_SEARCH_PAGES = 4

# Enough sessions to hear a trend in, few enough for one spoken answer.
_HISTORY_SESSIONS = 6

# Epley overestimates badly past about a dozen reps, so higher rep sets are
# left out of the estimate rather than let a set of twenty win it.
_MAX_REPS_FOR_ESTIMATE = 12

_KG_PER_LB = 0.45359237


def _get(path, **params):
    # Read at call time, so a rotated key in .env takes effect on restart.
    key = os.environ.get("HEVY_API_KEY")
    if not key:
        raise RuntimeError("no HEVY_API_KEY in .env")
    response = requests.get(f"{_API}/{path}", headers={"api-key": key},
                            params=params, timeout=_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def pounds(kg):
    """Kilograms back to the pounds he logged, to the half pound. None for no
    weight, including the 0 older sets carry."""
    if not kg:
        return None
    lb = round(kg / _KG_PER_LB * 2) / 2
    return int(lb) if lb == int(lb) else lb


def seconds_words(seconds):
    """"1 minute 30 seconds". Hangs and holds are short enough that dropping
    the seconds, as the Oura durations do, would lose the whole set."""
    if not seconds:
        return None
    minutes, secs = divmod(int(seconds), 60)
    parts = []
    if minutes:
        parts.append(f"{minutes} minute{'' if minutes == 1 else 's'}")
    if secs:
        parts.append(f"{secs} second{'' if secs == 1 else 's'}")
    return " ".join(parts)


def _is_weighted_bodyweight(exercise_title):
    return "weighted" in exercise_title.lower()


def set_words(s, exercise_title):
    """One set as he would say it."""
    parts = []
    reps, rep_range = s.get("reps"), s.get("rep_range")
    if reps:
        parts.append(f"{reps} rep{'' if reps == 1 else 's'}")
    elif rep_range and rep_range.get("start") and rep_range.get("end"):
        parts.append(f"{rep_range['start']} to {rep_range['end']} reps")
    lb = pounds(s.get("weight_kg"))
    if lb is not None:
        if _is_weighted_bodyweight(exercise_title):
            parts.append(f"with {lb} pounds added")
        else:
            parts.append(f"at {lb} pounds")
    held = seconds_words(s.get("duration_seconds"))
    if held:
        parts.append(f"for {held}" if parts else held)
    if s.get("distance_meters"):
        parts.append(f"over {s['distance_meters']} meters")
    # A routine hold with no time set, like his handstand practice, is a
    # plan without a number, not an empty set.
    words = " ".join(parts) or "no set target"
    kind = s.get("type") or s.get("set_type")
    if kind and kind != "normal":
        words += f" ({kind})"
    return words


def _local(iso, tz=None):
    """Hevy's times are UTC. A session that ran past 8pm Eastern is filed
    under the next day in UTC, and "did I train Tuesday" means his Tuesday."""
    return datetime.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(tz)


def _when(iso, today, tz=None):
    day = _local(iso, tz).date()
    return {"date": day.isoformat(), "weekday": day.strftime("%A"),
            "days_ago": (today - day).days}


def summarize_workout(w, today, tz=None):
    start, end = _local(w["start_time"], tz), _local(w["end_time"], tz)
    out = {"name": w.get("title"), **_when(w["start_time"], today, tz),
           "duration": seconds_words((end - start).total_seconds())}
    if w.get("description"):
        out["his_notes"] = w["description"]
    exercises = []
    for e in w.get("exercises", []):
        entry = {"exercise": e["title"],
                 "sets": [set_words(s, e["title"]) for s in e.get("sets", [])]}
        if e.get("notes"):
            entry["his_notes"] = e["notes"]
        exercises.append(entry)
    out["exercises"] = exercises
    return out


def _words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def match_title(query, titles):
    """The titles, of lifts or routines, that `query` names. An exact title
    wins alone.
    Otherwise every word of the query must appear in the title, so "bench"
    finds all three bench presses and the caller asks which one he meant,
    rather than this picking one for him."""
    exact = [t for t in titles if t.lower() == query.strip().lower()]
    if exact:
        return exact[:1]
    wanted = _words(query)
    if not wanted:
        return []
    return sorted(t for t in titles if wanted <= _words(t))


def _strength(s):
    """What makes one set heavier than another: weight first, then reps, then
    time held. A 225 triple outranks a 185 set of eight."""
    return (s.get("weight_kg") or 0, s.get("reps") or 0, s.get("duration_seconds") or 0)


def _estimated_max_kg(s, exercise_title):
    # A weighted pull up stores only the plate on the belt. Epley on that alone
    # said 31 pounds, so these get no estimate rather than a guessed bodyweight.
    if _is_weighted_bodyweight(exercise_title):
        return None
    weight, reps = s.get("weight_kg"), s.get("reps")
    if not weight or not reps or reps > _MAX_REPS_FOR_ESTIMATE:
        return None
    return weight * (1 + reps / 30)


def summarize_history(title, sets, today, tz=None):
    """Per session top sets, newest first, and his best ever.

    Warm ups are left out of both. A warm up is never the top set of a
    session, and counting its reps would make an easy day look like work."""
    working = [s for s in sets if s.get("set_type", "normal") != "warmup"]
    if not working:
        return None

    sessions = {}
    for s in working:
        sessions.setdefault(s["workout_id"], []).append(s)
    ordered = sorted(sessions.values(), key=lambda ss: ss[0]["workout_start_time"], reverse=True)

    recent = []
    for ss in ordered[:_HISTORY_SESSIONS]:
        top = max(ss, key=_strength)
        recent.append({**_when(top["workout_start_time"], today, tz),
                       "workout": top.get("workout_title"),
                       "working_sets": len(ss),
                       "top_set": set_words(top, title)})

    best = max(working, key=_strength)
    out = {"exercise": title,
           "sessions_logged": len(sessions),
           "recent_sessions_newest_first": recent,
           "best_set_ever": {**_when(best["workout_start_time"], today, tz),
                                 "set": set_words(best, title)}}

    estimates = [(m, s) for s in working if (m := _estimated_max_kg(s, title))]
    if estimates:
        est_kg, source = max(estimates, key=lambda pair: pair[0])
        out["estimated_one_rep_max"] = {
            "pounds": pounds(est_kg),
            "from_set": set_words(source, title),
            **_when(source["workout_start_time"], today, tz),
            "how": "Epley formula, an estimate, not a lift he has done",
        }
    if len(sessions) > _HISTORY_SESSIONS:
        out["older_sessions_not_listed"] = len(sessions) - _HISTORY_SESSIONS
    return out


def summarize_routines(routines, workouts, today, tz=None):
    """Each routine's planned sets, and when he last did it.

    Workouts name the routine they were started from, so the last done date is
    what answers "is today upper or lower", not any schedule written down."""
    last_done = {}
    for w in workouts:
        rid = w.get("routine_id")
        if rid and (rid not in last_done or w["start_time"] > last_done[rid]):
            last_done[rid] = w["start_time"]

    out = []
    for r in routines:
        entry = {"name": r.get("title")}
        if r["id"] in last_done:
            entry["last_done"] = _when(last_done[r["id"]], today, tz)
        else:
            entry["last_done"] = f"not in his last {len(workouts)} workouts"
        entry["exercises"] = [
            {"exercise": e["title"],
             "planned_sets": [set_words(s, e["title"]) for s in e.get("sets", [])],
             **({"his_notes": e["notes"]} if e.get("notes") else {})}
            for e in r.get("exercises", [])
        ]
        out.append(entry)
    return out


def working_weight(sets):
    """The weight he did the most sets at, the heavier on a tie. Not the
    heaviest set: on Sep 22 2026 he benched four sets at 185 and one single at
    225, and the heaviest set made the target "one more rep at 225" and would
    have made his next normal session read as a regression."""
    counts = {}
    for s in sets:
        w = s.get("weight_kg") or 0
        counts[w] = counts.get(w, 0) + 1
    return max(counts, key=lambda w: (counts[w], w))


def _session_score(sets):
    """How good one session of a lift was: the working weight, then the reps
    done at it across every set, then the longest hold. Total reps rather than
    the best set, because double progression moves by adding a rep to any set,
    and 185 for 6, 6, 6 beats 185 for 6, 6, 5."""
    weight = working_weight(sets)
    reps = sum((s.get("reps") or 0) for s in sets if (s.get("weight_kg") or 0) == weight)
    held = max((s.get("duration_seconds") or 0) for s in sets)
    return (weight, reps, held)


# The jumps he agreed on Sep 28 2026, in pounds. Squat and deadlift, and
# their front squat and RDL variants, go up by ten; every other barbell lift,
# every dumbbell and the belt go up by five. Machines and cables have no fixed
# jump, because every stack is different, so they say the next pin instead.
_TEN_POUND_LIFTS = {"Squat (Barbell)", "Front Squat", "Deadlift (Barbell)",
                    "Romanian Deadlift (Barbell)"}


def _on_a_stack(exercise_title):
    return "(Machine)" in exercise_title or "Cable" in exercise_title or exercise_title == "Face Pull"


def _increment(exercise_title):
    """Pounds to add. Machines and cables go up ten, his choice on Oct 1 2026:
    every stack is different, so he takes the nearest pin and the log keeps
    what he actually used."""
    if exercise_title in _TEN_POUND_LIFTS or _on_a_stack(exercise_title):
        return 10
    return 5


def _load_words(lb, exercise_title):
    return f"{lb} pounds added" if _is_weighted_bodyweight(exercise_title) else f"{lb} pounds"


def _planned(routine_sets):
    """The routine's working sets for one lift: how many, and their rep range
    if it has one. Warm ups are not what the range is judged on."""
    working = [s for s in routine_sets if s.get("type") != "warmup"]
    ranged = next((s["rep_range"] for s in working
                   if (s.get("rep_range") or {}).get("start") and s["rep_range"].get("end")), None)
    return len(working), ranged


def _earned(last_sets, weight, count, high):
    """Every working set at the working weight reached the top of the range,
    and there were at least as many as the routine plans."""
    at_weight = [s for s in last_sets if (s.get("weight_kg") or 0) == weight and s.get("reps")]
    return len(at_weight) >= max(count, 1) and all(s["reps"] >= high for s in at_weight)


def earned_jump(last_sets, exercise_title, routine_sets=()):
    """(from, to) in pounds when the last session earned more weight, else None.

    The one decision the spoken target and the automatic raise in
    hevy_progress.py both read, so what Nova says and what the routine holds
    cannot disagree. Bodyweight lifts have nothing to raise and return None."""
    count, rep_range = _planned(routine_sets)
    weight = working_weight(last_sets)
    lb = pounds(weight)
    if not rep_range or not lb or not _earned(last_sets, weight, count, rep_range["end"]):
        return None
    return lb, lb + _increment(exercise_title)


def target(last_sets, exercise_title, routine_sets=()):
    """His double progression: add reps within the range until every working
    set reaches the top, then add weight and drop to the bottom.

    Every working set means the routine's full count at the working weight,
    so two sets at the top of a three set lift has not earned the jump. With no
    range in the routine, a loaded lift can only be told to add a rep, and a
    drill with fixed reps, like three muscle up attempts, is told to hit the
    plan, because "one more rep" on a single attempt is not progress."""
    count, rep_range = _planned(routine_sets)
    weight = working_weight(last_sets)
    lb = pounds(weight)

    if not rep_range:
        if not weight and any(s.get("reps") for s in routine_sets):
            return "Hit the planned sets"
        if any(s.get("duration_seconds") for s in last_sets) and not any(s.get("reps") for s in last_sets):
            return "Hold any set longer than last time"
        if lb:
            return f"Stay at {_load_words(lb, exercise_title)} and add one rep to any set"
        return "Add one rep to any set"

    low, high = rep_range["start"], rep_range["end"]

    if not lb:
        if _earned(last_sets, weight, count, high):
            return f"Every set hit {high} last time: make it harder, with added weight or a harder version"
        return f"Add one rep to any set; when every set hits {high}, make it harder"
    jump = earned_jump(last_sets, exercise_title, routine_sets)
    if jump:
        up = _load_words(jump[1], exercise_title)
        if _on_a_stack(exercise_title):
            up += ", or the nearest pin,"
        return f"Go up to {up} and start at {low} reps"
    return f"Stay at {_load_words(lb, exercise_title)} and add one rep to any set; go up when every set hits {high}"


def lift_sessions(workouts):
    """Every lift's sessions, newest first, keyed by template id, warm ups left
    out. A lift done twice in one workout counts as one session."""
    by_lift = {}
    for w in sorted(workouts, key=lambda w: w["start_time"], reverse=True):
        for e in w.get("exercises", []):
            sets = [s for s in e.get("sets", []) if s.get("type") != "warmup"]
            if not sets:
                continue
            sessions = by_lift.setdefault(e["exercise_template_id"], [])
            if sessions and sessions[-1]["start_time"] == w["start_time"]:
                sessions[-1]["sets"].extend(sets)
            else:
                sessions.append({"start_time": w["start_time"], "sets": sets})
    return by_lift


def plan_session(routine, sessions_by_lift, today, tz=None, raised_after=lambda start: {}):
    """Each lift in the routine: what he did last time, what to aim for, and
    whether it has stalled. The target reads the routine's rep range, so it
    can say when to add weight.

    Stalled means two sessions running without beating the one before, the
    deload trigger his programme names. It is a flag, not advice to deload:
    whether a stall means rest or a bad week is his call."""
    lifts = []
    for e in routine.get("exercises", []):
        title = e["title"]
        entry = {"exercise": title,
                 "planned_sets": [set_words(s, title) for s in e.get("sets", [])]}
        past = sessions_by_lift.get(e["exercise_template_id"], [])
        if not past:
            entry["last_time"] = "not in his recent logged workouts"
            lifts.append(entry)
            continue

        last = past[0]
        entry["last_time"] = {**_when(last["start_time"], today, tz),
                              "sets": [set_words(s, title) for s in last["sets"]]}
        entry["target"] = target(last["sets"], title, e.get("sets", []))
        # Said so she can tell him the routine moved, rather than him finding a
        # new number at the gym; hevy_progress.py does the raising.
        raised = raised_after(last["start_time"]).get(title)
        if raised:
            entry["raised_in_routine_after_last_time"] = f"{raised[0]:g} to {raised[1]:g} pounds"
        if len(past) >= 3:
            s0, s1, s2 = (_session_score(p["sets"]) for p in past[:3])
            entry["stalled"] = s0 <= s1 and s1 <= s2
        else:
            entry["stalled"] = "too few sessions to tell"
        lifts.append(entry)
    return {"routine": routine.get("title"), "lifts": lifts}


def _all_pages(path, key, max_pages):
    items = []
    for page in range(1, max_pages + 1):
        body = _get(path, page=page, pageSize=_PAGE_SIZE)
        items.extend(body.get(key, []))
        if page >= body.get("page_count", 0):
            break
    return items


@tool(
    name="get_recent_workouts",
    description=(
        "Lethanial's most recent workouts from Hevy, newest first: the date and "
        "weekday, how long it took, and every set of every exercise with reps and "
        "pounds. Call this when he asks what he did at the gym, how a workout "
        "went, when he last trained, or whether a session happened on a given "
        "day. He logs a session the moment he finishes it, so call this every "
        "time he asks, even if you said what his last workout was earlier "
        "today: that answer was true then and may not be now."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": _PAGE_SIZE,
                      "description": "How many workouts, newest first. Default 3."},
        },
        "required": [],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_recent_workouts(count=3):
    count = max(1, min(int(count), _PAGE_SIZE))
    workouts = _get("workouts", page=1, pageSize=count).get("workouts", [])
    if not workouts:
        return "No workouts logged in Hevy."
    today = datetime.date.today()
    return [summarize_workout(w, today) for w in workouts]


@tool(
    name="get_lift_history",
    description=(
        "One exercise across Lethanial's Hevy workouts: the top set of each "
        "recent session, his heaviest set ever, and an estimated one rep max. "
        "Call this when he asks about progress, a record, or what he lifted last "
        "time on a lift. Pass the exercise as Hevy names it, like 'Bench Press "
        "(Barbell)'. If several lifts match, ask him which one he means."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "exercise": {"type": "string",
                         "description": "The exercise name, as in his workouts or routines."},
        },
        "required": ["exercise"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_lift_history(exercise):
    # The history endpoint wants a template id, and names only live on the
    # workouts, so the name is looked up in his own log. Searching the 451
    # templates instead would find lifts he has never done.
    ids = {}
    for w in _all_pages("workouts", "workouts", _NAME_SEARCH_PAGES):
        for e in w.get("exercises", []):
            ids.setdefault(e["title"], e["exercise_template_id"])

    matches = match_title(exercise, ids)
    if not matches:
        return f"No exercise called {exercise} in his logged workouts."
    if len(matches) > 1:
        return f"Several logged exercises match {exercise}: {', '.join(matches)}. Ask which one."

    title = matches[0]
    sets = _get(f"exercise_history/{ids[title]}").get("exercise_history", [])
    return summarize_history(title, sets, datetime.date.today()) or f"No working sets of {title} logged."


@tool(
    name="get_routines",
    description=(
        "Lethanial's saved Hevy routines: the planned sets of each, and the date "
        "he last did it. Call this when he asks what is in a routine, what he "
        "should lift today, whether today is upper or lower, or which routine "
        "is next."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_routines():
    routines = _all_pages("routines", "routines", _NAME_SEARCH_PAGES)
    if not routines:
        return "No routines saved in Hevy."
    workouts = _get("workouts", page=1, pageSize=_PAGE_SIZE).get("workouts", [])
    return summarize_routines(routines, workouts, datetime.date.today())


@tool(
    name="get_session_plan",
    description=(
        "What Lethanial should aim for on each lift of one Hevy routine: his "
        "last session of each lift, a target to beat it, and whether the lift "
        "has stalled for two sessions running, which his programme treats as "
        "the sign to deload. Call this when he asks what to lift today, what "
        "weight to use, whether to add weight or reps, or whether he is "
        "stalling. Pass the routine name as in his routines, like 'Upper "
        "Strength'. If several routines match, ask him which one. Each target "
        "already decides between adding a rep and adding weight; say it as "
        "given. A stall is a flag, and whether to deload is his call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "routine": {"type": "string",
                        "description": "The routine name, like 'Upper (A)' or 'Lower (B)'."},
        },
        "required": ["routine"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_session_plan(routine):
    routines = {r["title"]: r for r in _all_pages("routines", "routines", _NAME_SEARCH_PAGES)}
    matches = match_title(routine, routines)
    if not matches:
        return f"No routine called {routine}. His routines: {', '.join(routines) or 'none'}."
    if len(matches) > 1:
        return f"Several routines match {routine}: {', '.join(matches)}. Ask which one."

    workouts = _all_pages("workouts", "workouts", _NAME_SEARCH_PAGES)
    return plan_session(routines[matches[0]], lift_sessions(workouts), datetime.date.today(),
                        raised_after=hevy_raises_after)
