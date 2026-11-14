"""Change his Hevy routines by voice, asked first.

He asked for it on Nov 3 2026, when Nova said "Let me update Lower Strength"
with no tool behind it and then admitted she could not. Until then routines
were edited only in the Hevy app, and the reasons that kept it that way still
hold, so each one is answered in code here:

- Hevy replaces a routine whole on update and has no undo. Every write is
  backed up to data/hevy_backups first, and only the one exercise asked about
  changes: everything else is sent back exactly as it was read.
- An exercise name has to be one of Hevy's templates, about 450 of them.
  Names are matched in code, and when several fit, Nova asks which one rather
  than this picking for him.
- Nothing is written on the turn it is asked for. edit_routine stages the
  change through pending_action and returns a question built here; only a yes
  on his next turn runs it. On confirmation the routine is read again and the
  same change applied to that fresh copy, so an edit he made in the app in
  between is kept, and several changes confirmed together build on each other.
- His phone keeps its own copy of each routine and can write it back when he
  saves a workout (Oct 2 2026). That cannot be prevented from here, so every
  confirmed change ends by telling him to refresh routines in the app.

Read only lifting stays in hevy_tools.py. The weight raises after a session
stay in hevy_progress.py; this is only what he asks for.
"""

import copy
import datetime
import json
import os
import re
import time

import requests

import hevy_tools
import pending_action
from tools import Permission, ToolNotice, tool

_BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "hevy_backups")

# The template list allows 100 a page, where workouts and routines stop at 10.
_TEMPLATE_PAGE_SIZE = 100

# Templates only change when he makes a custom exercise, so a day is plenty,
# and it keeps a proposal from costing five requests every time.
_TEMPLATE_TTL_S = 24 * 3600

# More choices than this is not a question anyone can answer by ear.
_MAX_CHOICES = 6

_MAX_SETS = 10

_SET_KEYS = ("type", "weight_kg", "reps", "distance_meters", "duration_seconds",
             "custom_metric", "rep_range")

# What each kind of Hevy exercise counts. A swap between kinds cannot carry the
# old targets over: three sets of eight reps mean nothing on a timed hold.
_COUNTS_REPS = {"weight_reps", "reps_only", "bodyweight_weighted", "bodyweight_assisted"}
_COUNTS_TIME = {"duration", "distance_duration"}

_templates_cache = (0.0, None)


# ── Reading Hevy ──

def _all_templates():
    global _templates_cache
    loaded, cached = _templates_cache
    if cached is not None and time.monotonic() - loaded < _TEMPLATE_TTL_S:
        return cached
    templates, page = [], 1
    while True:
        body = hevy_tools._get("exercise_templates", page=page, pageSize=_TEMPLATE_PAGE_SIZE)
        templates.extend(body.get("exercise_templates", []))
        if page >= body.get("page_count", 0):
            break
        page += 1
    _templates_cache = (time.monotonic(), templates)
    return templates


def _routines():
    return hevy_tools._all_pages("routines", "routines", hevy_tools._NAME_SEARCH_PAGES)


def _backup(routine):
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    path = os.path.join(_BACKUP_DIR, f"edit_{routine['id']}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(routine, f, indent=1)
    return path


def _put(routine_id, body):
    response = requests.put(f"{hevy_tools._API}/routines/{routine_id}",
                            headers={"api-key": os.environ["HEVY_API_KEY"]},
                            json=body, timeout=hevy_tools._TIMEOUT_S)
    response.raise_for_status()


# ── Names ──

def _stem(word):
    # "pushups" is how he says it and "Pike Pushup" is how Hevy spells it,
    # and "push ups" needs "ups" made "up" too. "press" keeps its s.
    return word[:-1] if len(word) > 2 and word.endswith("s") and not word.endswith("ss") else word


def _key_words(text):
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def _squashed(text):
    # "push ups", "pushups" and "Push Up" all become "pushup".
    return "".join(_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()))


def match_names(query, titles):
    """The titles `query` names. An exact title wins alone, then a title with
    exactly the query's words. Otherwise every title holding all of its words,
    or its letters run together, so the caller can ask which one."""
    wanted = query.strip().lower()
    exact = [t for t in titles if t.lower() == wanted]
    if exact:
        return exact[:1]
    words, squashed = _key_words(query), _squashed(query)
    if not words:
        return []
    found = sorted({t for t in titles if words <= _key_words(t) or squashed in _squashed(t)})
    same = [t for t in found if _key_words(t) == words]
    return same if len(same) == 1 else found


def _one(query, titles, what):
    matches = match_names(query, titles)
    if not matches:
        raise ToolNotice(f"There is no {what} called {query}. Ask him what it is called.")
    if len(matches) > 1:
        shown = matches[:_MAX_CHOICES]
        more = len(matches) - len(shown)
        tail = f", and {more} more" if more else ""
        raise ToolNotice(f"Several {what}s match {query}: {', '.join(shown)}{tail}. "
                         f"Ask him which one he means. Nothing was changed.")
    return matches[0]


# ── The change itself ──

def _kind(template_type):
    if template_type in _COUNTS_TIME:
        return "time"
    if template_type in _COUNTS_REPS:
        return "reps"
    return "other"


def _shaped(sets, spec, kind, fresh_targets=False, require_targets=False):
    """The sets with the asked for count and targets, warm ups untouched.

    fresh_targets clears what the sets counted before, for a swap to a lift
    of another kind, where the old reps or seconds no longer mean anything.
    require_targets refuses sets with nothing to aim for, which only matters
    for sets made here: an existing hold with no time set stays as he built it."""
    warm = [dict(s) for s in sets if s.get("type") == "warmup"]
    work = [dict(s) for s in sets if s.get("type") != "warmup"] or [{"type": "normal"}]
    if fresh_targets:
        for s in work:
            for key in ("reps", "rep_range", "duration_seconds", "distance_meters"):
                s.pop(key, None)
    if spec.get("sets") is not None:
        count = spec["sets"]
        work = work[:count] + [dict(work[-1]) for _ in range(count - len(work))]
    for s in work:
        s.pop("index", None)
        if spec.get("reps") is not None:
            s["reps"] = spec["reps"]
            s.pop("rep_range", None)
        if spec.get("rep_min") is not None:
            s["rep_range"] = {"start": spec["rep_min"], "end": spec["rep_max"]}
            s["reps"] = None
        if spec.get("weight_lb") is not None:
            s["weight_kg"] = spec["weight_lb"] * hevy_tools._KG_PER_LB if spec["weight_lb"] else None
        if spec.get("seconds") is not None:
            s["duration_seconds"] = spec["seconds"]
        s.setdefault("type", "normal")
    if require_targets and kind == "reps" and not any(s.get("reps") or s.get("rep_range") for s in work):
        raise ToolNotice("That lift counts reps. Ask him how many reps, or what rep "
                         "range, before changing it. Nothing was changed.")
    if require_targets and kind == "time" and not any(s.get("duration_seconds") for s in work):
        raise ToolNotice("That exercise is timed. Ask him how many seconds a set "
                         "before changing it. Nothing was changed.")
    return warm + work


def _targets(exercise):
    """"3 sets of 5 to 8 reps at 135 pounds", from the working sets."""
    title = exercise["title"]
    work = [s for s in exercise["sets"] if s.get("type") != "warmup"]
    said = [hevy_tools.set_words(s, title) for s in work]
    n = len(work)
    sets = f"{n} set{'' if n == 1 else 's'}"
    if len(set(said)) == 1:
        return f"{sets} with no set target" if said[0] == "no set target" else f"{sets} of {said[0]}"
    return f"{sets}: " + ", ".join(said)


def _rest_words(seconds):
    return f", resting {hevy_tools.seconds_words(seconds)}" if seconds else ""


def apply(routine, spec):
    """(exercises, question) for this change on this routine. Pure, so the
    proposal and the confirmation run exactly the same change, the second on a
    copy read fresh from Hevy."""
    exercises = copy.deepcopy(routine.get("exercises", []))
    title = routine["title"]
    change = spec["change"]
    titles = [e["title"] for e in exercises]

    def where(name):
        if name not in titles:
            raise ToolNotice(f"{name} is no longer in {title}, so nothing was changed.")
        return titles.index(name)

    if change == "remove":
        if len(exercises) == 1:
            raise ToolNotice(f"{spec['exercise']} is the only exercise in {title}. A "
                             f"routine needs at least one, so nothing was changed.")
        exercises.pop(where(spec["exercise"]))
        return exercises, f"Remove {spec['exercise']} from {title}"

    if change == "swap":
        i = where(spec["exercise"])
        old = exercises[i]
        new_kind = _kind(spec["new_type"])
        moved_kinds = new_kind != _kind(spec["old_type"])
        # A different lift's weight means nothing on this one, so it goes
        # unless he named a new one.
        swap_spec = dict(spec)
        if swap_spec.get("weight_lb") is None:
            swap_spec["weight_lb"] = 0
        exercises[i] = {**old, "title": spec["new_title"],
                        "exercise_template_id": spec["new_template_id"],
                        "sets": _shaped(old["sets"], swap_spec, new_kind, fresh_targets=moved_kinds,
                                        require_targets=moved_kinds)}
        if spec.get("rest_seconds") is not None:
            exercises[i]["rest_seconds"] = spec["rest_seconds"]
        return exercises, (f"In {title}, swap {spec['exercise']} for {spec['new_title']}, "
                           f"{_targets(exercises[i])}{_rest_words(exercises[i].get('rest_seconds'))}")

    if change == "add":
        if spec["new_title"] in titles:
            raise ToolNotice(f"{spec['new_title']} is already in {title}. Ask whether he "
                             f"wants to change its sets instead. Nothing was changed.")
        new = {"title": spec["new_title"], "exercise_template_id": spec["new_template_id"],
               "superset_id": None, "rest_seconds": spec.get("rest_seconds"), "notes": None,
               "sets": _shaped([], {"sets": 3, **{k: v for k, v in spec.items() if v is not None}},
                               _kind(spec["new_type"]), require_targets=True)}
        at = where(spec["exercise"]) + 1 if spec.get("exercise") else len(exercises)
        exercises.insert(at, new)
        after = f" after {spec['exercise']}" if spec.get("exercise") else ""
        return exercises, (f"Add {spec['new_title']} to {title}{after}, {_targets(new)}"
                           f"{_rest_words(new.get('rest_seconds'))}")

    # adjust
    i = where(spec["exercise"])
    lift = exercises[i]
    kind = _kind(spec.get("old_type"))
    lift["sets"] = _shaped(lift["sets"], spec, kind)
    if spec.get("rest_seconds") is not None:
        lift["rest_seconds"] = spec["rest_seconds"]
    return exercises, (f"Change {spec['exercise']} in {title} to {_targets(lift)}"
                       f"{_rest_words(lift.get('rest_seconds'))}")


def put_body(routine, exercises):
    """The routine as the API takes it back. Hevy rejects rep_range: null, so
    the key is left out where it is empty, as in hevy_progress."""
    out = []
    for e in exercises:
        sets = [{k: s.get(k) for k in _SET_KEYS if s.get(k) is not None or k != "rep_range"}
                for s in e["sets"]]
        out.append({"exercise_template_id": e["exercise_template_id"],
                    "superset_id": e.get("superset_id"), "rest_seconds": e.get("rest_seconds"),
                    "notes": e.get("notes"), "sets": sets})
    return {"routine": {"title": routine["title"], "notes": routine.get("notes"),
                        "exercises": out}}


def _run(routine_id, spec, question):
    """What a yes runs: read the routine again, apply the same change, back up, write."""
    fresh = next((r for r in _routines() if r["id"] == routine_id), None)
    if fresh is None:
        return "Failed, not done: that routine is no longer in Hevy."
    try:
        exercises, _ = apply(fresh, spec)
    except ToolNotice as notice:
        return f"Failed, not done: {notice}"
    _backup(fresh)
    _put(routine_id, put_body(fresh, exercises))
    return (f"Done: {question[0].lower() + question[1:]}. Tell him to refresh his "
            f"routines in the Hevy app before his next session, because the app's old "
            f"copy can undo this when he saves a workout.")


def _ask(question):
    """As calendar_tools._ask: the question is built here and said word for word."""
    return (f"Nothing has changed yet. Ask Lethanial exactly this, with nothing "
            f"added before or after: {question} Then end your turn. When he "
            f"answers, call confirm_pending_action. If you proposed several changes "
            f"this turn, this question covers all of them; ask only the one from "
            f"your last result.")


def _checked(spec):
    """Refuse numbers that cannot be meant before anything is looked up."""
    if spec.get("sets") is not None and not 1 <= spec["sets"] <= _MAX_SETS:
        raise ToolNotice(f"Sets must be between 1 and {_MAX_SETS}; to drop the lift, "
                         f"remove it. Nothing was changed.")
    if (spec.get("rep_min") is None) != (spec.get("rep_max") is None):
        raise ToolNotice("A rep range needs both ends, like 6 to 8. Nothing was changed.")
    if spec.get("rep_min") is not None and not 1 <= spec["rep_min"] <= spec["rep_max"]:
        raise ToolNotice("The rep range is backwards or below one. Nothing was changed.")
    if spec.get("reps") is not None and spec.get("rep_min") is not None:
        raise ToolNotice("Give either a rep count or a rep range, not both. Ask which "
                         "he meant. Nothing was changed.")
    for key in ("reps", "seconds"):
        if spec.get(key) is not None and spec[key] < 1:
            raise ToolNotice(f"{key} must be at least one. Nothing was changed.")
    if spec.get("weight_lb") is not None and spec["weight_lb"] < 0:
        raise ToolNotice("Weight cannot be negative. Nothing was changed.")
    if spec.get("rest_seconds") is not None and spec["rest_seconds"] < 0:
        raise ToolNotice("Rest cannot be negative. Nothing was changed.")


@tool(
    name="edit_routine",
    description=(
        "Change one of Lethanial's saved Hevy routines: swap an exercise for "
        "another, add one, remove one, or adjust one's sets, reps, rep range, "
        "weight, seconds per set or rest. One change per call; call it again for "
        "each further change he asks for in the same breath. It changes nothing "
        "by itself: it returns a question, asked for you word for word, and the "
        "change runs only if he says yes on his next turn, through "
        "confirm_pending_action. Say nothing before or after calling it, and "
        "never say a routine was changed until confirm_pending_action says Done. "
        "Exercise names are matched to Hevy's own; if several match, it tells "
        "you to ask which. Use get_routines first if you do not know which "
        "routine or exercise he means. This edits routines only, never a "
        "workout he already logged."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "routine": {"type": "string", "description": "The routine, like 'Lower Strength'."},
            "change": {"type": "string", "enum": ["swap", "add", "remove", "adjust"],
                       "description": "swap one exercise for another; add a new one; "
                                      "remove one; adjust one's sets, reps, weight or rest."},
            "exercise": {"type": "string",
                         "description": "The exercise already in the routine to swap, "
                                        "remove or adjust. For add, optionally the one "
                                        "to put the new exercise after; omit to add it "
                                        "at the end."},
            "new_exercise": {"type": "string",
                             "description": "For swap and add: the exercise to put in, "
                                            "in his words, like 'pike pushups'."},
            "sets": {"type": "integer", "description": "How many working sets. Warm ups are kept."},
            "reps": {"type": "integer", "description": "A fixed rep count for every working set."},
            "rep_min": {"type": "integer", "description": "Bottom of a rep range, like 6 in 6 to 8."},
            "rep_max": {"type": "integer", "description": "Top of a rep range, like 8 in 6 to 8."},
            "weight_lb": {"type": "number",
                          "description": "Pounds for every working set; 0 clears the weight."},
            "seconds": {"type": "integer", "description": "Seconds per set, for a timed hold."},
            "rest_seconds": {"type": "integer", "description": "Rest between sets, in seconds."},
        },
        "required": ["routine", "change"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def edit_routine(routine, change, exercise=None, new_exercise=None, sets=None, reps=None,
                 rep_min=None, rep_max=None, weight_lb=None, seconds=None, rest_seconds=None):
    spec = {"change": change, "sets": sets, "reps": reps, "rep_min": rep_min,
            "rep_max": rep_max, "weight_lb": weight_lb, "seconds": seconds,
            "rest_seconds": rest_seconds}
    _checked(spec)

    routines = _routines()
    found = {r["title"]: r for r in routines}
    title = _one(routine, list(found), "routine")
    chosen = found[title]
    in_routine = {e["title"]: e for e in chosen.get("exercises", [])}

    if change in ("swap", "remove", "adjust") and not exercise:
        raise ToolNotice(f"Ask which exercise in {title} he means. Nothing was changed.")
    if change in ("swap", "add") and not new_exercise:
        raise ToolNotice("Ask which exercise to put in. Nothing was changed.")
    if change == "adjust" and all(spec[k] is None for k in
                                  ("sets", "reps", "rep_min", "weight_lb", "seconds", "rest_seconds")):
        raise ToolNotice("Ask what to change about it: sets, reps, weight or rest. "
                         "Nothing was changed.")

    templates = None
    if exercise:
        spec["exercise"] = _one(exercise, list(in_routine), f"exercise in {title}")
    if change in ("swap", "add", "adjust"):
        templates = {t["id"]: t for t in _all_templates()}
    if change in ("swap", "adjust"):
        old = templates.get(in_routine[spec["exercise"]]["exercise_template_id"], {})
        spec["old_type"] = old.get("type")
    if new_exercise:
        by_title = {t["title"]: t for t in templates.values()}
        new_title = _one(new_exercise, list(by_title), "Hevy exercise")
        spec.update(new_title=new_title, new_template_id=by_title[new_title]["id"],
                    new_type=by_title[new_title]["type"])

    _, question = apply(chosen, spec)
    routine_id = chosen["id"]
    return _ask(pending_action.propose(question, lambda: _run(routine_id, spec, question)))
