"""The nightly review: close out today and plan tomorrow, by proposing.

The first session type. At 21:30 the timer opens one for the day and pushes it
to his phone. Code opens with a proposal, no model call: how today went, what
tomorrow's calendar holds, and up to three must dos with times in his free
gaps (review_plan.py). He answers "good" and the model calls confirm_plan, or
he asks for a change and the model calls set_plan with the whole plan he now
wants. Either way code writes the message he reads, so no time in it can be
wrong; the engine holds the model's own words to one short line (see
sessions.guard_text).

The model words it; code does every count and every time. That is why it runs
on Haiku (SESSION_MODELS): the cheapest model that does the job.

Its tools live in their own registry (session_types.py explains why), and
every calendar write goes to the MILES calendar only, through calendar_tools,
which also refuses any event Evallo owns.
"""

import datetime

import calendar_roles
import calendar_tools
import evallo_filter
import sessions
from config import SESSION_BLOCK_EARLIEST, SESSION_BLOCK_LATEST
from database import (add_attending, add_task, attending_set, backlog_tasks, complete_task,
                      get_task, last_session_summary, move_task, must_do_counts,
                      save_day_plan, session_has_user_message, session_plan,
                      set_session_plan, set_task_calendar_event, set_task_status, task_day,
                      tasks_between, tasks_for_day, unschedule_task)
from review_plan import (ASK, MUST_DO_CAP, block_description, block_title, build_proposal,
                         clock_words, day_words,
                         default_minutes, existing_block, find_slot, fitting_gaps, local_busy, live_block,
                         overlaps, place_blocks, place_plan, plan_text, range_words,
                         read_events, render_plan, short_day, today_line, week_view,
                         window_of)
from session_types import (Schedule, SessionType, current_session_id, post_message,
                           register_type, request_close)
from tools import Permission, ToolNotice, ToolRegistry

NAME = "nightly_review"
UPCOMING_DAYS = 7

# The contract's times: created and pushed at 21:30, reminded at 22:30 if never
# opened, settled at 03:00 the next morning.
SCHEDULE = Schedule(opens_at=datetime.time(21, 30), remind_at=datetime.time(22, 30),
                    settles_at=datetime.time(3, 0))

registry = ToolRegistry()
tool = registry.register


# ── words ──

def _day(value):
    return datetime.date.fromisoformat(value) if isinstance(value, str) else value


def _count(n, one, many):
    return f"{n} {one if n == 1 else many}"


_STATUS_WORDS = {"done": "done", "skipped": "skipped", "moved": "moved",
                 "pending": "not done yet"}


# ── pure rules, tested without Google or a model ──

def must_do_overflow(existing, new_tasks, cap=MUST_DO_CAP):
    """The days a batch would push past the cap, as {day: (slots left, asked)}.

    existing is {day: must dos already there}; new_tasks is (day, must_do)
    pairs. Empty means the batch fits. The whole batch is judged together, so
    three new must dos for one day are refused as one, not two saved and one
    refused."""
    asked = {}
    for day, must_do in new_tasks:
        if must_do:
            asked[day] = asked.get(day, 0) + 1
    over = {}
    for day, count in asked.items():
        left = max(cap - existing.get(day, 0), 0)
        if count > left:
            over[day] = (left, count)
    return over


def _gap_list(gaps):
    return ", ".join(f"{clock_words(a)} to {clock_words(b)}" for a, b in gaps)


# ── input checks ──

def _date_arg(value, name="date"):
    try:
        return datetime.date.fromisoformat(str(value))
    except ValueError:
        raise ToolNotice(f"{name} {value!r} is not a date; send it as YYYY-MM-DD.") from None


def _clock_arg(value, name):
    try:
        return datetime.time.fromisoformat(str(value))
    except ValueError:
        raise ToolNotice(f"{name} {value!r} is not a clock time; send it as HH:MM, "
                         "24 hour.") from None


def _active_task(task_id):
    task = get_task(task_id)
    if task is None or task["archived_at"] is not None:
        raise ToolNotice(f"There is no task with id {task_id}. Call get_tasks_for_date "
                         "for the ids.")
    return task


def _clock():
    """The time, as one function so a test can set it. Tools take a now
    argument for direct calls; through the engine nothing passes one."""
    return datetime.datetime.now()


def _today(now=None):
    return _day(task_day(now))


# ── tools ──

@tool(
    name="get_tasks_for_date",
    description=(
        "The tasks for one day, with ids, statuses (pending, done, skipped, moved), "
        "must do flags and whether each has a calendar block. For today it also "
        "includes earlier tasks still pending. Call it first, for today, to get the "
        "ids update_task_status needs."
    ),
    input_schema={
        "type": "object",
        "properties": {"date": {"type": "string", "description": "YYYY-MM-DD"}},
        "required": ["date"],
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_tasks_for_date(date, now=None):
    day = _date_arg(date)
    tasks = tasks_for_day(day.isoformat(), today=task_day(now))
    return {"date": day.isoformat(), "day": day_words(day), "tasks": [
        {"id": t["id"], "title": t["text"], "status": t["status"], "must_do": t["must_do"],
         "daily": t["recurrence"] == "daily", "notes": t["notes"],
         "scheduled_for": t["scheduled_for"], "has_block": bool(t["calendar_event_id"])}
        for t in tasks]}


@tool(
    name="update_task_status",
    description=(
        "Set one task's status from what he said: done, skipped, moved (with "
        "move_to, the new date, which makes a copy there) or pending to undo a "
        "mistake. A daily task can only be marked done. Only for tasks he told you "
        "about."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "status": {"type": "string", "enum": ["done", "skipped", "moved", "pending"]},
            "move_to": {"type": "string", "description": "YYYY-MM-DD, only with moved."},
        },
        "required": ["task_id", "status"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
)
def update_task_status(task_id, status, move_to=None, now=None):
    task = _active_task(task_id)
    title = task["text"]
    if task["status"] == "moved":
        raise ToolNotice(f"{title} was already moved; update the copy on its new day.")

    if task["recurrence"] == "daily":
        if status != "done":
            raise ToolNotice(f"{title} is a daily task; it can only be marked done.")
        complete_task(task_id, now)
        return f"Marked {title} done for today."

    if status == "done":
        complete_task(task_id, now)
        return f"Marked {title} done."
    if status in ("skipped", "pending"):
        set_task_status(task_id, status, now)
        return f"Marked {title} {_STATUS_WORDS[status]}."
    if status != "moved":
        raise ToolNotice(f"{status!r} is not a status; use done, skipped, moved or pending.")

    if task["status"] == "done":
        # Moving finished work would bring it back as pending on the new day.
        raise ToolNotice(f"{title} is already done; there is nothing to move.")
    if not move_to:
        raise ToolNotice(f"Moving {title} needs move_to, the new date.")
    new_day = _date_arg(move_to, "move_to")
    if new_day < _today(now):
        raise ToolNotice(f"{day_words(new_day)} has already passed.")
    if task["scheduled_for"] == new_day.isoformat():
        raise ToolNotice(f"{title} is already on {day_words(new_day)}.")
    if task["must_do"]:
        over = must_do_overflow(must_do_counts([new_day.isoformat()]),
                                [(new_day.isoformat(), True)])
        if over:
            raise ToolNotice(f"{day_words(new_day)} already has {MUST_DO_CAP} must dos, so "
                             f"{title} cannot move there as one. Nothing was changed.")
    new_id = move_task(task_id, new_day.isoformat(), now)
    if new_id is None:
        raise ToolNotice(f"{title} could not be moved; it is no longer pending.")
    return f"Moved {title} to {day_words(new_day)} (the copy there is task {new_id})."


@tool(
    name="get_upcoming_items",
    description=(
        "What is already planned for the days after today, as data for your own "
        "reading: tasks by date, every calendar event with its note (club, "
        "holiday, likely cancelled), and the conflicts between events. Never "
        "repeat times from it to him; use show_week to show him the week. If the "
        "calendar cannot be read, the result says so and the tasks are still there."
    ),
    input_schema={
        "type": "object",
        "properties": {"days": {"type": "integer", "description": "How many days, 1 to 14. Default 7."}},
    },
    permission=Permission.READ,
    returns_to_model=True,
    min_tier="hokage",
)
def get_upcoming_items(days=UPCOMING_DAYS, now=None):
    now = now or _clock()
    if not isinstance(days, int) or not 1 <= days <= 14:
        raise ToolNotice("days must be a whole number from 1 to 14.")
    first = _today(now) + datetime.timedelta(days=1)
    last = first + datetime.timedelta(days=days - 1)
    result = {"from": first.isoformat(), "to": last.isoformat(), "tasks": [
        {"id": t["id"], "title": t["text"], "date": t["scheduled_for"],
         "day": day_words(t["scheduled_for"]), "status": t["status"],
         "must_do": t["must_do"], "has_block": bool(t["calendar_event_id"])}
        for t in tasks_between(first.isoformat(), last.isoformat())]}
    try:
        window = calendar_tools.reviewed_events(
            datetime.datetime.combine(first, datetime.time(0)),
            datetime.datetime.combine(last + datetime.timedelta(days=1), datetime.time(0)))
    except Exception as exc:
        # The plan can still be made from the tasks; Nova says the calendar
        # was not readable rather than presenting an empty week.
        result["calendar_error"] = f"could not read the calendar: {exc}"
        return result
    events = []
    for e in window["events"]:
        start, end = calendar_roles.event_span(e)
        when = "all day" if e["all_day"] else range_words(start, end)
        events.append({"date": start.date().isoformat(), "day": day_words(start.date()),
                       "when": when, "title": e["title"], "calendar": e["calendar"],
                       "note": e["mark"] or None, "busy": e["busy"]})
    result["events"] = events
    result["conflicts"] = [
        {"date": c["start"].date().isoformat(), "between": [c["a"]["title"], c["b"]["title"]],
         "overlap": range_words(c["start"], c["end"])}
        for c in calendar_roles.find_conflicts(window["events"])]
    if window.get("unreadable"):
        result["calendars_not_read"] = window["unreadable"]
    return result


# ── the plan ──

def _session_plan():
    """(session id, saved plan) of the session this turn is in."""
    session_id = current_session_id()
    plan = session_plan(session_id) if session_id else None
    if plan is None:
        raise ToolNotice("There is no saved plan for this session.")
    return session_id, plan


def _session_today(session_id):
    from database import get_session
    session = get_session(session_id)
    return session["for_day"] or task_day()


def _item_matching(plan, title):
    """The plan item (must do or optional) whose title is `title`, ignoring
    case, or the only one that contains it. None when there is no single one."""
    items = plan["must_dos"] + plan["optional"]
    wanted = title.casefold().strip()
    exact = [i for i in items if i["title"].casefold().strip() == wanted]
    if exact:
        return exact[0]
    partial = [i for i in items if wanted in i["title"].casefold()
               or i["title"].casefold() in wanted]
    return partial[0] if len(partial) == 1 else None


def _task_by_title(title, day):
    """A task already on that day or in the backlog whose title is `title`
    exactly (ignoring case), so naming a known task in a plan links to it
    instead of making a second copy."""
    wanted = title.casefold().strip()
    for task in tasks_between(day, day) + backlog_tasks():
        if task["text"].casefold().strip() == wanted:
            return task
    return None


def _entry(raw, plan, must):
    """One item of a set_plan request as a plan item, with its ids resolved: by
    task_id when given, else from the saved plan, else a task of that name.
    The model sees titles only, never ids, so code keeps them."""
    if isinstance(raw, str):
        raw = {"title": raw}
    title = str(raw.get("title") or "").strip()
    if not title:
        raise ToolNotice("Every item needs a title. Nothing was changed.")
    task_id = raw.get("task_id")
    old = _item_matching(plan, title)
    if task_id:
        task = _active_task(task_id)
        if task["status"] != "pending" or task["recurrence"] is not None:
            raise ToolNotice(f"{task['text']} is {task['status'] if task['recurrence'] is None else 'a daily task'}, "
                             "so it cannot be planned. Nothing was changed.")
        title = title or task["text"]
    elif old and old.get("task_id"):
        task_id = old["task_id"]
    else:
        task = _task_by_title(title, plan["day"])
        task_id = task["id"] if task else None
    item = {"title": title, "task_id": task_id,
            "short_title": (raw.get("short_title") or "").strip() or (old or {}).get("short_title")}
    if must:
        minutes = raw.get("minutes")
        if minutes is not None and (not isinstance(minutes, int) or not 5 <= minutes <= 600):
            raise ToolNotice("minutes must be a whole number from 5 to 600. Nothing was changed.")
        start = raw.get("start")
        if start:
            _clock_arg(start, "start")
        item.update(minutes=minutes if minutes is not None else (old or {}).get("minutes"),
                    start=start or (old or {}).get("start"),
                    reason=(old or {}).get("reason") or "Your choice", slot=None)
    return item


@tool(
    name="set_plan",
    description=(
        "Change tomorrow's plan. Send the FULL plan he now wants, not just the "
        "difference: must_dos (at most 3, in priority order, each a title, with "
        "minutes and a fixed start HH:MM only if he said so) and optional tasks. "
        "Use the titles as they were shown; code keeps their ids. Code finds "
        "the times and posts the updated plan to him, so do not write times or "
        "repeat the plan. Add earliest and latest (HH:MM) to widen or narrow "
        "the hours."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "must_dos": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "task_id": {"type": "integer"},
                "short_title": {"type": "string",
                                "description": "Optional 2 to 4 word name for its calendar block."},
                "minutes": {"type": "integer"}, "start": {"type": "string"}},
                "required": ["title"]}},
            "optional": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "task_id": {"type": "integer"}},
                "required": ["title"]}},
            "earliest": {"type": "string", "description": "HH:MM, 24 hour."},
            "latest": {"type": "string", "description": "HH:MM, 24 hour."},
        },
        "required": ["must_dos"],
    },
    permission=Permission.WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def set_plan(must_dos, optional=None, earliest=None, latest=None, now=None):
    now = now or _clock()
    session_id, plan = _session_plan()
    if not isinstance(must_dos, list):
        raise ToolNotice("must_dos has to be a list.")
    if len(must_dos) > MUST_DO_CAP:
        raise ToolNotice(f"At most {MUST_DO_CAP} must dos a day, and this asked for "
                         f"{len(must_dos)}. Ask which to make optional. Nothing was changed.")
    new = dict(plan)
    new["must_dos"] = [_entry(m, plan, True) for m in must_dos]
    new["optional"] = [_entry(o, plan, False) for o in (optional or [])]
    if earliest:
        new["earliest"] = _clock_arg(earliest, "earliest").strftime("%H:%M")
    if latest:
        new["latest"] = _clock_arg(latest, "latest").strftime("%H:%M")
    if window_of(new)[1] <= window_of(new)[0]:
        raise ToolNotice("latest has to be after earliest. Nothing was changed.")
    new = place_plan(new, now)
    set_session_plan(session_id, new)
    post_message(plan_text(_session_today(session_id), new, now))
    left = [m["title"] for m in new["must_dos"] if not m["slot"]] if not new["unchecked"] else []
    return "Posted the updated plan." + (f" No time found for: {', '.join(left)}." if left else "")


@tool(
    name="confirm_plan",
    description=(
        "He said the plan is good. Call it once: code checks the calendar again, "
        "saves tomorrow's tasks, books every must do on the MILES calendar, "
        "posts the final plan and closes the review. Write only a short good "
        "night beside it, with no times."
    ),
    input_schema={"type": "object", "properties": {}},
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def confirm_plan(now=None):
    now = now or _clock()
    session_id, plan = _session_plan()
    if plan.get("confirmed"):
        # A second call (the model sending it twice in one round, or again next
        # turn) must not insert the tasks or book the blocks a second time.
        raise ToolNotice("This plan was already confirmed and booked. Nothing was changed.")
    day = _day(plan["day"])
    # A task finished (in the app or by voice) while the review was open is not
    # planned again: no day change, no block.
    done = {e["task_id"] for e in plan["must_dos"] + plan["optional"]
            if e.get("task_id") and (get_task(e["task_id"]) or {}).get("status") == "done"}
    if done:
        plan = _plan_without(plan, done)
    # The calendar may have changed since 21:30: read it again, keep every slot
    # that is still free, and place again any whose slot is not.
    plan = place_plan(plan, now, keep_slots=True)
    if plan["unchecked"]:
        raise ToolNotice("The calendar could not be read, so nothing was booked. "
                         "Try again in a moment.")

    # A must do he dropped from the plan stays on the day as an optional task,
    # so nothing is lost, but it no longer holds a must do slot or a block.
    must_ids = {m["task_id"] for m in plan["must_dos"] if m.get("task_id")}
    dropped = [t for t in tasks_between(day.isoformat(), day.isoformat())
               if t["must_do"] and t["id"] not in must_ids]

    items = [{"task_id": m.get("task_id"), "text": m["title"], "must_do": True,
              "minutes": default_minutes(m["title"], m.get("minutes")),
              "short_title": m.get("short_title")} for m in plan["must_dos"]]
    items += [{"task_id": o.get("task_id"), "text": o["title"], "must_do": False,
               "short_title": o.get("short_title")} for o in plan["optional"]]
    items += [{"task_id": t["id"], "text": t["text"], "must_do": False} for t in dropped]
    for task in dropped:
        _remove_block_of(task, now)
    ids = save_day_plan(day.isoformat(), items, now)

    # The ids code just made go back into the plan and are saved at once, so a
    # failure below (Google) leaves a plan that a retry completes instead of
    # one that would insert every task again.
    for entry, task_id in zip(plan["must_dos"] + plan["optional"], ids):
        entry["task_id"] = task_id
    set_session_plan(session_id, plan)

    problems = []
    for m in plan["must_dos"]:
        try:
            task = get_task(m["task_id"])
            existing = existing_block(task)
            if (existing and existing["start"].date() == day) or not m.get("slot"):
                continue
            start = datetime.datetime.combine(day, datetime.time.fromisoformat(m["slot"][0]))
            end = datetime.datetime.combine(day, datetime.time.fromisoformat(m["slot"][1]))
            title = block_title(task["text"], task.get("short_title"))
            description = block_description(task, m.get("reason"))
            if existing:
                # The task's block is on another day: move that one here, so no
                # block is left behind and none is booked twice.
                calendar_tools.move_miles_event(existing["id"], start, end, title, description)
            else:
                _, event_id = calendar_tools.add_miles_event(title, start, end, description)
                set_task_calendar_event(task["id"], event_id, now)
        except Exception as exc:
            problems.append(f"{m['title']} ({exc})")
    plan["confirmed"] = True
    set_session_plan(session_id, plan)
    post_message(render_plan(day, order=[m["task_id"] for m in plan["must_dos"]]))
    request_close()
    return "Booked, and closing." + (f" Could not book: {'; '.join(problems)}." if problems else "")


@tool(
    name="add_to_backlog",
    description=(
        "Add something to his backlog, with a due date (YYYY-MM-DD) if he gave "
        "one. The backlog is what he has to do with no day chosen yet."
    ),
    input_schema={
        "type": "object",
        "properties": {"title": {"type": "string"},
                       "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                       "short_title": {"type": "string",
                                       "description": "Optional 2 to 4 word name for its calendar block."}},
        "required": ["title"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
)
def add_to_backlog(title, due_date=None, short_title=None, now=None):
    now = now or _clock()
    title = str(title or "").strip()
    if not title:
        raise ToolNotice("The backlog item needs a title.")
    due = _date_arg(due_date, "due_date").isoformat() if due_date else None
    task_id = add_task(title[:1].upper() + title[1:], now=now, due_date=due,
                       short_title=(short_title or "").strip() or None)
    return f"Added {title} to the backlog (task {task_id})" + (f", due {day_words(due)}." if due else ".")


def _plan_without(plan, task_ids=(), item=None):
    """The plan with those tasks (and `item`, a plan entry without a task yet)
    taken out of both lists. One place, shared by push_to_backlog and the Tasks
    tab's backlog route, so neither can leave the other's item behind."""
    ids = set(task_ids)
    new = dict(plan)
    for key in ("must_dos", "optional"):
        new[key] = [e for e in plan[key]
                    if e is not item and e.get("task_id") not in ids]
    return new


def drop_from_open_reviews(task_id):
    """Take a task out of the saved plan of every open, unconfirmed nightly
    review. The Tasks tab calls this after sending a task to the backlog;
    without it confirming the review would put the task back on the day and
    book a new block. Placement is left alone: confirm_plan places again."""
    from database import list_sessions
    for session in list_sessions("open", ["nightly_review"]):
        plan = session_plan(session["id"])
        if not plan or plan.get("confirmed"):
            continue
        new = _plan_without(plan, [task_id])
        if len(new["must_dos"]) + len(new["optional"]) != len(plan["must_dos"]) + len(plan["optional"]):
            set_session_plan(session["id"], new)


def _remove_block_of(task, now=None):
    """Delete a task's block from MILES if it is there, and forget its id."""
    from database import clear_task_calendar_event
    if task["calendar_event_id"]:
        calendar_tools.delete_miles_event(task["calendar_event_id"])
        clear_task_calendar_event(task["id"], now)


@tool(
    name="push_to_backlog",
    description=(
        "Take something out of tomorrow's plan and put it back in the backlog: "
        "give a task_id, or the title as shown in the plan. Removes its block "
        "from the MILES calendar if it had one, and posts the updated plan."
    ),
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "integer"}, "title": {"type": "string"}},
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=False,
    min_tier="hokage",
)
def push_to_backlog(task_id=None, title=None, now=None):
    now = now or _clock()
    session_id, plan = _session_plan()
    item = None
    if task_id:
        task = _active_task(task_id)
    elif title:
        item = _item_matching(plan, title)
        if item is None:
            raise ToolNotice(f"No single item in the plan matches {title!r}.")
        task = get_task(item["task_id"]) if item.get("task_id") else None
    else:
        raise ToolNotice("Give a task_id or the title of a plan item.")

    if task is not None:
        if task["status"] != "pending" or task["recurrence"] is not None:
            raise ToolNotice(f"{task['text']} cannot go to the backlog; it is "
                             f"{'a daily task' if task['recurrence'] else task['status']}.")
        _remove_block_of(task, now)
        unschedule_task(task["id"], now)
        name = task["text"]
    else:
        name = item["title"]
        add_task(name, now=now, minutes=item.get("minutes"))

    new = _plan_without(plan, {task["id"]} if task else set(), item)
    if len(new["must_dos"]) + len(new["optional"]) == len(plan["must_dos"]) + len(plan["optional"]):
        return f"Moved {name} to the backlog. It was not in the plan."
    new = place_plan(new, now)
    set_session_plan(session_id, new)
    post_message(plan_text(_session_today(session_id), new, now))
    return f"Moved {name} to the backlog and posted the updated plan."


@tool(
    name="show_week",
    description=(
        "Post the next seven days to him: each day with its events, conflicts "
        "and open days. Code writes it from the calendar, so write no times of "
        "your own."
    ),
    input_schema={"type": "object", "properties": {}},
    permission=Permission.READ,
    returns_to_model=False,
    min_tier="hokage",
)
def show_week(now=None):
    now = now or _clock()
    first = _today(now) + datetime.timedelta(days=1)
    events, error = read_events(first, UPCOMING_DAYS)
    if error:
        raise ToolNotice(f"The week could not be shown: {error}.")
    post_message("The week ahead\n\n" + week_view(events, first, UPCOMING_DAYS))
    return "Posted the week."


# ── blocks ──

def _block_of(task_id):
    """(task, its live MILES event) for move_block and remove_block. Looks the
    event up on the MILES calendar only. An id not found there is refused: an
    event on his own calendar is not Nova's to change, and a block he deleted
    by hand is gone, so its stale id is cleared."""
    from database import clear_task_calendar_event
    task = _active_task(task_id)
    if not task["calendar_event_id"]:
        raise ToolNotice(f"{task['text']} has no block on the MILES calendar.")
    event = calendar_tools.get_miles_event(task["calendar_event_id"])
    if event is None:
        clear_task_calendar_event(task_id)
        raise ToolNotice(f"{task['text']} has no block on the MILES calendar any more "
                         "(it is not there, or it is not a MILES event). Nothing was changed. "
                         "Book it again with schedule_block if he wants it.")
    return task, event


@tool(
    name="move_block",
    description=(
        "Move a task's block on the MILES calendar to a new start (HH:MM, 24 "
        "hour, same day), optionally with a new length in minutes. Refused if "
        "it overlaps his calendar, and refused for anything not on MILES."
    ),
    input_schema={
        "type": "object",
        "properties": {"task_id": {"type": "integer"},
                       "start_time": {"type": "string", "description": "HH:MM, 24 hour."},
                       "minutes": {"type": "integer"}},
        "required": ["task_id", "start_time"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def move_block(task_id, start_time, minutes=None, now=None):
    now = now or _clock()
    task, event = _block_of(task_id)
    day = event["start"].date()
    length = (int((event["end"] - event["start"]).total_seconds() // 60)
              if minutes is None else minutes)
    if not isinstance(length, int) or not 5 <= length <= 600:
        raise ToolNotice("minutes must be a whole number from 5 to 600.")
    start = datetime.datetime.combine(day, _clock_arg(start_time, "start_time"))
    end = start + datetime.timedelta(minutes=length)
    busy, _ = local_busy(day, exclude_ids={event["id"]})
    if overlaps(busy, start, end):
        raise ToolNotice(f"{clock_words(start)} to {clock_words(end)} on {day_words(day)} overlaps "
                         "something on his calendar. Nothing was moved.")
    # Only the time: a title or description he edited by hand stays as he left it.
    calendar_tools.move_miles_event(event["id"], start, end)
    return (f"Moved {task['text']} to {clock_words(start)} to {clock_words(end)} on "
            f"{day_words(day)}.")


@tool(
    name="remove_block",
    description=(
        "Delete a task's block from the MILES calendar. The task stays; it just "
        "has no block. Refused for anything not on MILES."
    ),
    input_schema={"type": "object", "properties": {"task_id": {"type": "integer"}},
                  "required": ["task_id"]},
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def remove_block(task_id, now=None):
    now = now or _clock()
    task, event = _block_of(task_id)
    _remove_block_of(task, now)
    return f"Removed the block for {task['text']} from the MILES calendar."


@tool(
    name="mark_attending",
    description=(
        "He goes to this club event, so it counts as busy. Give the title and the "
        "day it falls on (YYYY-MM-DD). whole_series true (the default) marks every "
        "repeat of a repeating event."
    ),
    input_schema={
        "type": "object",
        "properties": {"title": {"type": "string"},
                       "day": {"type": "string", "description": "YYYY-MM-DD"},
                       "whole_series": {"type": "boolean"}},
        "required": ["title", "day"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def mark_attending(title, day, whole_series=True, now=None):
    now = now or _clock()
    when = _date_arg(day, "day")
    start = datetime.datetime.combine(when, datetime.time(0))
    window = calendar_tools.events_for_app(start, start + datetime.timedelta(days=1))
    found = calendar_roles.find_attendable(window["events"], title)
    if not found:
        raise ToolNotice(f"No club event with {title!r} in its title on {day_words(when)}.")
    series = {e["series_id"] for e in found}
    if len(series) > 1:
        raise ToolNotice("More than one club event matches: "
                         + ", ".join(sorted({e["title"] for e in found})) + ". Use the full title.")
    event = found[0]
    series_id = event["series_id"] if whole_series else event["id"]
    new = add_attending(event["calendar_id"], series_id, event["title"], now)
    scope = "every repeat" if whole_series and event["repeating"] else "that one"
    return (f"Marked {event['title']} as attended ({scope}); it now counts as busy."
            if new else f"{event['title']} was already marked as attended.")


def _local_busy(start, end):
    """Busy blocks on the selected calendars as naive local times, the way
    every other time here is kept, and the calendars that could not be read."""
    busy, unreadable = calendar_tools.busy_blocks(start, end)
    local = [(a.astimezone().replace(tzinfo=None), b.astimezone().replace(tzinfo=None))
             for a, b in busy]
    return local, unreadable


@tool(
    name="schedule_block",
    description=(
        "Put a task on the MILES calendar as a block on the task's own day. Give "
        "start_time (HH:MM, 24 hour) for a time he chose, or leave it out to take "
        "the first free gap that fits; earliest and latest (HH:MM) narrow where it "
        "looks. duration_minutes defaults to 60 (120 for studying for an exam). "
        "To book several at once send tasks, a list of {task_id, duration_minutes, "
        "start_time}; they are placed in order without overlapping. A time that "
        "overlaps his calendar is refused with the free gaps listed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "duration_minutes": {"type": "integer"},
            "start_time": {"type": "string", "description": "HH:MM, 24 hour."},
            "earliest": {"type": "string", "description": "HH:MM, 24 hour."},
            "latest": {"type": "string", "description": "HH:MM, 24 hour, when it must end by."},
            "tasks": {"type": "array", "items": {"type": "object", "properties": {
                "task_id": {"type": "integer"}, "duration_minutes": {"type": "integer"},
                "start_time": {"type": "string"}}, "required": ["task_id"]}},
        },
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def schedule_block(task_id=None, duration_minutes=None, start_time=None, earliest=None,
                   latest=None, tasks=None, now=None):
    now = now or _clock()
    entries = tasks or ([{"task_id": task_id, "duration_minutes": duration_minutes,
                          "start_time": start_time}] if task_id else [])
    if not entries:
        raise ToolNotice("Give a task_id, or a list of tasks.")

    ids = [e.get("task_id") for e in entries]
    if len(set(ids)) != len(ids):
        raise ToolNotice("The same task is listed twice, which would book two blocks for it. "
                         "Nothing was added.")
    prepared = []
    for entry in entries:
        task = _active_task(entry.get("task_id"))
        title = task["text"]
        # A block he deleted by hand leaves its id behind; live_block checks
        # Google and clears a dead one, so only a block that exists refuses.
        if task["recurrence"] is not None or not task["scheduled_for"]:
            raise ToolNotice(f"{title} has no date, so it has no day to block out.")
        existing = existing_block(task)
        if existing and existing["start"].date() == _day(task["scheduled_for"]):
            raise ToolNotice(f"{title} already has a block on the calendar.")
        if task["status"] != "pending":
            raise ToolNotice(f"{title} is {task['status']}, so it needs no block.")
        minutes = entry.get("duration_minutes") or task.get("minutes")
        minutes = default_minutes(title, minutes)
        if not isinstance(minutes, int) or not 5 <= minutes <= 600:
            raise ToolNotice("duration_minutes must be a whole number from 5 to 600.")
        prepared.append({"task": task, "title": title, "minutes": minutes,
                         "move_id": existing["id"] if existing else None,
                         "day": _day(task["scheduled_for"]),
                         "start": (_clock_arg(entry["start_time"], "start_time")
                                   if entry.get("start_time") else None)})

    booked, unreadable_all = [], []
    plans = []
    for day in sorted({p["day"] for p in prepared}):
        group = [p for p in prepared if p["day"] == day]
        early = _clock_arg(earliest, "earliest") if earliest else datetime.time(*SESSION_BLOCK_EARLIEST)
        late = _clock_arg(latest, "latest") if latest else datetime.time(*SESSION_BLOCK_LATEST)
        window_start = datetime.datetime.combine(day, early)
        window_end = datetime.datetime.combine(day, late)
        if window_end <= window_start:
            raise ToolNotice("latest has to be after earliest.")
        midnight = datetime.datetime.combine(day, datetime.time(0))
        busy, unreadable = _local_busy(midnight, midnight + datetime.timedelta(days=1))
        unreadable_all += unreadable
        for p in group:
            if p["start"] is not None and datetime.datetime.combine(day, p["start"]) < now:
                raise ToolNotice(f"{clock_words(datetime.datetime.combine(day, p['start']))} on "
                                 f"{day_words(day)} has already passed.")
        not_before = now if day == now.date() else None
        slots, unplaced = place_blocks(busy, day, group, early, late, not_before)
        if unplaced:
            p = unplaced[0]
            if p["start"] is not None:
                start = datetime.datetime.combine(day, p["start"])
                end = start + datetime.timedelta(minutes=p["minutes"])
                gaps = fitting_gaps(busy, window_start, window_end, p["minutes"], now)
                free = (f"Free gaps of {p['minutes']} minutes or more between "
                        f"{clock_words(window_start)} and {clock_words(window_end)}: {_gap_list(gaps)}."
                        if gaps else "There is no free gap that long between "
                        f"{clock_words(window_start)} and {clock_words(window_end)}.")
                raise ToolNotice(f"{clock_words(start)} to {clock_words(end)} on {day_words(day)} "
                                 f"overlaps something on his calendar. Nothing was added. {free}")
            raise ToolNotice(f"No free {p['minutes']} minute gap between "
                             f"{clock_words(window_start)} and {clock_words(window_end)} on "
                             f"{day_words(day)}. Ask for a time, or a wider window. "
                             "Nothing was added.")
        plans += slots

    for slot in plans:
        p = slot["item"]
        title = block_title(p["title"], p["task"].get("short_title"))
        description = block_description(p["task"])
        if p["move_id"]:
            # Its block is on another day: move it here instead of booking a second.
            calendar_tools.move_miles_event(p["move_id"], slot["start"], slot["end"],
                                            title, description)
        else:
            _, event_id = calendar_tools.add_miles_event(title, slot["start"], slot["end"],
                                                         description)
            set_task_calendar_event(p["task"]["id"], event_id, now)
        booked.append(f"Blocked {p['title']} on {day_words(p['day'])} from "
                      f"{clock_words(slot['start'])} to {clock_words(slot['end'])} on the "
                      "MILES calendar.")
    said = " ".join(booked)
    if unreadable_all:
        said += f" These calendars could not be read, so check them: {', '.join(unreadable_all)}."
    return said


@tool(
    name="close_session",
    description=(
        "Close tonight's review. Call it when he says he is done, with a short "
        "good night written beside it."
    ),
    input_schema={"type": "object", "properties": {}},
    permission=Permission.CONTROL,
    returns_to_model=False,
)
def close_session():
    request_close()
    return "closing"


STAGES = {
    "get_tasks_for_date": "Reading today's tasks",
    "update_task_status": "Updating the task",
    "get_upcoming_items": "Checking your week",
    "show_week": "Reading your week",
    "set_plan": "Updating the plan",
    "confirm_plan": "Booking the plan",
    "add_to_backlog": "Adding to the backlog",
    "push_to_backlog": "Moving it to the backlog",
    "schedule_block": "Finding a time on your calendar",
    "move_block": "Moving the block",
    "remove_block": "Removing the block",
    "mark_attending": "Marking the event",
    "close_session": "Wrapping up",
}


# ── the session's own words ──

def opening(now):
    """The proposal, built in code with no model call: how today went, what
    tomorrow holds, up to three must dos with times. Returns (text, plan); the
    engine saves the plan with the session so "good" confirms exactly what he
    was shown. If anything in the build fails, a plain opening still goes out,
    so a calendar outage at 21:30 never means no review."""
    day = task_day(now)
    with evallo_filter.collecting() as seen:
        text, plan = _opening_body(day, now)
    # The outcome of the reads that produced this plan, never a second read.
    line = evallo_filter.note(evallo_filter.worst(seen))
    return (f"{line}.\n\n{text}" if line else text), plan


def _opening_body(day, now):
    try:
        plan = build_proposal(day, now)
        return plan_text(day, plan, now), plan
    except Exception as exc:
        print(f"Nightly review proposal failed: {exc}", flush=True)
        tomorrow = (_day(day) + datetime.timedelta(days=1)).isoformat()
        plan = {"day": tomorrow, "earliest": "%02d:%02d" % SESSION_BLOCK_EARLIEST,
                "latest": "%02d:%02d" % SESSION_BLOCK_LATEST, "unchecked": True,
                "must_dos": [], "optional": []}
        return (f"{today_line(day)}\n\nI could not build tomorrow's plan just now. "
                "What matters tomorrow?"), plan


def push_body(now):
    day = task_day(now)
    count = len(tasks_for_day(day, today=day))
    head = "No tasks today." if count == 0 else f"{_count(count, 'task', 'tasks')} today."
    return f"{head} What got done, and what matters tomorrow?"


def system_prompt(session):
    """The flow, today's date and the week's dates, and last night's summary.

    Depends on the session alone, never the clock, so it is the same bytes on
    every turn of the session and stays cached."""
    day = _day(session["for_day"]) if session["for_day"] else _today()
    week = "\n".join(
        f"- {day_words(day + datetime.timedelta(days=n))}: "
        f"{(day + datetime.timedelta(days=n)).isoformat()}" for n in range(1, UPCOMING_DAYS + 1))
    last = last_session_summary(NAME, day.isoformat())
    last_night = (f"\n\nLast night's review ({day_words(last[0])}) ended with:\n{last[1]}"
                  if last else "")
    return f"""You are Nova, running Lethanial's nightly review in the Nova app. He reads it on his phone, so write short plain sentences. No markdown headings. Never use hyphens.

Today, the day being reviewed, is {day_words(day)} ({day.isoformat()}). The days ahead:
{week}

Your opening message already proposed tomorrow's plan: how today went, tomorrow's calendar, up to three suggested must dos with times, and optional tasks. It ended by asking if it is good. Code wrote it, and code writes every plan and week view he sees after it. You do not.

What he says, and what you do:
He says it is good (good, yes, looks great, book it): call confirm_plan once. It books everything, posts the final plan and closes the review. Write only a short good night beside it.
He wants a change (swap one, drop one, another time, another task, a different length): call set_plan with the FULL plan he now wants, at most three must dos in order, and optional tasks. Use the titles as they were shown. Pass minutes or a fixed start (HH:MM, 24 hour) only when he said so. Code finds the times and posts the updated plan.
He says he also did something, or skipped it: call get_tasks_for_date for today to get the ids, then update_task_status. Only for tasks he told you about.
He wants something out of the plan and back in the backlog: call push_to_backlog.
He wants something remembered for later: call add_to_backlog, with a due date if he gave one.
He wants to see the week: call show_week.
He wants a block moved or removed: call move_block or remove_block.
He says he goes to a club event: call mark_attending.
Use get_upcoming_items or get_tasks_for_date only for your own reading.

Rules:
Whenever a tool posts a plan or the week, write at most one short line beside it and never write a clock time, a time range or the plan again. Code shows the times; anything you add about times is cut.
Dates you send are YYYY-MM-DD and clock times are HH:MM, 24 hour. Take dates from the list above; do not work them out.
Never work out free time or counts yourself; the tools do that.
Never say something was saved, moved or booked unless a tool said so.
If a tool refuses, tell him what it said and ask what he wants instead.{last_night}"""


_SUMMARY_INSTRUCTIONS = (
    "You write the note a nightly review leaves for the next night's review. Two or "
    "three plain sentences: what he got done or skipped today and why if he said, "
    "and what he planned for tomorrow. No lists, no headings, no hyphens, no "
    "numbers beyond what the transcript says.")


def summary_facts(for_day):
    """The counts a summary starts with, from the table, never from the model."""
    tomorrow = (_day(for_day) + datetime.timedelta(days=1)).isoformat()
    today_tasks = tasks_for_day(for_day, today=for_day)
    by_status = {s: sum(1 for t in today_tasks if t["status"] == s)
                 for s in ("done", "skipped", "moved", "pending")}
    planned = tasks_between(tomorrow, tomorrow)
    must_dos = sum(1 for t in planned if t["must_do"])
    blocks = sum(1 for t in planned if t["calendar_event_id"])
    return (f"Today: {by_status['done']} done, {by_status['skipped']} skipped, "
            f"{by_status['moved']} moved, {by_status['pending']} left open, of "
            f"{_count(len(today_tasks), 'task', 'tasks')}. "
            f"Tomorrow: {_count(len(planned), 'task', 'tasks')}, "
            f"{_count(must_dos, 'must do', 'must dos')}, "
            f"{_count(blocks, 'block', 'blocks')} on the calendar.")


def on_close(session):
    """Code's counts first, then a short summary in words from one cheap model
    call over the transcript. If the call fails the counts alone are saved, so
    the 3am close never depends on the network."""
    facts = summary_facts(session["for_day"] or task_day())
    if not session_has_user_message(session["id"]):
        return facts
    try:
        words = sessions.summarize(session["model"] or REVIEW.model, _SUMMARY_INSTRUCTIONS,
                                   sessions.transcript(session["id"]),
                                   session_id=session["id"])
    except Exception as exc:
        print(f"Nightly review summary call failed: {exc}", flush=True)
        return facts
    return f"{facts}\n\n{words}" if words else facts


REVIEW = register_type(SessionType(
    name=NAME,
    purpose="Close out today and plan tomorrow",
    system_prompt=system_prompt,
    tools=registry,
    opening=opening,
    on_close=on_close,
    schedule=SCHEDULE,
    push_title="Nightly review",
    push_body=push_body,
    reminder_body="Still open. Two minutes to plan tomorrow.",
    stages=STAGES,
    posts_messages=True,
))
