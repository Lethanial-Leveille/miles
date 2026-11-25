"""The nightly review: close out today and plan tomorrow.

The first session type. At 21:30 the timer opens one for the day and pushes it
to his phone. Nova opens with today's tasks, read from the table in code, and
asks what got done. Then she updates each status, shows the next seven days,
asks for two or three must dos for tomorrow, puts those on the calendar as
blocks, confirms the plan, and closes when he says he is done.

The model words it; code does every count and every time. The must do cap, the
free slot, the date a task moved to and the summary's numbers are worked out
here, so Haiku only has to call simple tools and talk.

Its tools live in their own registry (session_types.py explains why), and
schedule_block writes only to the MILES calendar, through
calendar_tools.add_miles_event.
"""

import datetime

import calendar_tools
import sessions
from config import SESSION_BLOCK_EARLIEST, SESSION_BLOCK_LATEST
from database import (add_planned_tasks, complete_task, get_task, last_session_summary,
                      move_task, must_do_counts, session_has_user_message,
                      set_task_calendar_event, set_task_status, task_day, tasks_between,
                      tasks_for_day)
from session_types import Schedule, SessionType, register_type, request_close
from tools import Permission, ToolNotice, ToolRegistry

NAME = "nightly_review"
MUST_DO_CAP = 3
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


def day_words(day):
    """"Wednesday November 25"."""
    return _day(day).strftime("%A %B %-d")


def clock_words(moment):
    """"4 PM", "4:30 PM"."""
    return moment.strftime("%-I:%M %p").replace(":00", "")


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


_STEP = datetime.timedelta(minutes=15)


def _round_up(moment):
    """Up to the next quarter hour, so a block picked at 9:07 starts at 9:15."""
    base = moment.replace(minute=0, second=0, microsecond=0)
    while base < moment:
        base += _STEP
    return base


def overlaps(busy, start, end):
    return any(a < end and start < b for a, b in busy)


def fitting_gaps(busy, window_start, window_end, minutes, not_before=None):
    """Free gaps inside the window at least `minutes` long, earliest first.

    not_before trims the window's start (to now, for a block today) and is
    rounded up to a quarter hour."""
    start = window_start
    if not_before is not None and not_before > start:
        start = _round_up(not_before)
    if start >= window_end:
        return []
    need = datetime.timedelta(minutes=minutes)
    clipped = [(max(a, start), min(b, window_end)) for a, b in busy
               if a < window_end and b > start]
    gaps = calendar_tools.free_gaps(calendar_tools.merge_busy(clipped), start, window_end)
    return [(a, b) for a, b in gaps if b - a >= need]


def find_slot(busy, window_start, window_end, minutes, not_before=None):
    """(start, end) of the first free stretch that fits, or None."""
    gaps = fitting_gaps(busy, window_start, window_end, minutes, not_before)
    if not gaps:
        return None
    start = gaps[0][0]
    return start, start + datetime.timedelta(minutes=minutes)


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
        "What is already planned for the days after today: tasks by date and "
        "every calendar event, so you can show him the week before he plans "
        "tomorrow. If the calendar cannot be read, the result says so and the "
        "tasks are still there."
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
        window = calendar_tools.events_for_app(
            datetime.datetime.combine(first, datetime.time(0)),
            datetime.datetime.combine(last + datetime.timedelta(days=1), datetime.time(0)))
    except Exception as exc:
        # The plan can still be made from the tasks; Nova says the calendar
        # was not readable rather than presenting an empty week.
        result["calendar_error"] = f"could not read the calendar: {exc}"
        return result
    events = []
    for e in window["events"]:
        if e["all_day"]:
            when = "all day"
        else:
            start = datetime.datetime.fromisoformat(e["start"]).astimezone().replace(tzinfo=None)
            end = datetime.datetime.fromisoformat(e["end"]).astimezone().replace(tzinfo=None)
            when = f"{clock_words(start)} to {clock_words(end)}"
        events.append({"date": e["start"][:10], "day": day_words(e["start"][:10]),
                       "when": when, "title": e["title"], "calendar": e["calendar"]})
    result["events"] = events
    if window.get("unreadable"):
        result["calendars_not_read"] = window["unreadable"]
    return result


@tool(
    name="create_tasks",
    description=(
        "Save the tasks he chose, each with a title and a date, must_do true for a "
        "must do. At most three must dos a day; a batch that would pass that is "
        "refused whole, with how many slots are left. Call it once he has said "
        "what he wants, before scheduling blocks."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                        "must_do": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                    "required": ["title", "date"],
                },
            },
        },
        "required": ["tasks"],
    },
    permission=Permission.WRITE,
    returns_to_model=True,
)
def create_tasks(tasks, now=None):
    if not isinstance(tasks, list) or not tasks:
        raise ToolNotice("create_tasks needs at least one task.")
    today = _today(now)
    rows = []
    for task in tasks:
        title = str(task.get("title") or "").strip()
        if not title:
            raise ToolNotice("Every task needs a title. Nothing was saved.")
        day = _date_arg(task.get("date"))
        if day < today:
            raise ToolNotice(f"{day_words(day)} has already passed. Nothing was saved.")
        rows.append({"text": title, "scheduled_for": day.isoformat(),
                     "must_do": bool(task.get("must_do")), "notes": task.get("notes") or None})

    days = sorted({r["scheduled_for"] for r in rows})
    over = must_do_overflow(must_do_counts(days), [(r["scheduled_for"], r["must_do"]) for r in rows])
    if over:
        parts = [f"{day_words(day)} has room for {_count(left, 'more must do', 'more must dos')} "
                 f"and this asked for {asked}" for day, (left, asked) in sorted(over.items())]
        raise ToolNotice(f"Nothing was saved. At most {MUST_DO_CAP} must dos a day: "
                         + "; ".join(parts) + ". Ask which to make optional.")

    ids = add_planned_tasks(rows, now)
    return {"created": [{"id": task_id, "title": r["text"], "date": r["scheduled_for"],
                         "must_do": r["must_do"]} for task_id, r in zip(ids, rows)]}


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
        "looks. A time that overlaps his calendar is refused with the free gaps "
        "listed. Ask how long the task takes first."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "duration_minutes": {"type": "integer"},
            "start_time": {"type": "string", "description": "HH:MM, 24 hour."},
            "earliest": {"type": "string", "description": "HH:MM, 24 hour."},
            "latest": {"type": "string", "description": "HH:MM, 24 hour, when it must end by."},
        },
        "required": ["task_id", "duration_minutes"],
    },
    permission=Permission.EXTERNAL_WRITE,
    returns_to_model=True,
    min_tier="hokage",
)
def schedule_block(task_id, duration_minutes, start_time=None, earliest=None, latest=None,
                   now=None):
    now = now or datetime.datetime.now()
    task = _active_task(task_id)
    title = task["text"]
    if task["calendar_event_id"]:
        raise ToolNotice(f"{title} already has a block on the calendar.")
    if task["recurrence"] is not None or not task["scheduled_for"]:
        raise ToolNotice(f"{title} has no date, so it has no day to block out.")
    if task["status"] != "pending":
        raise ToolNotice(f"{title} is {task['status']}, so it needs no block.")
    if not isinstance(duration_minutes, int) or not 5 <= duration_minutes <= 600:
        raise ToolNotice("duration_minutes must be a whole number from 5 to 600.")

    day = _day(task["scheduled_for"])
    window_start = datetime.datetime.combine(
        day, _clock_arg(earliest, "earliest") if earliest else datetime.time(*SESSION_BLOCK_EARLIEST))
    window_end = datetime.datetime.combine(
        day, _clock_arg(latest, "latest") if latest else datetime.time(*SESSION_BLOCK_LATEST))
    if window_end <= window_start:
        raise ToolNotice("latest has to be after earliest.")

    midnight = datetime.datetime.combine(day, datetime.time(0))
    busy, unreadable = _local_busy(midnight, midnight + datetime.timedelta(days=1))
    length = datetime.timedelta(minutes=duration_minutes)

    if start_time:
        start = datetime.datetime.combine(day, _clock_arg(start_time, "start_time"))
        end = start + length
        if start < now:
            raise ToolNotice(f"{clock_words(start)} on {day_words(day)} has already passed.")
        if overlaps(busy, start, end):
            gaps = fitting_gaps(busy, window_start, window_end, duration_minutes, now)
            free = (f"Free gaps of {duration_minutes} minutes or more between "
                    f"{clock_words(window_start)} and {clock_words(window_end)}: {_gap_list(gaps)}."
                    if gaps else "There is no free gap that long between "
                    f"{clock_words(window_start)} and {clock_words(window_end)}.")
            raise ToolNotice(f"{clock_words(start)} to {clock_words(end)} on {day_words(day)} "
                             f"overlaps something on his calendar. Nothing was added. {free}")
    else:
        slot = find_slot(busy, window_start, window_end, duration_minutes, now)
        if slot is None:
            raise ToolNotice(f"No free {duration_minutes} minute gap between "
                             f"{clock_words(window_start)} and {clock_words(window_end)} on "
                             f"{day_words(day)}. Ask for a time, or a wider window.")
        start, end = slot

    _, event_id = calendar_tools.add_miles_event(title, start, end)
    set_task_calendar_event(task_id, event_id, now)
    said = (f"Blocked {title} on {day_words(day)} from {clock_words(start)} to "
            f"{clock_words(end)} on the MILES calendar.")
    if unreadable:
        said += f" These calendars could not be read, so check them: {', '.join(unreadable)}."
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
    "create_tasks": "Saving tomorrow's tasks",
    "schedule_block": "Finding a time on your calendar",
    "close_session": "Wrapping up",
}


# ── the session's own words ──

def _task_line(task):
    marks = []
    if task["must_do"]:
        marks.append("must do")
    if task["recurrence"] == "daily":
        marks.append("daily")
    label = f" ({', '.join(marks)})" if marks else ""
    return f"{task['text']}{label}: {_STATUS_WORDS.get(task['status'], task['status'])}"


def opening(now):
    """Today's tasks with their statuses, then the question. Built from the
    table, so the first thing he reads costs no model call and cannot be wrong
    about what was on his list."""
    day = task_day(now)
    tasks = tasks_for_day(day, today=day)
    if not tasks:
        return (f"Nothing was on your list for {day_words(day)}. What did you get done "
                "today, and what matters tomorrow?")
    lines = "\n".join(f"{i}. {_task_line(t)}" for i, t in enumerate(tasks, 1))
    return f"Here is {day_words(day)}:\n\n{lines}\n\nWhat got done?"


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
    return f"""You are Nova, running Lethanial's nightly review in the Nova app. He reads it on his phone, so write short plain sentences and short lists. No markdown headings. Never use hyphens.

Today, the day being reviewed, is {day_words(day)} ({day.isoformat()}). The days ahead:
{week}

The review goes in this order:
1. Close out today. Your opening message listed today's tasks and asked what got done. Call get_tasks_for_date for today to get the ids, then call update_task_status for each task he tells you about: done, skipped, or moved with the date he gives. Ask about any he did not mention. Never mark a task he has not told you about.
2. Show the next seven days. Call get_upcoming_items and give him a short view of what is already planned and on his calendar.
3. Ask for two or three must dos for tomorrow, plus any optional tasks. Save them with create_tasks, must_do true for the must dos. Three must dos a day is the most the tool will take, and it says how many slots are left.
4. Offer to put each must do on the calendar. Ask how long each takes, then call schedule_block. If he names a time, pass it as start_time. If a time overlaps, offer one of the free gaps the tool lists.
5. Confirm the plan in a few lines: tomorrow's must dos with their times, then the optional tasks.
6. When he says he is done, call close_session and write a short good night beside it.

Rules:
Dates you send are YYYY-MM-DD and clock times are HH:MM, 24 hour. Take them from the list above; do not work them out.
Never work out free time or counts yourself; the tools do that.
Never say something was saved, moved or scheduled unless a tool said so.
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
                                   sessions.transcript(session["id"]))
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
))
