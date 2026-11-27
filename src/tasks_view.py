"""The Tasks tab's view of the tasks table: the Task object, the overview and the
actions behind POST /tasks, reopen and backlog. The shapes are fixed by
docs/TASKS_CONTRACT.md. Everything reads and writes through database.py and the
nightly review's guarded block removal, so a tap leaves the same rows as the
voice tools do.

Dates are plain YYYY-MM-DD strings and "today" is the 4am day (task_day)."""

from datetime import date, datetime, timedelta

from database import (
    add_task, apply_task_completion, backlog_tasks, clear_task_calendar_event,
    get_task, task_completed_on, task_day, tasks_between, tasks_for_day,
    unschedule_task, upcoming_daily_tasks,
)


class TaskError(Exception):
    """status is the HTTP status the route answers with."""

    def __init__(self, status, detail, *, code=None, today=None, start_date=None):
        super().__init__(detail)
        self.status, self.detail = status, detail
        self.code, self.today, self.start_date = code, today, start_date


def now():
    """The clock for every function here; a test replaces it."""
    return datetime.now()


def serialize(task, today, done_on=None):
    """A task dict (from get_task or the task_for_day family) as the contract's
    Task, exactly its keys. done_on says a daily task has a completion today."""
    daily = task["recurrence"] is not None
    status = ("done" if done_on else "pending") if daily else task["status"]
    overdue = (not daily and status == "pending"
               and any(d and d < today for d in (task["scheduled_for"], task["due_date"])))
    return {
        "id": task["id"],
        "title": task["text"],
        "short_title": task["short_title"],
        "notes": task["notes"],
        "daily": daily,
        "status": status,
        "must_do": bool(task["must_do"]),
        "scheduled_for": task["scheduled_for"],
        "due_date": task["due_date"],
        "start_date": task["start_date"],
        "overdue": overdue,
        "has_block": bool(task["calendar_event_id"]),
        "source": task["source"],
    }


def one(task_id, today=None):
    """The Task for an id as of today, or None when there is no active task."""
    task = get_task(task_id)
    if task is None or task["archived_at"] is not None:
        return None
    today = today or task_day(now())
    done_on = (task["recurrence"] is not None
               and (task["start_date"] is None or task["start_date"] <= today)
               and task_completed_on(task_id, today))
    return serialize(task, today, done_on)


def _day_label(day, today):
    if day == today:
        return "Today"
    return date.fromisoformat(day).strftime("%A %b %-d")


def overview(days=3):
    today = task_day(now())
    first = date.fromisoformat(today)
    shown = []
    for n in range(days):
        day = (first + timedelta(days=n)).isoformat()
        if n == 0:
            rows = tasks_for_day(day, today)
        else:
            rows = tasks_between(day, day)
        tasks = [serialize(t, today, t["status"] == "done")
                 for t in rows
                 if t["status"] in ("pending", "done") and (n == 0 or t["recurrence"] is None)]
        shown.append({"date": day, "label": _day_label(day, today), "tasks": tasks})
    backlog = sorted(backlog_tasks(),
                     key=lambda t: (t["due_date"] is None, t["due_date"] or "", t["id"]))
    return {"today": today, "days": shown,
            "backlog": [serialize(t, today) for t in backlog],
            "upcoming_daily": [serialize(t, today) for t in upcoming_daily_tasks(today)]}


def create(title, where, due_date, start_date=None):
    captured = now()
    today = task_day(captured)
    daily = where == "daily"
    if daily and due_date is not None:
        raise TaskError(422, "A daily task has no due date")
    try:
        task_id = add_task(title, recurrence="daily" if daily else None, now=captured,
                           scheduled_for=today if where == "today" else None,
                           due_date=due_date, start_date=start_date)
    except ValueError as exc:
        raise TaskError(422, str(exc)) from None
    return one(task_id, today=today)


def apply_completion(task_id, expected_day, *, complete):
    captured = now()
    today = task_day(captured)
    result, task = apply_task_completion(task_id, complete=complete, now=captured,
                                         today=today, expected_day=expected_day)
    if result == "inactive":
        raise TaskError(404, "No active task with that id")
    if result == "day_changed":
        raise TaskError(409, "The day changed at 4 a.m. Nothing was changed.",
                        code="day_changed", today=today)
    if result == "not_started":
        start_date = task["start_date"]
        first_day = date.fromisoformat(start_date).strftime("%a %b %-d")
        raise TaskError(409, f"{task['text']} starts {first_day}.",
                        code="not_started", start_date=start_date)
    if result == "not_done":
        detail = "That task is not done today" if task["recurrence"] is not None else "That task is not done"
        raise TaskError(409, detail, code="not_done")
    return today, serialize(task, today, done_on=complete)


def to_backlog(task_id, remove_block, drop_from_reviews=None):
    """remove_block is nightly_review._remove_block_of, passed in by the route:
    it deletes the block only from the MILES calendar and forgets the id."""
    task = get_task(task_id)
    if task is None or task["archived_at"] is not None:
        raise TaskError(404, "No active task with that id")
    if task["recurrence"] is not None:
        raise TaskError(409, "A daily task cannot go to the backlog")
    if task["scheduled_for"] is None:
        raise TaskError(409, "That task is already in the backlog")
    if task["status"] != "pending":
        raise TaskError(409, "Only a pending task can go to the backlog")
    try:
        remove_block(task, now())
    except Exception as exc:
        raise TaskError(502, "The calendar block could not be removed") from exc
    unschedule_task(task_id, now())
    if drop_from_reviews:
        drop_from_reviews(task_id)
    return one(task_id)
