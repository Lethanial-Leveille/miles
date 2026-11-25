"""The Tasks tab's view of the tasks table: the Task object, the overview and the
actions behind POST /tasks, reopen and backlog. The shapes are fixed by
docs/TASKS_CONTRACT.md. Everything reads and writes through database.py and the
nightly review's guarded block removal, so a tap leaves the same rows as the
voice tools do.

Dates are plain YYYY-MM-DD strings and "today" is the 4am day (task_day)."""

from datetime import date, datetime, timedelta

from database import (
    add_task, backlog_tasks, clear_task_calendar_event, complete_task, get_task,
    set_task_status, task_completed_on, task_day, tasks_between, tasks_for_day,
    unschedule_task, remove_task_completion,
)


class TaskError(Exception):
    """status is the HTTP status the route answers with."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status, self.detail = status, detail


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
        "overdue": overdue,
        "has_block": bool(task["calendar_event_id"]),
        "source": task["source"],
    }


def one(task_id):
    """The Task for an id as of today, or None when there is no active task."""
    task = get_task(task_id)
    if task is None or task["archived_at"] is not None:
        return None
    today = task_day(now())
    done_on = task["recurrence"] is not None and task_completed_on(task_id, today)
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
            "backlog": [serialize(t, today) for t in backlog]}


def create(title, where, due_date):
    today = task_day(now())
    task_id = add_task(title, now=now(),
                       scheduled_for=today if where == "today" else None, due_date=due_date)
    return one(task_id)


def complete(task_id):
    if not complete_task(task_id, now()):
        raise TaskError(404, "No active task with that id")
    return one(task_id)


def reopen(task_id):
    task = get_task(task_id)
    if task is None or task["archived_at"] is not None:
        raise TaskError(404, "No active task with that id")
    today = task_day(now())
    if task["recurrence"] is not None:
        if not remove_task_completion(task_id, today):
            raise TaskError(409, "That task is not done today")
    else:
        if task["status"] != "done":
            raise TaskError(409, "That task is not done")
        set_task_status(task_id, "pending", now())
    return one(task_id)


def to_backlog(task_id, remove_block):
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
    return one(task_id)
