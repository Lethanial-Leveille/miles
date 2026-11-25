# Tasks tab: the shared contract

Approved Nov 25 2026. The backend and the Nova app build against this file and nothing else.
Neither side changes it; a change is proposed, approved, then written here first.

## Task object

```json
{"id": 7, "title": "COP3530 Project 3: Campus Compass, due 11:59 PM", "short_title": "Project 3",
 "notes": null, "daily": false, "status": "pending", "must_do": false,
 "scheduled_for": null, "due_date": "2026-11-30", "overdue": false, "has_block": false,
 "source": "manual"}
```

- `title` is the task text; `short_title`, `notes`, `scheduled_for` and `due_date` can be null.
- `status` is `pending` or `done`. A daily task is `done` when it has a completion on the 4am day the
  response is for; a one time task is `done` when its status is done.
- `overdue` is true for a pending one time task whose `scheduled_for` or `due_date` is before today.
- `has_block` is true when the task has a MILES calendar block id (no Google call is made to check it).
- All dates are plain `YYYY-MM-DD` strings, never converted to a time zone. "Today" is the 4am day
  (`task_day()`): at 1am it is still the previous date.

## Routes

All need the normal login JWT. Errors are `{"detail": "..."}`.

### `GET /tasks/overview?days=3`

`days` is 1 to 7, default 3 (today plus the next two days). Answers:

```json
{"today": "2026-11-25",
 "days": [{"date": "2026-11-25", "label": "Today", "tasks": [Task]},
          {"date": "2026-11-26", "label": "Thursday Nov 26", "tasks": [Task]}],
 "backlog": [Task]}
```

- Today: daily tasks; one time tasks scheduled today or earlier that are pending (earlier ones have
  `overdue` true); and one time tasks completed today (status done), so a checked item stays visible.
- Each later day: one time tasks scheduled on that date, pending or done.
- Backlog: one time tasks with no `scheduled_for`, pending only, ordered by `due_date` (soonest first,
  null last), then by when they were added.
- Moved, skipped and archived tasks never appear.

### `POST /tasks`

Body `{"title": "...", "where": "today" | "backlog", "due_date": "YYYY-MM-DD" | null}`. `title` is 1 to
200 characters. `today` sets `scheduled_for` to today's 4am day; `backlog` leaves it null. Answers 201
`{"task": Task}`. An invalid date or `where` is 422. Scheduling a backlog item onto a specific day is
not offered here; that stays in the nightly review.

### `POST /tasks/{id}/complete` (existing)

Unchanged behavior and keys (`completed`, `day`), plus `"task": Task`. 404 when there is no active
task with that id.

### `POST /tasks/{id}/reopen`

Undoes done for today: a one time task returns to pending and its completion for today is removed; a
daily task loses today's completion. Answers `{"task": Task}`. 404 unknown, 409 when it is not done.

### `POST /tasks/{id}/backlog`

Clears `scheduled_for` and `must_do`, and removes the task's MILES block if it has one, only through the
guarded, MILES only delete path (an Evallo or non MILES event is never touched). Answers
`{"task": Task}`. 404 unknown, 409 for a daily task or a task already in the backlog.
- 409 also for a task that is done or skipped: only a pending task can go to the backlog.
- 502 when Google fails while removing the block: nothing is changed (day, must do and block id are kept), so a retry is safe.

## Nova

- Tab order: Chat, Sessions, Tasks, History, Settings. The old Tasks link in Settings is removed;
  Lists stays in Settings.
- Sections: Today, the next two days (a header per day with its label), Backlog.
- Each row: title (`short_title` when present), badges for due date, must do and has block; overdue
  shown in red.
- Actions: tap the circle to complete, tap again to reopen; swipe to Backlog (not offered on daily
  tasks or backlog items); pull to refresh; a plus button to create a task with a title, Today or
  Backlog, and an optional due date.
