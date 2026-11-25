# Sessions

Built Nov 24 2026 on `feature/sessions`, Phase 1. The shapes the app depends on
(schema, endpoints, stream events, push payload, time rules) are in
[SESSIONS_CONTRACT.md](SESSIONS_CONTRACT.md) and are changed only there, with
his approval first. This doc is the design and the reasons.

## What a session is

A chat with one job, kept apart from Nova's main conversation. Phase 1 has one
type, the nightly review: close out today, plan tomorrow. A plain chat thread
and learn sessions come later as new types, with no change to the engine or
the schema.

## Why it is not part of ask_nova

Nova's turn is built for a voice in a room: Haiku, short replies, history
trimmed to thirty words a reply, a speaker, barge in, a bridge line while a
tool runs. A session is written, keeps its whole transcript (every tool call
and result, untrimmed), has its own tools and model, and has a beginning and
an end. Sharing `ask_nova` would mean a branch on each of those, so the engine
is its own module and imports nothing from `brain.py` or the voice path. What
it does share is the policy: `tools.permits` gates every call and
`database.log_tool_call` logs it, exactly as `brain._run_tools` does.

## The pieces

| File | What it holds |
|---|---|
| `src/session_types.py` | The registry. A `SessionType` is a name, a purpose, a system prompt (a callable, so it can carry live data), its own `ToolRegistry`, an opening built in code, what to save on close, a model from `SESSION_MODELS`, and an optional `Schedule` with its push words. Also the contextvar a tool reads to find the session it is in |
| `src/sessions.py` | The engine: create, read, list, a streamed tool loop per turn, close, and the timer's `tick` |
| `src/session_schedule.py` | When to create, push, remind, close or mark missed. Pure: the time and the facts in, a decision out |
| `src/nightly_review.py` | The first type: its prompt, opening, tools and summary |
| `src/review_plan.py` | The review's plan: placement, suggestions, the week view, the plan text, the final plan |
| `src/calendar_roles.py` | Calendar roles, break days, likely cancelled classes, conflicts. Pure |
| `scripts/sessions_tick.py`, `systemd/miles-sessions.*` | The timer, every five minutes on the clock's marks |

**Each type has its own tool registry.** Nova's main registry never sees
`close_session` or `schedule_block`, so a voice turn cannot reach them, and her
capability block and the tests that pin it are untouched.

**Messages are stored whole.** Each row of `session_messages` is the list of
content blocks the API sent or received, as JSON, with tokens, model and
milliseconds on assistant rows. The next turn sends exactly what happened. An
assistant tool call and the user row holding its result are written in one
transaction, because a call stored without its result would make every later
request for that session fail.

**The opening costs no model call.** It is built from the task table, so the
first thing he reads cannot be wrong about what was on his list. The API wants
a conversation to start with the user, so the opening is sent behind a short
fixed line that is never stored.

**The system prompt depends on the session, not the clock,** so it is the same
bytes on every turn and is cached (`cache_control` on it).

**A turn is at most six rounds,** and the sixth is sent with `tool_choice`
none, so a turn always ends in words. Tools stay in that request because the
history holds tool calls.

**A per session lock** stops two turns (or a turn and a close) interleaving.

**Closing** asks the type for its summary. The nightly review writes code's
counts first (done, skipped, moved, left open; tomorrow's tasks, must dos and
blocks), then two or three sentences from one cheap model call over the
transcript. If that call fails the counts alone are saved, so the 3am close
never depends on the network. `close_session` only asks for the close; the
engine does it after the turn's tools, so the summary call is not made from
inside a tool, under the turn's own lock.

## The nightly review (v2: propose, do not interrogate)

Code does every count and every time; the model words one line and calls
simple tools. That is why it runs on Haiku (`SESSION_MODELS`).

**The flow.** The opening is built in code when the session is created (no
model call) and its proposal is saved in `sessions.plan` as JSON:

```
Today: 3 of 4 done, 1 moved to Friday.

Tomorrow, Wednesday Nov 25
• 10:40 to 11:30 AM  DSA (likely cancelled, Thanksgiving break)
• 12:30 to 2 PM  Test Prep with Jordan

Suggested must dos
1. Study for linear algebra test, 2 to 4 PM
   Due Monday
...
Optional
• Order NVMe HAT and SSD

Good, or change something?
```

- "Good" makes the model call `confirm_plan`. Code reads the calendar again
  (it may have changed since 21:30), keeps every slot still free and places
  again any that is not, saves tomorrow's tasks (must do true), books every
  must do on MILES in order, posts the final plan and closes the session the
  same way `close_session` does, closed by the user.
- A change makes the model call `set_plan` with the whole plan he now wants
  (at most three must dos; titles as shown, code keeps the ids). Code places
  the times, saves the proposal and posts the updated plan in exactly the
  opening's format.
- The other tools: `add_to_backlog`, `push_to_backlog` (out of the plan, its
  MILES block deleted, `scheduled_for` cleared), `update_task_status` for "I
  also did X", `show_week`, `move_block`, `remove_block`, `mark_attending`,
  `schedule_block` (one task or a list, placed together), `get_tasks_for_date`
  and `get_upcoming_items` (JSON with conflicts, for the model's own reading).
  Tiers: calendar writes EXTERNAL_WRITE hokage, task writes WRITE, reads READ
  hokage, close CONTROL.
- The must do cap of three is still enforced in code (`must_do_overflow`).

**Code authored messages and the guard.** A tool posts a message with
`session_types.post_message`; the engine stores it after the model's own text
as its own assistant row (`session_messages.author = 'code'`) and streams it
as `delta` events, and it is part of `done`. In any turn where code posted,
the model's own text is held back until the round's tools have run and then
cut to one short line, with every line containing a clock time (`2 PM`,
`2:30`, `14:00`, `2 to 4`, `9am`) dropped (`sessions.guard_text`). It is
applied to what is streamed and what is stored. The reason is the live
night: the model wrote a time about a block that no longer existed.
The final plan (`review_plan.render_plan`) reads must dos and optional tasks
from the database, and a must do's time comes only from fetching its event on
the MILES calendar by `calendar_event_id`; an event that is gone or cancelled
shows "not booked", never a time.

**Blocks.** `move_block` and `remove_block` look the event up on the MILES
calendar only (`calendar_tools.get_miles_event`, `move_miles_event`,
`delete_miles_event`); an id not found there is refused. Before a task with a
`calendar_event_id` is refused a new booking, the event is checked: if he
deleted it (404, 410 or cancelled) the id is cleared and booking goes ahead.

**Placement.** `review_plan.place_blocks(busy, day, items, earliest, latest)`
puts each item in order in the first free gap, fixed starts first, never
overlapping busy time or each other and never outside the window (10:00 to
21:00 by default); what did not fit is reported. Default length 60 minutes,
120 for studying for an exam.

**Moves and copies.** Moving a task copies it (the copy keeps title, notes,
must do flag, source, links, due date and minutes, not the block).

## Calendar roles, breaks and conflicts

`config.CALENDAR_ROLES` classifies calendars by id. Blocking: his primary
calendar and MILES. Context only: the club calendars (ACM, IEEE, NSBE) and the
US holidays calendar; their events show in the week view marked `(club)` or
`(holiday)` and never count as busy or conflict. Any calendar not listed,
such as the imported "Calendar", is blocking, so a new calendar counts as busy
until someone classifies it.

The roles apply inside `calendar_tools.busy_blocks`, so the voice
`check_calendar_freebusy` agrees with the review. This changed that tool on
purpose: club events stop counting as busy. Busy time is now built from the
events themselves (`reviewed_events`), not Google's freebusy, because a class
on a break day has to come out of it.

`calendar_attending` (migration 44) lists the club events or series he goes
to (`mark_attending`, or `scripts/calendar_roles.py attend|list|drop`). An
attended one is blocking.

`find_conflicts` flags timed, busy events that overlap. Touching is not a
conflict and all day events are ignored.

Breaks: `UF_BREAKS` (verify any other dates against UF's calendar before
adding them) plus any all day event on the holidays calendar. On a break day a
class (an event whose whole title is in `CLASS_TITLES`) and any recurring club
event show `(likely cancelled, <break name>)`; they are not busy, never
conflict, are never suggested, and a club event stays cancelled that day even
if he attends the series.

**Evallo guard.** His primary calendar is synced two ways with Evallo, so a
write there could cancel a real tutoring session. Every event write (the voice
edit, delete, undo and rename, the app's tap routes, `move_block`,
`remove_block`) goes through `calendar_tools._write_patch` or `_write_delete`,
which fetch the event first and refuse, before any API write, if its private
extended properties hold `evalloEvent`, `evalloSessionId` or `orgId`. It is
independent of the MILES only rule. Nothing filters Evallo events by title.

## The backlog and suggestions

A one time task with no `scheduled_for` is in the backlog (`due_date` and
`minutes` are optional columns). It is not on today's list. Voice `task_add`
sets today's day; a capture goes to the backlog ("Added to backlog: ...").

Suggestions for tomorrow, at most three, in this order: backlog items overdue
or due within three days; exams in the next seven days (calendar events whose
title has exam, test, midterm, final, quiz, OA or assessment as a word, on any
calendar but MILES and not tutoring like "Test Prep", and backlog items that
say exam or test and are due in seven days); conflicts to fix (from
`find_conflicts`); then the oldest backlog items. Each carries a reason ("Due
Monday", "Exam Monday", "Conflict to fix", "Oldest in backlog"). What is
already planned for tomorrow comes first and is never suggested twice.

## Usage metrics

`scripts/session_metrics.py` prints, per night and in total: completion rate
(reviews he closed himself over every settled one), cost per review (every
assistant turn plus the close summary, at `MODEL_PRICES_PER_MTOK`), and must
dos planned against done for days already over. Definitions are in
`src/session_metrics.py`; the numbers come from rows the sessions already
write (`closed_by` and the summary usage arrived in migration 42), so nothing
runs at night to collect them.

## Time

All naive local time, as every table stores it. The review for day D (its 4am
day) is created at D 21:30, pushed then, reminded at 22:30 only if never
opened, never pushed after midnight, and settled at D+1 03:00: missed if he
never wrote, otherwise closed with a summary. Settling compares against
`for_day`, never against "is it 3am", so a Pi that was off at three settles it
on the next run. A naive comparison is also what makes a daylight saving night
right: 03:00 is the clock reading 03:00 whichever way the clocks moved.

Only a session the timer made is pushed. One he started by hand before 21:30
is never announced, and one he has opened is never nudged. A push is marked
sent only when Apple accepted it, so one that could not go is tried again on
the next run until midnight.

## Installing the timer

Not installed by this branch. After merge, on the Pi:

```bash
sudo cp systemd/miles-sessions.service systemd/miles-sessions.timer /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now miles-sessions.timer
```

## Not built yet

The room test (TESTING.md rung 4) and a replay over real days (rung 3). The
first real night is the first run against Apple and Google.
