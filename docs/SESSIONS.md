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

## The nightly review

Code does every count and every time; the model words it and calls simple
tools. That is why it runs on Haiku (`SESSION_MODELS`): the cheapest model that
does the job. A stronger model earns its price later, for teaching and for
browsing.

- **Must dos are capped at three a day, in code.** `create_tasks` judges the
  whole batch and refuses it whole, saying how many slots are left, so a plan
  is never half saved. A moved must do frees its old slot.
- **Moving a task copies it.** The old row is marked moved and keeps its date,
  so each day's plan still counts what was planned for it; the copy keeps the
  title, notes, must do flag, source and links, but not the calendar block,
  which was for the old day.
- **schedule_block writes only to MILES**, through
  `calendar_tools.add_miles_event`, whose calendar id comes from the same
  lookup every other write uses. Busy time comes from `calendar_tools.busy_blocks`,
  shared with `check_calendar_freebusy`, so the two can never disagree. An
  overlapping time is refused with the free gaps listed; with no time given it
  takes the first free gap in `SESSION_BLOCK_EARLIEST` to `SESSION_BLOCK_LATEST`,
  starting on a quarter hour. The slot finding and the cap are pure functions,
  tested without Google.

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
