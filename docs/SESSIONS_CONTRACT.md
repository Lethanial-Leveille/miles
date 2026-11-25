# Sessions: the shared contract (Phase 1)

Both sides build against this file and nothing else. Neither side changes it.
A change is proposed back to Lethanial, approved, and then written here first.

## 1. Schema: migration 41 (`_migration_041_sessions_and_planned_tasks`)

```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,                       -- free text: 'nightly_review', later 'chat', 'learn_session'
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'closed', 'missed')),
    title TEXT,                               -- NULL until set; a chat type will auto title
    for_day TEXT,                             -- 'YYYY-MM-DD' 4am day for scheduled types, NULL otherwise
    model TEXT,                               -- the model that ran it
    created_at TEXT NOT NULL,
    opened_at TEXT,                           -- first GET /sessions/{id}
    last_message_at TEXT,
    closed_at TEXT,
    summary TEXT,
    push_sent_at TEXT,
    reminder_sent_at TEXT,
    UNIQUE (type, for_day)                    -- one nightly review a night; NULLs never collide
);
CREATE INDEX idx_sessions_status ON sessions (status, type);

CREATE TABLE session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,                    -- JSON list of API content blocks, stored whole
    model TEXT,                               -- assistant rows only
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    ms INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_session_messages ON session_messages (session_id, id);

-- the existing tasks table (migration 33) gains:
ALTER TABLE tasks ADD COLUMN notes TEXT;
ALTER TABLE tasks ADD COLUMN scheduled_for TEXT;          -- 'YYYY-MM-DD' or NULL
ALTER TABLE tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'done', 'skipped', 'moved'));
ALTER TABLE tasks ADD COLUMN must_do INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tasks ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'
    CHECK (source IN ('manual', 'canvas', 'study_plan', 'meal_prep'));
ALTER TABLE tasks ADD COLUMN assignment_id INTEGER;       -- table arrives in Phase 2
ALTER TABLE tasks ADD COLUMN topic_id INTEGER;            -- table arrives in Phase 2
ALTER TABLE tasks ADD COLUMN calendar_event_id TEXT;
ALTER TABLE tasks ADD COLUMN updated_at TEXT;
CREATE INDEX idx_tasks_scheduled ON tasks (scheduled_for);
-- backfill: one time tasks with any completion become status 'done'
```

Rules that come with it:
- `for_day` is nullable. It is set only for once per day types (today only
  `nightly_review`); every other type, including the later `chat` and
  `learn_session`, leaves it NULL and can have any number of sessions a day.
  SQLite never treats two NULLs as equal, so UNIQUE (type, for_day) does not
  apply to them. A test creates two sessions of one type with NULL for_day on
  the same day and expects both to exist.
- `text` stays the task's title column (no rename).
- A daily task keeps `task_completions`; its `status` stays `pending`.
- A one time task's `status` is its truth. Marking it done also writes a
  completion row, so voice and the review agree.
- Moving a task marks the row `moved` and inserts a copy on the new date with
  the same title, notes, must_do, source and links, status `pending`.
- `tasks_today` shows a one time task only while `pending` and dated today or
  earlier (or undated).
- Must dos are capped at 3 per `scheduled_for` day, enforced in code.

### Migration 42 (`_migration_042_session_metrics`, added Nov 24 2026)

Internal columns for the usage metrics; none of them is in any endpoint.

```sql
ALTER TABLE sessions ADD COLUMN closed_by TEXT
    CHECK (closed_by IS NULL OR closed_by IN ('user', 'timer'));
ALTER TABLE sessions ADD COLUMN summary_model TEXT;
ALTER TABLE sessions ADD COLUMN summary_input_tokens INTEGER;
ALTER TABLE sessions ADD COLUMN summary_output_tokens INTEGER;
```

## 2. Time

All times are the Pi's local time, America/New_York, ISO 8601 with no offset
(`2026-11-24T21:30:04.118201`, sometimes without the fraction), as every other
table stores them. This includes every timestamp in every endpoint below.

**Nova must parse them as America/New_York, not the device time zone.** A
formatter with `timeZone = TimeZone(identifier: "America/New_York")`, accepting
both with and without fractional seconds. On an iPad set to another zone, or
on a trip, the device zone would shift every time by hours. `for_day` is a
plain date (`YYYY-MM-DD`) and is shown as a date, never converted.
 The nightly review for day D is created at
D 21:30, reminded at D 22:30 if never opened, and settled at D+1 03:00:
`missed` if he never sent a message, otherwise closed with a summary. No push
is sent after midnight, and no review is created after midnight either: if
the Pi was off from 21:30 to midnight there is none that night unless he
starts one by hand.

## 3. Endpoints

All need the normal login JWT (`get_current_user`). Errors use FastAPI's
`{"detail": "..."}`.

### Session object

```json
{
  "id": 12,
  "type": "nightly_review",
  "title": null,
  "purpose": "Close out today and plan tomorrow",
  "status": "open",
  "for_day": "2026-11-24",
  "created_at": "2026-11-24T21:30:04.118201",
  "opened_at": null,
  "last_message_at": "2026-11-24T21:30:04.120931",
  "closed_at": null,
  "summary": null
}
```

`purpose` comes from the session type registry, not the table, and is never
null: a row whose type is no longer registered shows its type name. `title`,
`for_day`, `opened_at`, `last_message_at`, `closed_at` and `summary` can be
null. `closed_at` is when the session was settled, so a `missed` session has
one too.

### Message object (what the app renders)

```json
{"id": 40, "role": "assistant", "text": "Here is today...", "created_at": "2026-11-24T21:30:04.120931"}
```

Only the text blocks of a stored message, joined with a blank line. A message
with no text (a tool call or a tool result alone) is left out entirely. So is
the text of an assistant round that called a tool when another assistant round
followed its results: that is exactly the text the live stream threw away with
`reset`, so a reload reads the way the turn looked live.

### `GET /sessions?status=open`

`status` is `open` (default), `closed`, `missed` or `all`. Newest activity
first (`last_message_at`, falling back to `created_at`).

```json
{"sessions": [ <Session>, ... ]}
```

### `GET /sessions/{id}`

```json
{"session": <Session>, "messages": [ <Message>, ... ]}
```

Sets `opened_at` the first time it is called, and the returned session already
shows it. 404 for an unknown id.

### `POST /sessions`

Body `{"type": "nightly_review"}`. Starts one by hand (for testing, or when he
wants it early). Answers 201 with the same body as `GET /sessions/{id}`, the
opening message included. For a scheduled type that already has a session for
today's 4am day, answers 200 with that one instead of making a second. 400 for
an unknown type. No push is sent for a session started by hand.

### `POST /sessions/{id}/turns`

Body `{"message": "..."}`, 1 to 4000 characters. Answers `text/event-stream`
with the same events and payloads `/chat/stream` uses, so the app's stream
reader is reused:

| event | data | when |
|---|---|---|
| `delta` | `{"text": "<piece>"}` | as the reply is written |
| `stage` | `{"text": "Checking your calendar"}` | while a tool runs |
| `reset` | `{"text": ""}` | text already streamed this round is being thrown away (a tool round follows); the app clears the bubble |
| `closed` | `{"text": "<summary>"}` | the session was closed during this turn; always before `done` |
| `done` | `{"text": "<the whole reply>"}` | last event on success |
| `error` | `{"text": "<what went wrong>"}` | last event on failure |

Keep alive comments (`: keep-alive`) every 15 seconds, as now. 404 unknown id,
409 when the session is not open (before any stream starts).

### `POST /sessions/{id}/close`

No body. Closes it now and writes the summary.

```json
{"session": <Session>}
```

`status` is `closed` and `summary` is set. 409 when it is not open, 404
unknown.

## 4. Push payload

Sent by the session timer, never by a hand start. `kind` is `session`, TTL 3
hours, `apns-collapse-id` `session-<id>` so the reminder replaces the first
push on the lock screen.

```json
{
  "aps": {
    "alert": {"title": "Nightly review", "body": "3 tasks today. What got done, and what matters tomorrow?"},
    "thread-id": "session",
    "sound": "default"
  },
  "kind": "session",
  "session_id": 12,
  "session_type": "nightly_review",
  "url": "nova://session/12"
}
```

The reminder is the same shape with body `Still open. Two minutes to plan
tomorrow.` No `category`, so no action buttons.

## 5. URL scheme

- `nova://session/<id>` opens that session.
- `nova://sessions` opens the sessions list.

The push tap uses `session_id` directly; `url` is there so the same route works
from Shortcuts or a link.
