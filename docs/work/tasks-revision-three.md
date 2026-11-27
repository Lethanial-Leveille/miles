# Recovered Tasks revision three contract

Historical contract from November 25, 2026, section 3. The source plan SHA-256 is `9b6c24353e081fc89b650421ff9ff359c6b10da04c9183d84ed2975584abbb2d`. The proposal is preserved here as reviewed. Use the [continuation handoff](tasks-followups.md) for unresolved findings and approval gates.

Header line 3 becomes: "Approved Nov 25 2026; follow ups (day check, start date, dropped detail
guard) approved <date>."

### 3.1 Task object (replace the JSON and add bullets)

```json
{"id": 7, "title": "COP3530 Project 3: Campus Compass, due 11:59 PM", "short_title": "Project 3",
 "notes": null, "daily": false, "status": "pending", "must_do": false,
 "scheduled_for": null, "due_date": "2026-11-30", "start_date": null, "overdue": false,
 "has_block": false, "source": "manual"}
```

Add after the `has_block` bullet:
- `start_date` is the first 4am day a daily task applies to, or null. **Omitted, null, and a start day equal to today all store null** (one meaning: it applies every day, and appears in the previous day's review exactly like any other daily task). It is always null on a
  one time task. A daily task with null applies every day. A daily task whose `start_date` is
  after today is "not started": it is on no day's list before that date, cannot be completed
  for a day before it, and appears only in the overview's `upcoming_daily`.
- Completions dated before a task's `start_date` (only possible if one existed before the date
  was set, by hand or by a future edit route) are never deleted; they are simply on no list.

Migration 046 `_migration_046_task_start_date`: `ALTER TABLE tasks ADD COLUMN start_date TEXT`
(nullable, `YYYY-MM-DD`). No backfill, no CHECK (SQLite cannot add a cross column CHECK by
ALTER); code enforces "daily only". `start_date` joins `_TASK_COLUMNS`.
**It is atomic and resumable.** The migration runner (which the Evallo filter phase made atomic from
045 on) takes the writer lock (`BEGIN IMMEDIATE`) before reading the stored version, so concurrent
starters (server, voice, review timer) serialize and a second one sees 46 and skips. The migration
checks `PRAGMA table_info(tasks)` first and runs `ALTER` only if `start_date` is absent, so a run that
died after the `ALTER` but before the version write simply continues on restart. DDL and the version
update share one transaction. Tests: retry from "column present, version 45" keeps all rows and ends
at 46; two interleaved starters both succeed; the registered migration list is in order with 045
before 046; 044 to 045 to 046 on a fixture, and 045 to 046 on a fixture that already holds Evallo
verification rows, which stay intact.

### 3.2 `GET /tasks/overview?days=3` (add)

Answer gains one key:
```json
{"today": "...", "days": [...], "backlog": [Task],
 "upcoming_daily": [Task]}
```
- Today: daily tasks **whose `start_date` is null or on or before today** (rest unchanged).
- Later days: unchanged (one time tasks only; daily tasks are never listed on later days).
- `upcoming_daily`: active daily tasks with `start_date` after today, ordered by `start_date`
  then id, `status` always `pending`. Empty list when none. Clients that do not know the key
  ignore it.

### 3.3 `POST /tasks` (replace)

Body:
`{"title": "...", "where": "today" | "backlog" | "daily", "due_date": "YYYY-MM-DD" | null,
  "start_date": "YYYY-MM-DD" | null}`
- `title` 1 to 200 characters after trimming (blank is 422, as now).
- `today`: one time, `scheduled_for` = today's 4am day. `backlog`: one time, no day.
  `daily`: a daily task (`recurrence` daily, no `scheduled_for`, no `must_do`).
- **Null and absent mean the same ("no value") for `due_date` and `start_date`, on every
  destination.** A non null value that does not fit the destination is 422: `start_date` with `today`
  or `backlog` is 422 "start_date is only for daily tasks"; `due_date` with `daily` is 422 "A daily
  task has no due date". So `{"where": "daily", "due_date": null}` is valid and
  `{"where": "backlog", "start_date": null}` is valid. Nova omits hidden picker values after the
  destination changes.
- `start_date` must be a real date, not before today (422 "start_date cannot be before today,
  2026-11-25") and not more than 365 calendar days after today (422 "start_date is more than a year
  away"; exactly today + 365 is accepted, today + 366 is not). A `start_date` equal to today is
  stored as null (see 3.1).
- **Title validation order:** trim first, then the length rule 1 to 200 (blank is 422). The model
  checks length after trimming (a `mode="before"` validator), so a 200 character title padded with
  spaces is valid and a 201 character trimmed title is 422. Test both.
- Unknown fields are 422 (the model forbids extras), so a field this server does not store is
  never silently dropped.
- Answers 201 `{"task": Task}`. "Today" is read once per request and used for every check and
  the insert. Not idempotent: a retried create can make a second task (unchanged; Nova re-reads).

### 3.4 Day check, shared by complete and reopen (new subsection)

Optional JSON body `{"day": "YYYY-MM-DD"}`: the `today` of the overview the tap came from.
The handler reads the clock once; that one `today` is used for the check, the write and the
answer.
- **Legacy forms are exactly three:** no body, an empty body, or `{}`. They act on today and are
  logged at info with the task id, so stale clients can be counted. The body model forbids extra
  keys, so any other body is 422: `{"day": null}`, `{"date": "2026-11-25"}` (a mistyped key), a
  wrong type, an array. A malformed guarded tap can never become a legacy tap on the new day.
- `day` not a real `YYYY-MM-DD`: 422.
- `day` != today (earlier or later): **409**, nothing written, body
  `{"detail": "The day changed at 4 a.m. Nothing was changed. Today is Thu Nov 26.",
    "code": "day_changed", "today": "2026-11-26"}`.
- Order of checks: 422 body, then 404 unknown task, then 409 day_changed, then the route's own
  409s. Applies to daily and one time tasks alike.

### 3.5 `POST /tasks/{id}/complete` (replace)

Takes the day check. Answers `{"completed": true, "day": "<today>", "task": Task}`; `day` is the
same day the completion was stored under. 404 when there is no active task.
- Repeat on the same day: 200 again, one completion row (idempotent, as now).
- Daily task not started (`start_date` after today): **409**, nothing written,
  `{"detail": "Reading starts Mon Nov 30.", "code": "not_started", "start_date": "2026-11-30"}`.
- A retry whose first attempt succeeded before 4:00 and arrives after: when the original `day` is
  resent, 409 day_changed and the earlier tick stays stored under the earlier day (the guarantee
  applies when the client resends the original `day`; a legacy no body retry acts on the new day).

### 3.6 `POST /tasks/{id}/reopen` (replace the wording)

Takes the day check. A **one time** task returns to pending and **all** its completion rows are
deleted, whatever day they are dated (so one completed on an earlier day can be reopened). A
**daily** task loses only its completion dated today; other days are kept. Answers
`{"task": Task}`. 404 unknown. 409 "That task is not done" (one time) or "That task is not done
today" (daily), each with `"code": "not_done"`. For a daily task that has not started, reopen is **409 not_done and
deletes nothing, whatever rows exist** (a completion dated before its start date, possible only if it
existed before the date was set, is kept and listed nowhere). Test both conditions on one fixture:
Reading with start Nov 30 and a Nov 29 completion, reopened on Nov 29, answers 409 and the Nov 29
row is still there.

### 3.7 `POST /tasks/{id}/backlog`, `GET /tasks/today`

Backlog unchanged. `GET /tasks/today` leaves out daily tasks that have not started. **One captured
New York task day per request:** the handler reads the clock once, and that one day is the response
label and the day used by every query, including `upcoming_daily` and the overview; a request that
spans 4:00 can never return day Nov 29 with Reading (which starts Nov 30). Test a single request
crossing 4:00 with an injected clock.

### 3.8 Voice and review (new section "Voice tools")

Dates in tools are 4am days. Every write rule above holds for voice through the same
database functions (`complete_task` refuses a not started daily task for every caller).

**The model never computes a date, and never decides on its own which words are the start day.**
`task_add` schema:
```json
{"task": {"type": "string", "description": "The task, in his words"},
 "recurrence": {"type": "string", "enum": ["daily"], "description": "daily when it repeats every day; omit for a one time task"},
 "starts": {"type": "string", "description": "The start day exactly as he said it, copied from his words ('Monday', 'next Monday', 'December 1', 'tomorrow'). Never convert to a date, never work out a weekday. Only for a daily task that starts later. Omit when it starts today."},
 "not_saved": {"type": "string", "description": "Anything in his request this tool has no field for: a time, a place, a person, an end date. Empty when everything fit. Fill it rather than leave a detail out."}}
```

**The tool reads his words.** The brain passes the turn's user text (his transcript) to `task_add`
and the other task tools (a small brain.py change; if a tool cannot get it, it saves nothing). Code,
not the model, finds the day words in it:

`extract_day_words(utterance) -> list of spans`: weekday names with an optional `this` or `next`
(`Monday`, `Mon`, `next Monday`), `today`, `tomorrow`, month and day forms (`December 1`, `Dec 1st`,
`1 December`, `12/1`) and ISO dates, all matched on whole words, case folded, whitespace collapsed.
`extract_repeat_words(utterance)`: `every <weekday>`, `weekly`, `every other`, `monthly`, `every N
days` and similar repeats other than daily.

Preconditions, checked in this order, each saves NOTHING and says why in code written text:
1. **A repeat other than daily** in his words (`extract_repeat_words` non empty, or `every <weekday>`):
   saves nothing ("Nothing was saved: a repeat other than every day is not supported. Ask what he
   wants."), whatever the model passed.
2. **`starts` must be one of his own spans.** If `starts` is non empty and, after normalization, is
   not exactly one of the spans `extract_day_words` found in his utterance, nothing is saved
   ("Nothing was saved: 'December 1' is not what he said. Ask him for the start day again."). This
   refuses a model supplied ISO date or a shortened phrase ("Monday" when he said "next Monday",
   "2026-12-01" when he said "Monday"): ISO is accepted only when his words contained that ISO text.
3. **No uncovered day word.** If his utterance contains day words (a span from `extract_day_words`)
   that `starts` does not account for, nothing is saved ("Nothing was saved: I heard 'Friday' but it
   was not given as the start day. Ask which day he means."). This is what stops "add the dentist
   starting Monday" (one time) from being saved today with the day pushed into `not_saved`, and
   "add reading every day, due Friday" from guessing.
4. More than one day word span (two different spans) is ambiguous: nothing saved, ask.
5. **`starts` without `recurrence: daily`** saves nothing (see below).
6. **No duplicate:** if an active task with the same normalized title exists, `task_add` saves
   nothing ("Reading is already a task; changing its start day or repeat is not supported yet.").
   This is what "make reading start Monday" gets (never a second Reading).

Case table (one database outcome and one spoken outcome each):
| He says | Database | He hears |
|---|---|---|
| "add reading as a daily task starting Monday" (Wed Nov 25) | one daily task, `start_date` Nov 30 | "Added the daily task: reading, starting Monday Nov 30." |
| "add reading every day" | one daily task, null start | "Added the daily task: reading." |
| "add the dentist starting Monday" | nothing | precondition 3 or 5: asks whether it is daily or for a day |
| "add call mom at 5 pm tomorrow" (one time) | nothing (a day word with no `starts`) | "Nothing was saved: I heard 'tomorrow' ... Ask which day he means." |
| "add reading every day starting Monday at five" | daily, start Nov 30 | the start line plus "NOT saved: at five. Tell him this." |
| "add the report every Tuesday" | nothing | repeat other than daily |
| "make reading start Monday" (Reading exists) | nothing, no second row | the duplicate sentence above |
| "add run every day starting Friday" said on a Friday | nothing | ambiguous (today or a week from today), asks |

**Resolving `starts` (code, America/New_York, relative to the 4am day `today`)**, in one function
`resolve_start_phrase(phrase, today) -> (date | None, problem | None)` with no model involved. The
phrase (already verified to be one of his spans) is trimmed, lower cased, extra spaces collapsed,
and a leading "starting", "starts", "from" or "on" and a trailing "please" dropped; **a non empty
phrase that becomes empty after that is `unreadable`, never "today"**. Rules, exhaustive; anything
else is `unreadable`:
- An absent or null `starts` means it starts today (stored as null). The explicit word `today` also
  stores null. `tomorrow`: today + 1.
- A weekday name alone (`Monday`, `this Monday`, `Mon`): the first such weekday strictly after today.
  **If today is that weekday it is ambiguous** (today or in a week): saves nothing and asks.
- `next <weekday>`: the first such weekday strictly after today **only if that date is in a later
  calendar week than today (weeks start Monday)**; if the coming one is still in this week
  (`next Friday` said on a Wednesday) it is ambiguous (this week's or the week after), saves nothing
  and asks which date.
- A month and day without a year (`December 1`, `Dec 1st`, `1 December`, `12/1`, month first): **one
  rule: the next occurrence on or after today, unless its previous occurrence was within the last 14
  days, in which case it is ambiguous** (he probably meant the recent date; saves nothing and says
  the date already passed). So on Nov 25: `December 1` is Dec 1 2026; `January 3` is Jan 3 2027;
  `November 20` is ambiguous (5 days ago); Dec 31 then `January 1` is Jan 1 next year. An impossible
  date (`February 30`) is unreadable. `February 29` resolves to the next Feb 29 and is refused when
  that is more than 365 days away. Today's date means today (null).
- An explicit date with a year in the past is refused. ISO dates follow the span rule above.
- `next week`, `in a few days`, `soon`, a weekday with a time of day, or anything else: `unreadable`.
- The resolved date must be today or later and **at most 365 calendar days after today** (today + 365
  accepted, today + 366 refused), else nothing is saved.
- Every problem result says what was understood and why it cannot be used, in a sentence code writes
  (for example "Nothing was saved: 'Friday' is ambiguous because today is Friday, so it could mean
  today or Dec 4. Ask which day he means.").
Results (exact text; `<day>` is code generated, e.g. "Monday Nov 30"):
- one time: `Added the task: <task>.` daily: `Added the daily task: <task>.` (fire and forget)
- daily with `starts` resolved: `Added the daily task: <task>, starting <Monday Nov 30>.` The weekday
  and date are written by code from the resolved date and **spoken by code (see `Spoken` below)**, so
  what he hears is the date the task will actually start on.
- `starts` without `recurrence`: nothing saved: `Nothing was saved: a start day is only for daily
  tasks, and a one time task said aloud goes on today's list. Ask whether it is daily or for today.`
- `starts` ambiguous, unreadable, in the past or more than a year away: nothing saved, with the code
  written reason from the rules above and `Ask which day he means.`

**Delivery (the `Spoken` result).** `Spoken` is renamed `Spoken`: a tool result whose text is
delivered to him **verbatim by code** (the brain speaks the result text itself; the model does not
paraphrase it), not as an error, and logged with `is_notice`. `task_add` returns `Spoken` for a saved
dated add, for a NOT saved line, and for every refusal above. After a `Spoken` result, in the same
turn the brain **refuses a second `task_add` and any other task write** unless there is new user
input (a repeated write returns a plain refusal the model cannot turn into a second row), and a
refusal is never followed by a guessed write. Tests, in a mixed tool turn (a task_add refusal in the
same turn as another staged action): the emitted speech is exactly the code text, the row count does
not change, a second task_add in the turn writes nothing, and the room test checks nothing was said
before the call (the tool descriptions say "say nothing before or alongside the call").

Dropped detail guard, on `task_add`, `task_complete`, `archive_task`, `list_add`, `list_check`:
- Each gains optional `not_saved` (string). Trimmed; empty or whitespace means none; cut to 300
  characters.
- Non empty, after a successful write: the result is a **Spoken** result (delivered by code, not an error):
  `<normal result> NOT saved: <not_saved>. Tell him this, and do not call <tool> again for it.`
- Non empty when the tool saved nothing (no match, several matches, already on the list,
  rejected date): the notice text gets the same `NOT saved: <not_saved>.` sentence appended.
- Empty: exactly today's result, no NOT saved line.
- Each description gets: "If you fill not_saved, say nothing alongside the call; the result
  tells you what to say."
- `Spoken` (new, tools.py, replaces the earlier `Spoken` name): a return wrapper that makes the
  brain deliver the text itself and makes `_needs_second_call` true without `is_error`, logged with
  `is_notice`. Not an error, so the model does not retry; the repeat write rule above applies.

`tasks_today` (voice) and `GET /tasks/today`: not started daily tasks are left out.

Nightly review: `tasks_for_day(day)` uses **the requested day** for eligibility: it leaves out a
daily task whose `start_date` is after `day` and includes it for `day` on or after the start, so
`get_tasks_for_date`, `today_line`, `push_body` and `summary_facts` neither show nor count it before
its start. On Nov 29 a review query for Nov 30 includes Reading, although today's voice list and
`GET /tasks/today` (which use today) exclude it, and the overview's Nov 30 section still lists only
one time tasks.
`update_task_status` done on a not started daily task: ToolNotice `<title> starts <day>; nothing
was marked.` Voice `task_complete`: same notice (it still matches by name, so the answer is
"starts Monday", not "no task matches").

### 3.9 Nova section (add bullets)

- Every complete and reopen sends `{"day": ...}` where the day is **bound to the rendered row** (the
  `today` of the overview that produced that row, stored on the row model when it is rendered), not
  the latest `overview.today`, so a concurrent refresh cannot supply a newer day to an old row. On 409 `day_changed` Nova reloads
  and shows "The day changed at 4 a.m. Nothing was changed." On 409 `not_started` it shows the
  detail and reloads.
- Add sheet: Where is Today | Backlog | Daily. Daily hides the due date and shows "Starts later"
  with a date picker limited to tomorrow ... today + 365 (in the overview's `today`, not the
  phone's date). Off sends `start_date` null.
- A "Starts later" section after Backlog lists `upcoming_daily`, each row with a "Starts Mon Nov
  30" badge and no completion circle; hidden when empty.
