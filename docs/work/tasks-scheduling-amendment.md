# Proposed Tasks scheduling amendment

**Awaiting owner approval.** This replaces the scheduling preconditions and
the conflicting timing examples in section 3.8 of
[revision three](tasks-revision-three.md). It retains the start date storage
rules in sections 3.1 through 3.3 and pairs with backend `a9d2fda` and Nova
`43dde91`. The approved [day guard](../TASKS_CONTRACT.md), normal login
requirements and existing tool permissions continue to apply.

## Scheduling intent before any write

Code classifies scheduling clauses from the current user utterance under
the existing voice permission checks before considering model arguments or
default dates. Match the
whole clause, including qualifiers, and distinguish a recurrence, a first
day and a deadline. Model arguments may corroborate a supported clause.
They cannot change its role, shorten it, substitute a computed date or hide
an unsupported clause in `not_saved`. A genuinely undated request keeps the
existing default. If timing is present or reasonably possible and its role
cannot be established, ask for clarification with no write. Clear title
text remains title text. An isolated day token does not itself define a
start clause.

| Classification | Behavior |
|---|---|
| Supported | An untimed one time add keeps today's server task day. `daily`, `every day` and `each day` explicitly request daily recurrence. With no scheduling clause, daily starts today. A daily task may name one first day with an explicit start clause such as `starting Monday`, `starts on Monday` or `from Monday`. Explicit `starting today` stores null, including when the model omits `starts` as the tool describes. A later start requires corroboration with the original verified phrase. |
| Clarify, no write | Multiple candidate first days, a bare date in an unclear position, a qualified relative phrase such as `after Monday`, or a date whose interpretation is ambiguous under the resolver. If the model and transcript disagree or the original user text is unavailable, ask for the complete request again. |
| Reject, no write | Any recurrence other than daily, including named weekdays, weekday lists, weekdays only, intervals, alternate days, weekly and monthly rules. Also reject unsupported timing such as `starting next week`, clock times, recurrence end dates, one time voice starts, impossible dates, explicit past dates, dates beyond the horizon, and edits to an existing task's first day or recurrence. Reject a deadline in a voice add, including when a valid start is also present. |

For supported first days, keep revision three's code resolver and capture
one existing server task day for resolution, validation and storage. Today
and tomorrow are supported. A plain weekday or `this <weekday>` means the
first matching day strictly after today and is ambiguous when today already
matches. `next <weekday>` is ambiguous when its nearest upcoming match is
still in this calendar week. Keep the existing proposed month and day forms
and the recent past ambiguity rule of fourteen days. An exact ISO date must
come from the verified user clause. The inclusive horizon is today through
today plus 365 calendar days. Code speaks the stored first day using the
already proposed delivery requirements once that path is approved.

Treat `due`, deadline phrasing such as `by Friday` or `no later than Friday`,
and end qualifiers as separate from starting. A voice add containing a
deadline rejects the whole scheduling request. It cannot save just the start
or save an undated task. The existing structured one time `due_date` remains
supported and distinct from both `scheduled_for` and `start_date`. A future
structured create accepts nonnull `start_date` only for daily tasks and
nonnull `due_date` only for one time tasks. Supplying both with values returns
422 without a write. Null and absent keep the prior meaning. Impossible
structured due dates still return 422.

A clarification does not authorize a default write. This slice requires
the clarified add to contain its full scheduling intent. A bare “Tuesday”
or “yes” cannot be treated as an untimed add. A confirmed edit is not a create
request. Keep general duplicate and batch delivery policy in their existing
review items. No new task permissions, voice state framework or general
write policy enters this slice.

## Migration 046 and existing data

- Register `(46, _migration_046_task_start_date)` after Evallo 045. The
  checked backend and canonical main register through 045, and no local
  backend branch registers 046. Before building, recheck the slot. Before
  any later authorized deployment, recheck the exact release order and
  the applied 045 prerequisite. This checkpoint provides no new production
  verification or rollout authority.
- Add exactly one nullable `TEXT` column with
  `ALTER TABLE tasks ADD COLUMN start_date TEXT`. Keep the existing
  `BEGIN IMMEDIATE` runner and its version update in the same transaction.
  Check `PRAGMA table_info(tasks)` first. An already present nullable TEXT column with no nonnull default
  at version 45 is resumable. An incompatible column fails without stamping
  46. Do not add a new table, index or recurrence value.
- **No backfill or task transformation.** Existing tasks acquire NULL
  through SQLite. Preserve their existing recurrence, scheduling, due
  dates, status, history and completion rows, and preserve Evallo data.
  Do not infer starts from titles, deadlines, scheduled dates or completions.
  New starts are canonical dates for daily only and a first day equal to
  today stores NULL. Start eligibility filters reads and completion writes
  without deleting earlier historical rows. Keep the approved day guard and
  captured day behavior. Retain the existing proposed `upcoming_daily`
  projection and keep it additive for older clients.

## Focused verification and implementation order

| Synthetic case | Required result |
|---|---|
| `Add reading every day starting Monday` on Wednesday, or `starting today` | One daily task with the resolved Monday, or NULL for today. Missing model `starts` cannot turn Monday into today. |
| `Add reading every day starting next week`, `water plants on Mondays and Wednesdays`, `every other day` | Refuse. Model arguments that say daily, or omit timing, must not make these writes succeed. |
| `Starting after Monday`, `starting Monday or Tuesday`, or `next Friday` on Wednesday | Clarify with no write. A copied `Monday` or a bare follow up “yes” does not authorize a write. |
| `Add watch Friday Night Lights` | Save a normal one time task when the phrase is title text and carries no scheduling clause. |
| `Add reading every day starting Monday at five` or `starting February 30` | Refuse with no row written. Do not save a stripped Monday. |
| `Add reading every day, due Friday`, `add report by Friday`, or `daily starting Monday, due Friday` | Refuse with no write. A copied `Friday` must never populate a start. A structured create with both nonnull fields returns 422. |
| Existing task edit request, guarded add spanning 04:00, or rejected utterance | No edit through add. One captured day resolves and stores an authorized request across the boundary. For rejected requests assert task and completion rows are unchanged. |
| Migration fixtures at 44, at 45 with task and Evallo rows, and at 45 with the compatible column already present | Advance in order to 46 while preserving all rows. Test concurrent starters and rollback on an injected failure. An incompatible existing column does not advance the version. |

After approval, implement and test 046 and start eligibility with synthetic
SQLite fixtures, then the structured API. Validate the pure transcript
classifier and resolver offline
against the case table before wiring any voice writes. Actual voice write
enablement still depends on the earlier review gates for pre dispatch
mutation safety and for suppressing provisional speech. Nova's existing
day guard stays paired and unchanged by this proposal.

**Approval decisions.** This amendment proposes rejecting every voice
deadline and any mixed start and deadline rather than partially saving it.
It also requires a complete restatement after clarification. These are the
scope choices for owner approval. Supporting voice deadlines or carrying
partial intent across turns would need a separate contract decision.
