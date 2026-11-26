# Evallo session id filter: plan and contract (revision 5)

## Context

Evallo's two way sync writes the whole organization's tutoring sessions onto his primary Google
calendar: 745 events since Aug 2026, about 52 of them his. MILES reads that calendar for busy time,
the week view in Nova, conflicts, nightly review suggestions and voice answers. So other tutors'
sessions make him look busy, create false conflicts and get read to him. The sync must stay on, and
those events can never be written to (the guard in `calendar_tools._guard_not_evallo` already
refuses writes).

The fix, approved Nov 25 2026 (ROADMAP "Evallo rules (permanent)", next in the roadmap order after
nightly review v2, which is merged): the Mac reads his own sessions from Evallo's API and sends only
their session ids to the Pi. On the Pi, an Evallo event that is not his is ignored on every read path.
**Section C4 is the single rule for what counts as his; wherever this document says "ignored" or
"kept", C4 decides**, including its exceptions (an event with no usable session id, an event booked
after the last snapshot, and the title fallback). The exceptions are recorded in docs/ROADMAP.md.

## Current state (verified Nov 25 2026)

- Pi `~/miles` main at `3bc3722` (docs only on top of `8f2979f`), review v2 merged. Latest migration
  `_migration_044_review_v2`.
- Every calendar read goes through `calendar_tools._fetch_events(queries)` (src/calendar_tools.py:507):
  `events_for_app` → `reviewed_events` → `busy_blocks`, the nightly review and review_plan,
  `get_upcoming_events` and `_already_over` (voice), and `find_schedule_conflicts`.
- `EVALLO_KEYS = ("evalloEvent", "evalloSessionId", "orgId")` (calendar_tools.py:152). The Evallo
  `_id` equals the Google `evalloSessionId`.
- The Mac agent is `~/PersonalProjects/miles-evallo-mac`, **now a git repo** (baseline commit
  `48e813c`, 56 source files, `data/`, logs and logins ignored, `data/evallo_state.json` untracked). It
  has `src/miles_client.py` (`Miles` client, agent token in the Keychain), `src/evallo_session.py`
  (`load_state`, `token_expiry`, `days_left` on `data/evallo_state.json`), and `/agent/login-expired`.
- The Pi already has `get_agent_user` (server.py:750): a full login or the scoped agent token, which
  can never approve.
- A Mac probe found `api.evallo.org/api/events/summary` returns `$.data.allEvents[]` with `_id`,
  `tutorId`, `tutorName`, `date`. Logged in as him, it returns only his sessions. His `tutorId` is the
  `id` claim in his `evalloToken`. Pagination, date parameters, timezone and the canceled field are
  **not yet known**: the Step 3 probe settles them before any builder starts.

## Contract

### C1. Which events are Evallo events

An event is an Evallo event when its `extendedProperties.private` holds any of `evalloEvent`,
`evalloSessionId` or `orgId`. Its session id is `private.evalloSessionId` when that is a string
matching `^[0-9a-f]{24}\Z` (full match, a trailing newline is invalid). Anything else (missing, empty, wrong shape) means the event has **no
usable session id**.

### C2. Verified set on the Pi

Migration 045 adds two tables:

- `evallo_verified_sessions(session_id TEXT PRIMARY KEY)`
- `evallo_verification(id INTEGER PRIMARY KEY CHECK (id = 1), fetched_at TEXT NOT NULL,
  pushed_at TEXT NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL,
  count INTEGER NOT NULL, active INTEGER NOT NULL DEFAULT 0)`

`fetched_at` is **Evallo's own server time of the earliest moment the data could have been read**: the
`Date` header of the response to the run's first summary request, minus the elapsed time on the Mac
(monotonic clock) from sending that first request to receiving the last response of the run (UTC, ISO
with `Z`, one second resolution, rounded down). It is never the Mac's wall clock. Evallo's API sends a
`Date` header (checked Nov 25 2026: nginx, `date: Wed, 25 Nov 2026 19:10:37 GMT`, the same second as
the Mac's clock). `pushed_at` is the Pi's clock at receipt. `window_start` and `window_end` are dates (inclusive
start, exclusive end) in America/New_York: the range the Mac asked Evallo for **and** confirmed
complete (C6). A push replaces the whole set and the status row in one transaction. A failed push
leaves the previous set untouched.

**The filter is on only when all of these hold: there is a status row, the state was read, and
`active = 1`.** `active` is **stored in the database, so every process that reads the calendar (the
server, room voice, the nightly review) sees the same value.** An accepted push (the route answered
200, which includes an idempotent retry of the stored snapshot, C3) sets `active = 1` in the same
transaction as the replacement or retry. **`miles-server` sets `active = 0` once, at startup, before
it serves requests.** It is never derived from comparing timestamps (a clock correction cannot turn it
on), and the other processes never write it. Otherwise the filter is off: every Evallo event is kept
exactly as today (safe: busy, never free). So the filter is off after any `miles-server` restart,
including a redeploy that finds an old status row, until the first push accepted after the restart.
A restart of only voice or the review changes nothing. Nothing else turns it on. `count` must equal
the number of stored ids; if it does not, the state is `unreadable` (C4).

### C3. `PUT /agent/evallo/sessions`

- Auth: `get_agent_user` (agent token or full login). The response is 401 for no token, a bad token
  or a device scoped token.
- Body (pydantic, `extra="forbid"`, so any extra field such as a token is a 422):
  ```json
  {"fetched_at": "2026-11-25T18:00:03Z", "window_start": "2026-11-18", "window_end": "2027-01-24",
   "session_ids": ["<24 hex>", "..."]}
  ```
  - `session_ids`: strings matching `^[0-9a-f]{24}$`, at most 2000, no duplicates. Any one that fails
    makes the push a 422.
  - `window_start < window_end`, the window is at most 120 days, and it includes today
    (America/New_York). Otherwise 422.
  - `fetched_at` must parse as a UTC timestamp. It is truncated to whole seconds on receipt, and every
    comparison (ordering, equal, window checks) uses the truncated value. **422 if it is more than 5 minutes ahead of the Pi's
    clock** (exactly 5 minutes is accepted). 422 if it is more than 24 hours behind the Pi's clock.
    This is a sanity check at receipt; the bound that matters is enforced by the Mac when the snapshot
    is taken (C6).
  - **Ordering, checked inside the replacement transaction (`BEGIN IMMEDIATE`)**: 409 with the top level body
    `{"reason": "older_snapshot"}` if `fetched_at` is older than the stored one. If it is **equal**, the
    push is an idempotent retry and is accepted (200, nothing changes in the tables except that `active` becomes 1, C2) only when its window and its id set are identical to the stored ones; otherwise
    409 with the top level body `{"reason": "conflicting_snapshot"}`. A newer `fetched_at` replaces everything.
  - An empty `session_ids` list is valid. It means he has no sessions in the window.
  - **Validation errors never echo submitted values**: a 422 carries only the field path and the error
    type. A session id must match `^[0-9a-f]{24}\Z` (a trailing newline is invalid).
- A failed response never activates the filter: `active` becomes 1 only after the transaction
  committed and the 200 is about to be returned, and logging happens after.
- 200: `{"stored": <count>, "fetched_at": "<iso>", "pushed_at": "<iso>"}`.
- The Pi never logs the body. It logs only the count, the window and `fetched_at`.

### C4. Read filter (Pi): the single rule

One pure function, `evallo_filter.keep(event, state, now) -> bool`, is applied to the raw Google
items inside `_fetch_events` on every read, so no read path can bypass it. `state` comes from one
database transaction that reads the ids and the status row together, and it carries the outcome
(`active`, `stale`, `off` or `unreadable`, see C5). Each top level read uses the state it loaded for
its events and reports that same state's outcome. Non Evallo events are always kept. If the filter is
off or the state is unreadable (the failure is logged, and non Evallo events are unaffected), every
event is kept. The state is unreadable also when the stored `count` differs from the number of stored ids.
**The filter itself never raises and never drops a whole calendar:** if evaluating any one event
fails, that event is kept (and the failure is logged without its contents). Otherwise, for an Evallo
event, in this order:

| # | Case | Result |
|---|---|---|
| 1 | No usable session id (C1) | keep |
| 2 | Session id is in the set (always, even if the set is stale or the event is outside the window) | keep |
| 3 | The event's `updated` time is later than `fetched_at` minus 15 minutes, or `updated` is missing or unparseable. It may have been booked after the snapshot, so whose it is cannot be known yet. | keep |
| 4 | Set **fresh** (`now - fetched_at <= 24h`), the event's start date is inside the window, and the id is not in the set | drop |
| 5 | Otherwise (set stale, or the event is outside the window or its start is unusable): his full name is in the tutor position of the title | keep |
| 6 | Otherwise | drop |

The 15 minutes in row 3 are a margin for any disagreement between Evallo's clock and Google's (Google
stamps `updated`), widened from 7 on his decision. It is the only cutoff, and it is used the same way
in the tests. A wider margin only keeps more events, so it errs toward busy.

**Residual risk (accepted):** nothing in this design reads Google's clock. If Evallo's clock and
Google's disagree by more than about 13 minutes in the unsafe direction (Evallo's ahead), a session he
books after a snapshot can fall before the cutoff and be dropped (look free) until the next
successful snapshot (normally within the hour). Only the Mac's own clock is compared with Evallo's
(C6, within 5 minutes), which catches a badly wrong Evallo clock only when the Mac's is right. A booking
made between the data read and the stamp is covered by C6's elapsed time subtraction, within the
60 second limit.

The event's start date: a timed event uses the America/New_York date of `start.dateTime`, converted
from its own offset or `start.timeZone` (the event's zone, not its wall clock date); an all day
event uses `start.date`. A missing or unparseable start is "unusable" and takes row 5 (never row 4).

Row 5 applies to a fresh set too, for an event outside the window or with an unusable start. That is
an exception the owner decided on Nov 25 2026, wider than the permanent rule (which grants the
fallback only for a stale set), and it is recorded in docs/ROADMAP.md under the Evallo rules.

The full name rule matches, case insensitive, treating any run of whitespace (including a
non breaking space) as one space, with `Lethanial Leveille` in full:
`<student> <> Lethanial Leveille | ...` and `<student> / Lethanial Leveille`. A first name alone never
matches (`Test Prep | Lethanial <> X` does not count). A name in the student position does not count.

**Row 3 lasts until a successful snapshot's cutoff reaches or passes the event's `updated` time** (an
event updated at 10:58 is still kept by a successful 11:00 snapshot, whose cutoff is 10:53, and is
dropped by the next one). As long as pushes keep failing it lasts that long, and an event whose
`updated` is missing or unparseable stays kept for good. Evallo's own sync touching another tutor's event bumps its
`updated`, so such an event can reappear until the next successful push. All of these err toward
busy, never free, and are accepted costs.

**Uncertain events are returned anonymized.** An event kept only because of uncertainty (row 1: no
usable session id; row 3: updated after the snapshot, or `updated` missing or unparseable) **and whose
title does not match the full name rule** is returned by every read path, including direct calendar
lookups, as an anonymous busy block: title `Evallo session (unverified)`, and no description, location,
attendees, organizer, conference data or any name. It keeps only its id, start, end, `transparency`,
recurrence ids, its Evallo private extended properties (the write guard needs them), and a reduced
attendee list holding nothing but the owner's own response status (so declined still works). Events
kept by row 2 (id in the verified set) or row 5 (full name) keep their titles and details: they are
his. Nothing from an anonymized event is ever saved in `tool_call_log` or any other table, because it
never carries a name.

**Kept does not override availability.** A kept event still goes through the ordinary flags
downstream: an event marked transparent, or one he declined, is not busy time on any path (week view,
`busy_blocks`, voice, conflicts), exactly as for any other event. The filter only decides whether the
event exists for MILES at all.

A dropped event is gone from busy time, the week view (`/calendar` and `events_for_app`), conflicts,
suggestions, voice listings and `_already_over`. Plans stored earlier are not rewritten; they are
rebuilt on the next review. Writes are unchanged, and `_guard_not_evallo` still refuses any Evallo
event.

### C5. Status shown to him

- `evallo_filter.status` comes from the same state as C4 and returns
  `{"outcome": "active"|"stale"|"off"|"unreadable", "verified_at": iso|None}`. `verified_at` is
  `fetched_at` when there is a status row. `unreadable` is a failed state read. `off` is no row, or `active = 0` (C2). `stale` is a row older than 24h while the filter is on.
  **Precedence: `unreadable`, then `off`, then the freshness check.** So a restart with a 25 hour old
  row and the Mac offline is `off`: every event is kept, never run through the title fallback.
- Anything other than `active` shows one code generated line: "Evallo sessions not verified since
  <Mon Nov 24, 3:10 PM>" (or "Evallo sessions never verified", or "Evallo sessions could not be
  checked" for `unreadable`). The nightly review's opening and voice `get_upcoming_events` add it,
  using the outcome of the read that produced their events, never a second read. The Nova app
  contract does not change.

### C6. Mac push job

- New file `src/evallo_sessions.py`, `scripts/evallo_push_sessions.py` and a launchd plist
  `com.lethanial.miles-evallo-sessions.plist` that runs **every hour** (`StartInterval` 3600) and at
  load. Every deploy plan that restarts the Pi also runs this job once right after the restart.
- **Pinned owner.** Pinning is an explicit deploy step that happens **before the plist is loaded**.
  `scripts/evallo_pin_tutor.py` takes the tutor id **he supplies** (read from his own Evallo profile
  page, independently of the saved login; passed on stdin or from a mode 600 file, never in an
  argument and never echoed) and works in two steps. Step 1 (`--show`) writes nothing: it checks that
  the saved login's `id` claim equals the id he gave, and that the `tutorId` of his sessions in the API
  response equals it too, then prints the full name **exactly as Evallo returns it** (`tutorName`).
  Step 2 (`--confirm-name "<that exact name>"`, run only after he has confirmed the name) writes
  `~/.config/miles-evallo/tutor_pin` (mode 600, holding the id and that name) and refuses unless the
  name equals what Evallo returns. On a mismatch it writes nothing and exits non zero, saying only that they
  differ. If the token has no `id` claim, or its `id` differs from the pin, the job pushes nothing,
  even when the response is empty, and reports a failure (a different tutor's login must never replace
  his set). The plist cannot be loaded while the pin file is missing (the deploy step checks it).
- **HTTP client.** One client per run with **cookies never stored or resent** and redirects off. The
  Evallo token is sent only in the `authorization` header on requests to `api.evallo.org`.
- **Pin file.** Written with an exclusively created, unique, mode 600 temporary file in the same
  directory, then atomically renamed over the pin (no symlink is followed, no `O_TRUNC` of an existing
  file). Every run opens the pin without following symlinks and refuses it unless it is a regular file
  owned by the current user with mode 600. The pin script reads the id with hidden input on a
  terminal, or from stdin or a mode 600 file when piped, and no documented command puts the id in a
  shell command line or echoes it.
- **Loading the job.** `scripts/evallo_load_job.sh` runs `scripts/evallo_check_pin.py` and loads the
  plist only if it passes. The plist is never loaded any other way.
- **Second check on every run.** Every row that is pushed must also have `tutorName` equal to the
  pinned full name (trimmed, case folded; the probe records Evallo's exact form). If any pushed row
  does not, push nothing and report it.
- **Gate: no builder starts until the Step 3 probe has recorded, in this document, (a) the API's date
  range parameters and timezone, (b) how pagination or truncation shows in a response, (c) the
  canceled field, (d) what an empty result looks like, (e) whether any of his real sessions are
  marked transparent or declined, and (f) the exact form of `tutorName` on his sessions, and that the
  authenticated summary response carries a `Date` header.** If (e) finds any, stop and tell him before building. The job must
  be able to prove a response is complete. If the response signals more pages or a possible cut off
  (per (b)), it pushes nothing.
- Steps: load `data/evallo_state.json`, read `evalloToken` in memory only, and check the `id` claim
  against the pin. Call the summary API for the window (today − 7 days to today + 60 days).
  **`fetched_at` is the first response's `Date` header minus the elapsed time** on the Mac's monotonic
  clock from sending the first request to receiving the last response of the run (all pages). **If the
  elapsed time is over 60 seconds, push nothing and report it.** If the header is missing or does not
  parse as one complete HTTP date (`Wed, 25 Nov 2026 19:10:37 GMT` and nothing after it), push
  nothing and report it (the Mac's wall clock is never the stamp). **Elapsed time uses a clock that
  keeps counting while the Mac sleeps, and the 60 second budget is enforced during the run** (each
  request's timeout is the remaining budget; a run over budget stops and reports). **Clock witness:** the
  Mac compares that `Date` with its own clock when the response arrives, and if they differ by more
  than 5 minutes it pushes nothing and reports it. Validate, in this order:
  0. Envelope: every response (the list and each weekly one) must have exactly the keys `success`,
     `data`, `message`, with `success` equal to `true` and `data` holding exactly `allEvents`. Any other
     key (a `hasMore`, `nextPage`, `total`, a cursor), `success` not true, or an unrecognized shape is
     a possible cut off: push nothing and report it.
  0b. Completeness: **every row of every response is validated like step 1 (not only the list
     response)**, then each response is reduced the same way (owner, then not canceled), and the
     reduced id set of the list response must equal the union of the reduced id sets of the weekly
     responses. A disagreement in any id, owner or status means push nothing and report it. No
     response may have 200 rows or more.
  1. Shape: `data.allEvents[]` exists, and every row has `_id` matching `^[0-9a-f]{24}\Z` and a
     non blank string `tutorId`. Any row that
     does not is a shape error: no push.
  2. Owner: keep rows whose `tutorId` equals the pinned id. If the response has rows but none are his,
     it is a shape error: no push. If the response has no rows at all, the result is an empty set.
  3. Canceled: drop rows with `sessionStatus == "CANCEL"`. If all his rows were canceled,
     the result is an empty set (a valid push).
  4. Push the `_id`s with `Miles._call("PUT", ...)`, with `fetched_at` and the window. **The job
     reports success only if the Pi's answer has `stored` equal to the number of ids sent and the same
     `fetched_at`; anything else is a failure.** The window sent is `window_end` exclusive, and the
     requests ask through the last day before it.
- Failure rules:
  - Token missing or expired, or Evallo answers 401: send `/agent/login-expired` (the existing once a
    day note) and stop with no push and no retry.
  - Any other error, an incomplete response, a shape error or an identity mismatch: stop with no push
    and record a failure with `report_job`. The Pi keeps the last set. A 409 `older_snapshot` from the Pi is
    not an error: log it and stop. A 409 `conflicting_snapshot` is a failure and is reported.
- The token, cookies and raw API responses are never printed, logged or sent. The logs show counts,
  the window and `fetched_at` only.

### C7. Tests (must exist and pass before review)

Pi (`src/tests/test_evallo_filter.py`):
1. `keep` covers each row of the C4 table, including the 24h boundary, the window edges and the 15
   minute cutoff (an event updated exactly at the cutoff, and one second either side).
2. **A session booked after the last push is kept, with a fresh set and with a stale set**, including
   when Evallo's clock was ahead of Google's (an event updated 14 minutes before `fetched_at` is
   kept; one 16 minutes before is not). An
   unlisted event updated before the cutoff, **with a usable start inside the window**, is dropped when
   the set is fresh (even with his full name in the title, only if it is inside the window; test 3
   covers outside), and kept or dropped by the full name rule when the set is stale.
3. Full name patterns: both formats match, first name only never matches, a name in the student
   position does not match. A fresh set keeps a full name event outside the window.
4. Ids and dates: an empty or malformed `evalloSessionId` is kept; a timed event just after midnight
   UTC uses its New York date; an all day event uses `start.date`; an unusable start never takes the
   fresh drop row; a missing `updated` is kept.
5. **Delayed delivery:** a push delivered 30 minutes after it was fetched is accepted, its
   `fetched_at` is Evallo's time, and a session booked after that `fetched_at` is kept by row 3
   whenever it is read, with a fresh set and with a stale one. A `fetched_at` 4 minutes ahead at receipt
   is accepted and 5 minutes 1 second ahead is refused.
6. With no status row, every Evallo event is kept. If the state read raises, every event is kept and
   nothing else changes. **After a simulated process restart the filter is off until a push is
   accepted by the new process: deploy, push, rollback, redeploy with the Mac offline keeps every
   Evallo event. After a restart, an idempotent retry of the stored snapshot (answered 200) turns the
   filter on. A restart with a 25 hour old row and no push gives `off`, never `stale`, and keeps every
   event. A backward clock correction cannot turn the filter on.**
7. `_fetch_events` with the fake Google service: other tutors' sessions are gone from `busy_blocks`,
   `events_for_app`, `find_schedule_conflicts`, `get_upcoming_events` and `_already_over`, and his own
   remain, checked by specific ids and by busy or free answers for known slots. A kept event marked
   transparent or declined is not busy on any path, and a kept normal event is busy on all of them.
8. The PUT route: 401 without a token and with a device token; 422 on an extra field (`token`), a bad
   id, a duplicate, more than 2000 ids, a bad window, a `fetched_at` more than 5 minutes ahead of the
   Pi clock (exactly 5 accepted) and one more than 24 hours old; 409 when an older `fetched_at`
   arrives after a newer one, with reason `older_snapshot`; **an equal `fetched_at` with the same window
   and ids is a 200 no op, with different ids or window a 409 with reason `conflicting_snapshot`, and a
   delayed retry of A after B cannot restore A**; replace is
   atomic and a failed push keeps the old set; the first accepted push after the restart turns the
   filter on.
9. The nightly review and voice notes: appear for `stale`, `off` and `unreadable` with the right
   wording, absent for `active`, and a voice answer whose events came from a failed read carries the
   `unreadable` note even if a second read would succeed.
10. The existing Evallo guard tests still pass, and the full suite stays green.
11. **Revision 5:** `active` is read from the database by separate connections or processes (a
    second process sees the server's accepted push, and `miles-server` startup clears it, and a
    voice or review restart does not); a `count` mismatch is `unreadable`; the filter raising on one
    event keeps that event and the rest of the calendar survives; anonymized events (row 1, row 3, not
    full name) carry no title, description, attendees or names on every read path including direct
    lookups, and a row 5 or row 2 event keeps its title; nothing from an anonymized event reaches
    `tool_call_log`; ids with a trailing newline are row 1 and a 422 on the PUT; an event in another
    time zone uses its New York date; a non breaking space in the name matches; a 409 body is
    top level; fractional `fetched_at` ordering uses whole seconds; a failed log write after the
    commit does not leave the filter on for a failed response; migration 045 is atomic.

Mac (`src/tests/test_evallo_sessions.py`):
1. The payload keys are exactly `fetched_at, window_start, window_end, session_ids`, and the token
   string appears nowhere in the payload or the captured logs.
2. Only rows with a matching `tutorId` that are not canceled are sent.
3. A 401 calls login-expired and does not push. A shape error does not push.
4. A partial page that still contains a matching row does not push (using the completeness signal
   from the probe).
5. The four empty cases: no rows at all pushes an empty set; all his rows canceled pushes an empty
   set; foreign rows only does not push; one good row plus one row missing `_id` does not push.
6. A 409 `older_snapshot` is logged and is not reported as a failure; a 409 `conflicting_snapshot` is
   reported.
7. **A token whose `id` differs from the pin, or has no `id` claim, pushes nothing, including when the
   response is empty.** The pin script writes mode 600 and prints no full id, **refuses to write when
   the id he supplied differs from the saved login's `id`**, and the job refuses to run (and the plist
   check fails) while the pin file is missing.
8. **If any row that would be pushed has a `tutorName` other than the pinned full name, nothing is
   pushed** (and a single row with a different name among his rows is enough).
10. **Revision 5:** exactly one Evallo request on a 401 (no retry); rows with a bad `_id` (not 24
    lowercase hex, a trailing newline, a token string) or a blank `tutorId` push nothing, in the list
    and in a weekly response; `success: false` pushes nothing; any extra envelope key (`hasMore`)
    pushes nothing; a status disagreement between the list and the weekly responses pushes nothing;
    a token inside a response row never reaches the PUT body, the logs or the report; a Pi answer with
    the wrong `stored` or `fetched_at` is a failure; a `Date` with trailing text is refused; a cookie
    set by a response is never sent back; the pin write refuses to follow a symlink and is atomic;
    the elapsed clock keeps counting during a simulated sleep; the load script refuses without a pin.
9. **`fetched_at` is the response's `Date` header minus the elapsed time**, not the Mac's clock: a Mac
   clock 10 minutes wrong with a correct `Date` pushes nothing (witness), a missing or unparseable
   `Date` pushes nothing, an elapsed time of 61 seconds (first request to last response, with a fake
   slow response) pushes nothing, 30 seconds pushes `Date` minus 30 seconds, across several pages the
   elapsed time covers all of them, and
   a correct clock with a valid `Date` pushes that `Date` value minus the elapsed time.

## Probe results (Nov 25 2026, read only, his saved login in memory, nothing printed but counts)

These settle the C6 gate, items (a) to (f). Builders work from them.

- **(a) Request.** `GET https://api.evallo.org/api/events/summary`. Auth is the saved login's token sent
  as-is in an `authorization` header (no `Bearer` prefix), plus `accept`. Query keys, as the Evallo app
  sends them: `userIds[]=<pinned tutor id>`, `mode=list` or `week`, `currentTZ=America/New_York`,
  `withPag=true`, `startDate`, `endDate` (JavaScript `Date.toString()` style in New York time, for example
  `Wed Nov 18 2026 00:00:00 GMT-0500 (Eastern Standard Time)`, with the end at 23:59:59), `_t`
  (milliseconds), and `extraFields[]` twice (`avatar`, `locations`). The job sends the same keys and
  asks for the window today − 7 days to today + 60 days.
- **(b) Pagination.** The response has no page, limit or total field: top level `success`, `data`,
  `message`; `data` holds only `allEvents`. `withPag` changes nothing (same ids with `false`). A 190 day
  window in one request returned 74 rows, identical to the union of its week sized requests (74). So no
  truncation was seen, but the response cannot prove completeness. **Completeness rule:** the job asks
  for the window once in `list` mode and again as 7 day `week` chunks, and pushes only if the two id
  sets are equal. It also refuses if any one response has 200 rows or more (a cap larger than any seen).
  Every request counts toward the 60 second elapsed limit of C6 (about 11 requests, 0.1 to 0.3 s each).
- **(c) Canceled.** `sessionStatus == "CANCEL"`. Other values seen: `COMPLETED`, `SCH`. A row with
  `sessionStatus` null is not a tutoring session (see the last bullet).
- **(d) Empty result.** HTTP 200, `{"success": ..., "data": {"allEvents": []}, "message": ...}`.
- **(e) Transparent or declined.** None of his 12 real sessions on his Google calendar is marked
  transparent or declined (the other tutors' 24 events: none transparent either). No stop condition.
- **(f) Identity and stamp.** The saved login's `id` claim equals the tutor id he gave; every one of his
  rows has `tutorId` equal to it; `tutorName` on his rows has a single raw form and equals his full name
  after trim and case fold; authenticated responses carry a `Date` header (within a second of the Mac's
  clock). Row fields: `_id`, `tutorId`, `tutorName`, `date`, `eventEndDate`, `sessionStatus`,
  `studentIds`, `studentName`, `time`, `timeZone`, `service`, `location`, `recurring`, `eventTitle`.
- **Not his, but returned for his user id.** The response for his id also held one row from another
  tutor (Nov 24, no `sessionStatus` or `service`) whose `attendees` list names him. By C6 step 2 it is
  not his (its `tutorId` differs) and is never pushed. Whether attendee rows should count as busy is an
  open question for him; the contract is unchanged.

## Build plan (after the gate below clears)

1. **Mac repo: done.** Backup `~/miles-backup-2026-11-25/miles-evallo-mac.tgz` (mode 600), `.gitignore`
   first, baseline commit `48e813c`. Builder 2 works in a git worktree of that repo, using the
   folder's existing `.venv-evallo` to run tests.
2. **Gate before any builder:** Codex reviews only the changes since revision 2. Any High finding stops
   everything. Then the Step 3 probe runs. Any of his real sessions marked transparent or declined
   stops everything.
3. Read only probe (by me, not a builder): settle C6 (a) to (f), record them here and show him. It
   uses his saved login in memory only and prints counts and flags, never tokens or raw responses.
4. Builder 1 (Sonnet, Pi worktree `~/miles-evallo-filter`, branch `feature/evallo-filter` from main):
   migration 045, `src/evallo_filter.py`, the hook in `_fetch_events`, the PUT route, the status
   lines, a read only comparison script (below), and the Pi tests.
5. Builder 2 (Sonnet, Mac worktree, branch `feature/evallo-sessions`): C6, the pin script, the plist
   and the Mac tests. Starts after Builder 1 is merged for review, never in parallel.
6. End of phase: Codex adversarial review on astra-high. Fix what it finds.
7. Deploy plan with rollback for his approval. Order: DB backup; migration check against all open
   branches; merge and restart the Pi (filter **off**); merge the Mac branch into its `main` (launchd
   runs the repo path, no folder swap); **pin the tutor (he supplies his id; the script refuses a
   mismatch)**; load the plist, which refuses to load without the pin; **run the Mac push at once**;
   the first push accepted turns the filter on; run the verification below.

## Verification

- `pytest src/tests` on the Pi worktree, and the Mac suite in its worktree.
- A read only comparison, not the audit file (it has no session ids and the Google event `id` is not
  the Evallo `_id`): a Pi script writes, to a mode 600 file, one line per surviving Evallo event with
  its `evalloSessionId` (or "none"), its Google `updated` time and **the C4 row that kept it**, and a
  Mac script (`scripts/evallo_compare_survivors.py`, built in revision 5) compares the ids with Evallo's list for him. Every one of his is present; every other
  survivor has a row of 1, 3 or 5 as its reason, and an event with no id appears as row 1. A survivor
  with row 4 as its reason is a failure.
- Busy or free for known slots from his real sessions and from another tutor's, allowing events kept
  by C4 row 1, 3 or 5. The nightly review has no stale line once `active`.

## Rollback

- Revert the Pi code (`git revert` the merge) and restart. **Keep the database**: the two new tables
  are unused by the old code, and restoring the pre deploy backup would discard writes made since.
- On the Mac: `launchctl unload` the plist and remove it from `~/Library/LaunchAgents`, then `git
  revert` the Mac merge. The tarball is the last resort.
- Result: every Evallo event shows again, which is safe (busy, never free). A later redeploy starts
  with the filter off (C2) until a new push is accepted.
