# Proposed Tasks day guard and Nova row binding

**Status** Approved at `9e1fd59`. This file preserves the proposal as approved. This reconciles sections 3.4 through 3.6
and the rendered row rule in section 3.9 of the
[recovered revision three proposal](tasks-revision-three.md). The
[current shared contract](../TASKS_CONTRACT.md) incorporates the approved behavior. The owner authorized isolated local Tasks development while backup
acceptance remains pending. That exception leaves production rollout order and
all backup verification requirements in force.

## Complete and reopen requests

Both existing POST routes continue to require the normal login JWT and its
existing authorization checks. A guarded request body is exactly
`{"day":"2026-11-25"}`, where `day` is a real Gregorian date in the exact
`YYYY-MM-DD` form. The request date guards the action on the server's task
day. It never selects a historical or future completion.

For authorized requests, apply these checks in order.

1. Return 422 for an invalid body before task lookup. Accept a missing or
   zero byte body and an empty JSON object as legacy forms. They target the
   current server task day. Log the route and task ID at info level for
   legacy usage counting, without logging request bodies or credentials.
   Reject a nonempty malformed body, top level JSON null, arrays, scalar
   values, extra or duplicate keys, explicit null or nonstring `day`, and
   invalid calendar dates such as `2026-02-30`. A malformed guarded request
   never falls back to legacy handling.
2. Return 404 for a missing or archived task, regardless of whether the supplied
   day matches. Preserve the current active task rule. Check active status at
   the write boundary as well so a concurrent archive cannot authorize a write.
3. Return 409 for a valid supplied day earlier or later than the captured
   server task day. Return this body and change no task or completion state.

   ```json
   {
     "detail": "The day changed at 4 a.m. Nothing was changed.",
     "code": "day_changed",
     "today": "2026-11-26"
   }
   ```

4. Apply the requested action and any action specific conflict against the
   captured day.

Capture the backend clock once for the action after request validation and
before evaluating task state. Use the same instant and task day for checks,
writes, action response fields, and serialization of the returned Task.
Keep the existing backend clock policy and 04:00 local boundary. For example,
00:00 through 03:59 on November 26 belong to task day November 25. At 04:00
the task day becomes November 26. A request captured before 04:00 still acts
on the earlier day if its write or response finishes afterward. A request
captured at or after 04:00 with the earlier day returns 409.

## Action outcomes and repeats

| Action | One time task | Daily task |
|---|---|---|
| Complete | Preserve the existing done status update and completion recording for the captured day. | Record just the captured day's occurrence. Other occurrences are unchanged. |
| Reopen | If done, return the task to pending and remove all its completion rows, including rows from other days. | Remove only the occurrence for the captured day. Other occurrences remain, including yesterday's. |
| Reopen when already open | Return 409 with `code: "not_done"` and detail `That task is not done`. | Return 409 with `code: "not_done"` and detail `That task is not done today`. A completion on another day does not make this occurrence done. |

Complete returns 200 with the existing `{"completed":true,"day":"<captured day>","task":Task}`
shape. Reopen returns 200 with the existing `{"task":Task}` shape. The second
completion on the same day returns 200 and leaves at most one completion row
for that task and day. The second reopen on the same day returns 409
`not_done` and changes nothing. An original guarded request retried after
04:00 keeps its original `day` and receives 409 `day_changed`, even if the
original request succeeded. A legacy retry continues to use the server task
day at the time of that retry.

## Nova row binding

Bind `overview.today` to each rendered row and capture it with the row's ID
and action. Today, a later day section and Backlog all use the same day from
the overview snapshot that rendered them. The section date and the phone
clock do not supply the guard. If a refresh produces a new overview, an
existing row's queued tap or in flight request still sends its original day.
This applies equally to complete and reopen.

For 409 `day_changed`, show “The day changed at 4 a.m. Nothing was changed.”
and refresh the overview. Preserve the notice through that refresh. Do not
automatically replay the action with the new day or substitute an unguarded
request. A new deliberate tap on the refreshed row may act on the new day.
Keep the existing UI behavior for other errors and successful actions. The
paired Nova change follows backend verification. A legacy server can ignore
the body, so the paired client must not be released against a backend that
lacks the guard without a separately approved rollout decision.

## Scope and acceptance

Task creation stays on the current contract. An impossible `due_date`,
including `2026-02-30`, remains a 422 with no new task. Existing valid dates
and omitted or null dates remain supported. Backlog and calendar operations
keep their existing permission boundaries. This unit changes neither task
schema nor voice behavior. Migration 046, start dates and `not_started`,
create validation changes, and voice and review corrections remain separate.

- Backend synthetic checks cover both routes at midnight and 04:00, a
  request spanning 04:00, earlier and later supplied days, archived and
  missing tasks with matching and stale days, invalid body forms and the
  legacy forms, authorized and unauthorized callers, the impossible
  `due_date`, and both repeated action outcomes. Verify exact affected
  completion rows and the returned Task.
- Nova verification holds an old row across a refresh, triggers both
  actions and inspects the outgoing day. Verify the 409 message and refresh
  without replay. Include a later section or Backlog row to confirm the
  guard belongs to the overview snapshot.

No separate product decision is needed for this slice beyond approval of the
contract. Legacy omission stays compatible as previously agreed. Changing
that policy or enabling release against a legacy backend remains a separate
decision. After approval, promote the accepted contract text to the shared
backend contract and sync the paired Nova copy before implementation.
