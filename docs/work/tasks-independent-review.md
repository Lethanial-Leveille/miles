# Independent Tasks implementation review

Reviewed backend `fb4a786` against `51676b9` and Nova `43dde91` against
`d9e8c9f`. The review used [revision three](tasks-revision-three.md), the
approved [day guard](tasks-day-guard-contract.md) at `9e1fd59`,
[scheduling amendment](tasks-scheduling-amendment.md) at `2203bbc`, and
[voice safety amendment](tasks-voice-safety-amendment.md) at `c29d42a`.
The original finding IDs follow the [continuation checkpoint](tasks-followups.md).
The run header did not expose the model and reasoning profile for verification.

**Decision. Keep the scheduling resolver offline.** The guarded HTTP
actions and the paired Nova row binding pass synthetic verification, as do
migration 046 and structured daily eligibility. The live voice path can
still drop recurrence, speak a success without a write, dispatch an unrelated
change before a refusal, or confirm an unseen proposal. The offline parser
also returns supported plans for some unsupported timing. Whole phase
acceptance remains open.

## High

### 1. Unsupported timing can produce a successful ordinary add (H1, H2)

The live guard accepts “Add water plants Saturdays” when the model echoes the
whole phrase as the title. The synthetic database gained a one time task for
today and the reply reported success. The offline parser separately returns
an untimed supported plan for “Add call the bank at half past five” and
“Add water plants on the first of the month” if the model echoes the full
title. Matching the title independently of the model does not help when
code misses the scheduling clause. The implemented checks do cover the
contract examples for deadlines, qualified first days and date substitution.

Use code to identify unsupported clock and recurrence clauses and classify
unconsumed timing as a clarification before falling back to an untimed
plan. Keep the approved title exception narrow and contextual. Regress these
examples against both the current voice entry point and the proposed
resolver entry point. Assert no task or completion changes.
[Live guard](../../src/voice_action_safety.py#L16) ·
[Offline clause and title checks](../../src/task_scheduling.py#L128)

### 2. A control call or a missed intent can still speak success without a write (H3)

For “Add reading daily,” a fake model sent “Done. I saved the task.” with
only a `dismiss` call. The control was refused and both sentences still
reached the audio consumer. The first round releases arbitrary model draft
for control only blocks before preflight. A tool free model reply also
claimed success for “I need to take my vitamins every day,” an add form
documented by the tool and missed by `expects_action`. Tests for a refused
`task_add` in an ordinary mutation batch pass.

Use one committed intent classification for no tool replies, control
proposals and mutations. While an action is possible, release fixed neutral
speech or the code outcome only. Preserve normal read replies and valid
dismiss or ignore behavior. Assert audio and canonical text in both cases.
[Control speech](../../src/brain.py#L1399) ·
[No tool fallback](../../src/brain.py#L1649) ·
[Intent classification](../../src/voice_action_safety.py#L124)

### 3. Earlier model rounds can escape the write guard and transport replay can write twice (H4)

For “Add reading,” the first round proposed `set_timer` plus a task read.
The synthetic timer changed state. The next round proposed `task_add`,
which was refused because a mutation had already run. `_TASK_REQUEST`
misses the short add before a guarded tool appears. The same round archive
sibling cases pass in both orders. Separately, two deliveries of the same
synthetic `/chat` request and tool use ID each inserted a task. The HTTP
envelope has no committed request identity and the dispatch record lives
inside a fresh `TurnSafety` each time.

Classify the committed request before exposing write authority across rounds.
Reject unrelated writes before the first effect and resolve prerequisites
before releasing a guarded write. Carry a stable committed request identity
through transport retries and atomically reserve it before dispatch. Retain
the known or uncertain disposition for replay. Give a genuinely new request
a new identity. Test the earlier write, same round siblings, and lost
response retry against actual SQLite state.
[Preflight](../../src/voice_action_safety.py#L128) ·
[Turn state](../../src/brain.py#L1199) ·
[Chat request envelope](../../src/server.py#L102)

### 4. A second app conversation can confirm an undelivered proposal

Two synthetic ASGI clients used the same login and text channel. The first
client lost the `/chat` body before delivery. The second sent “Yes” and
executed the first client's staged callback. The scope for both is the
device, channel, login subject and a missing conversation start. The
nonstream route supplies no delivery evidence, and the proposal check
accepts its absence. The stream does record delivery at its server boundary
but uses the same account wide scope.

Require a stable conversation identity and bind the exact proposal to the
authenticated speaker, the conversation and delivery evidence. Cover both
app routes, the next turn rule and one time consumption. Revalidate every
specific action under the existing policy when consuming approval.
Regress same account conversation isolation and disconnect before delivery.
[App route](../../src/server.py#L198) ·
[Stream route](../../src/server.py#L216) ·
[Scope](../../src/brain.py#L1199) ·
[Approval predicate](../../src/pending_action.py#L118)

## Medium

### 5. Ordinary task wording and title words now fail recognition (M1)

The live guard refuses two ordinary forms advertised by the existing tool,
“Add call the bank to my tasks” and “I need to take my vitamins every day.”
It also refuses “Add watch Friday Night Lights.” The offline resolver
accepts that media title capitalized, but asks for clarification when the
verified transcript is lowercase. Explicit “starting today” does store a
null start in the offline plan as intended.

Recognize existing request forms and identify the title span before testing
its contents for timing. Handle transcript casing without relaxing the rule
against dropping unsupported timing. Test existing add and completion
phrasing, title casing, and bare follow ups alongside the timing refusals.
[Advertised task forms](../../src/lists_tasks.py#L208) ·
[Live grammar](../../src/voice_action_safety.py#L40) ·
[Offline media rule](../../src/task_scheduling.py#L42)

### 6. Spoken outcomes lose specific refusals and can report unusable confirmation results (M3)

Storage refused a prestart completion and the tool returned a specific
notice with the first day. The shared brain replaced it with “Nothing was
changed by that action.” A confirmation batch whose callbacks returned
`None` and `False` spoke “None False” with no action labels. The outer
dispatch rejects absent results but the confirmation resolver already
converted these values into text. The revised schema and bounded
`not_saved` handling remain absent from the covered tools.

Return typed, code written results carrying a user sentence, a sanitized
machine disposition and the action identity. Keep the first day and other
safe refusal details, separate model instructions, distinguish failed and
uncertain actions individually, and add the applicable dropped detail
handling to all five guarded tools. Preserve unaffected `ToolNotice`
behavior. Check exact audio, screen and model results against affected rows.
[Prestart notice](../../src/lists_tasks.py#L256) ·
[Notice handling](../../src/brain.py#L1058) ·
[Confirmation results](../../src/pending_action.py#L267)

### 7. Active title uniqueness is unenforced (M4)

Two concurrent adds of the same active daily title both committed and both
claimed success. General structured creation remains intentionally
nonidempotent. The voice slice needs a write boundary for the original
voice duplicate rule, independently of same request replay. Recognized
standalone edit commands are refused by the offline resolver.

Normalize the proposed title against the approved active task rule and
check and insert together under a writer transaction in the voice add
path. Preserve structured create behavior. Regress new adds against existing
daily and one time rows, archived titles, edits and concurrent voice
requests.
[Voice add](../../src/lists_tasks.py#L231) ·
[Insert](../../src/database.py#L2589) ·
[Offline edit check](../../src/task_scheduling.py#L109)

### 8. Nightly review can claim completion after a concurrent archive

The nightly review reads an active task and then ignores the Boolean from
`complete_task`. A synthetic archive between those operations prevented the
completion and the review still returned “Marked Reading done for today.”
The same unchecked result is present on the one time path. The guarded
HTTP action has a writer transaction covering active status, and the
ordinary voice completion checks the result.

Check the storage result in both review branches and return a safe refusal
when the task stopped being active. Regress the archive interleaving and
verify the review text and completion rows together.
[Review action](../../src/nightly_review.py#L177) ·
[Storage result](../../src/database.py#L2633)

### 9. The task day relies on process local time and voice add lacks one captured clock (M5)

The HTTP routes use one captured instant and were correct across 04:00 in
the tested local clock. The helper subtracts four hours without normalizing
to New York as specified in revision three sections 3.7 and 3.8. The same
instant, 08:59 UTC and 03:59 in New York on November 26, yielded different
task days when passed in the two time zones. A process using UTC therefore
shifts the cutoff. The current voice add calculates the scheduled day and
then lets storage read the clock independently. The offline resolver
accepts an injected day, but there is no capture shared with a live write.

Use the existing app clock policy explicitly for a New York task day across
Tasks and review entry points, or codify and enforce the process time zone
dependency. Preserve the existing stored timestamp interpretation. Carry one
captured instant through voice resolution, validation, storage and result
text. Test review for a requested prior or future day and cross 04:00 with a
single captured voice write.
[Clock helper](../../src/database.py#L2578) ·
[HTTP capture](../../src/tasks_view.py#L110) ·
[Voice add](../../src/lists_tasks.py#L231) ·
[Review read](../../src/nightly_review.py#L147)

## Low and rollout

- **Nova daily UI remains outstanding.** The approved checkpoints defer
  the daily UI to separate authorization. At `43dde91` the client
  safely ignores additive fields and the current guard still refreshes on
  action errors. It cannot display `upcoming_daily` or create a daily
  task through the sheet. Complete revision three section 3.9 in the
  authorized slice and sync the paired contract first.
  Nova `Nova/Models/NovaTask.swift:5`,
  `Nova/Views/TasksTabView.swift:64` and `:263`.
- **M7, M8 and L2 remain rollout gates.** Verify the exact release registers
  045 before 046 and confirm 045 is applied before any authorized restart.
  An older backend ignores the rendered day and a pre 046 backend also
  ignores stored future starts. Before an authorized rollback, pause
  affected writes or coordinate the paired client and backend, reconcile
  future rows from a protected backup, and verify both versions and the
  remaining rows. Record the deployed commit, exact rollback target,
  migration compatibility and the mainline parent if reverting a merge.
  [Migration order](../../src/database.py#L1245) ·
  [Pairing checkpoint](tasks-followups.md#paired-nova-checkpoint)

## All fifteen original findings

| ID | Disposition at the reviewed revisions |
| --- | --- |
| H1 | Outstanding. Common examples refuse, but unsupported timing and recurrence can produce default writes. See High 1. |
| H2 | Partial. Explicit role, qualifier and corroboration cases pass. Missed timing and recurrence clauses still get absorbed into the title. See High 1. |
| H3 | Partial. Ordinary guarded refusal holds draft speech. Control and missed intent paths still speak unsupported success. See High 2. |
| H4 | Partial. Same round sibling refusal and within turn duplicate checks pass. Earlier unguarded rounds and transport replay remain open. See High 3. |
| H5 | Resolved for the paired day guard. Both actions carry the rendered overview day from Today, later sections and Backlog. Refresh preserves the notice and does not replay the action. Nova `Nova/Models/NovaTask.swift:38`, `Nova/Services/APIService.swift:210`, `Nova/Views/TasksTabState.swift:61`. |
| M1 | Partial. Explicit today normalizes to null offline. Established wording and clear title cases remain open. See Medium 5. |
| M2 | Resolved in the offline slice under the amended full restatement rule. A bare “Tuesday” or “yes” returns no plan. Recheck original committed utterance binding and no write on follow up during live integration. |
| M3 | Partial. Ordinary guarded write outcomes replace model draft. Specific refusals, per action confirmation outcomes and dropped detail handling remain open. See Medium 6. |
| M4 | Outstanding. The offline edit guard covers recognized edits, but there is no atomic voice duplicate check. See Medium 7. |
| M5 | Partial. HTTP actions capture one instant and recurring reads use requested day eligibility. Explicit New York handling and a single voice write clock remain open. See Medium 9. |
| M6 | Resolved at the HTTP boundary. Body validation precedes lookup, archived and missing IDs precede the day comparison, and the writer lock prevents a concurrent archive from slipping between check and write. The separate nightly review result issue is Medium 8. |
| M7 | Implementation resolved synthetically. Release revision and applied 045 prerequisite remain rollout checks. |
| M8 | Outstanding rollout coordination. An older backend silently removes the guard and the eligibility rule. |
| L1 | Resolved. Impossible due dates and explicit action JSON null return 422 without mutation. Null optional create dates and bodyless legacy actions retain their documented behavior. |
| L2 | Outstanding rollout documentation. Record exact rollback mechanics after the release history is known. |

## Synthetic verification

- Backend suite on Python 3.13 with a disposable home and SQLite fixtures
  passed **639 tests**. Coverage included Tasks and Lists APIs and storage,
  historical and upcoming daily eligibility, migration 046, the offline
  resolver, nightly review and Sessions, Evallo regression, brain, voice
  preflight, pending approval, transport, speculative requests, permissions
  and TTS. Migration tests cover 044 and 045 upgrades, Evallo and task
  history preservation, compatible and incompatible preexisting columns,
  atomic rollback and retry, and concurrent starters applying 046 once.
- Fifteen additional synthetic probes reproduced the reported defects and
  confirmed the associated outputs and database or callback state. They
  used fake model rounds, audio consumers, in process HTTP clients and
  calendar callbacks. The probes assert the observed behavior on the pinned
  revision and are evidence of defects.
- Nova standalone Swift checks passed with the actual models, API service
  and view state and a synthetic transport. Both actions, authenticated
  request shape, stale snapshots, overlap, repeated taps and refresh after
  conflict were exercised. This run did not build the full iOS app or test
  a physical device. The earlier unsigned build and backend pairing remain
  recorded in the Nova handoff.
- Both pinned diffs passed Git whitespace checks. No production services,
  real data, external providers, microphones or devices participated.

## Nightly backup checkpoint

Preserve backend `feature/nightly-backup` at `28a9a2a`,
`docs/nightly-backup-independent-verification` at `c47522f`, and the
paired Mac `feature/backup-pull` at `bfe4384`. The existing verification
and subsequent focused corrections are recorded in
`28a9a2a:docs/work/nightly-backup.md`. The prior synthetic counts were
77 backend and 15 Mac tests. Native Linux boot clock, effective installed
watchdog configuration and real rollout checks remain open. Those include
timer and launchd execution, wake catch up, Tailscale transport and
timeouts, reporting, Pi verification cost and durability across power loss
or reboot. FileVault, the local destination outside cloud sync, the
privileged installer decision and owner approval remain rollout
prerequisites. Backup acceptance still precedes Tasks rollout.
