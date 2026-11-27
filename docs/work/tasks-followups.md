# Tasks follow ups continuation

Backend branch `feature/tasks-followups` starts at canonical `main` at
`51676b9`. This is a documentation checkpoint. The [revision three contract](tasks-revision-three.md)
is the historical proposal reviewed on November 26, 2026. The current
[shared API contract](../TASKS_CONTRACT.md) remains in force. The prior owner
decisions include retaining the optional legacy request, a 365 day horizon,
no start date edits, and resolving spoken dates in code. No later approved
revision was established in this preparation. Local source was checked only
where needed to map the existing review to the current base.

## Backup checkpoint for Claude

The backend nightly backup branch is at `28a9a2a` and the Mac backup pull
branch is at `bfe4384`. The verification commit `c47522f) and the subsequent
fixes are preserved. Read the final section of the backup handoff using
`git show 28a9a2a:docs/work/nightly-backup.md`. Claude's focused verification
covers backend `c47522f..28a9a2a` and Mac `3109e0d..bfe4384`. The prior run
passed 77 backend and 15 Mac synthetic tests. The native Linux boot clock,
effective installed watchdog setting and real rollout checks remain open.
Rollout still needs owner approval, FileVault confirmation and a privileged
installer decision. The roadmap places backup acceptance before Tasks. The
recorded November 26 Evallo deployment at `51676b9` applied migration 045 and
accepted the Mac session push. Reconfirm prerequisites separately before a
later deployment using authorized evidence.

## Prior review and applicability

The independent revision three report found five High, eight Medium and two
Low findings. The table preserves its substance and distinguishes a
contract gap from an already implemented rule. This is a targeted source
check against canonical main, with the recorded Nova Tasks checkout at
`d9e8c9f` checked for the paired row and request signatures.

| Prior finding | Status against the base |
|---|---|
| H1 Unsupported scheduling or recurrence falls through to a write | Open contract issue. `src/lists_tasks.py` currently accepts task and recurrence only. Require unsupported scheduling clauses to block defaults before introducing the resolver. |
| H2 A verbatim date span loses its role or qualifiers | Open contract issue. Distinguish start, due and relative qualifiers before allowing any voice write. A Friday due date is not a Friday start. |
| H3 Audio can confirm a write before refusal | Open integration issue. `src/brain.py` plays model text alongside tool execution. Verify with a model response containing a false confirmation and a refused call. |
| H4 A sibling or earlier write can escape the refusal guard | Open dispatch issue. `src/brain.py` iterates calls, so order matters even with sequential dispatch. Define the guarded batch boundary before the first mutation. Test affected rows, including archive state. |
| H5 Nova row captures the wrong day | Open. `src/server.py` actions have no day guard. Nova's existing actions take only the ID. Fix the conflicting revision three build instruction so both actions use the day of the rendered row even after refresh. |
| M1 Title words and explicit today can be refused | Open before voice implementation. Distinguish title spans and explicit today, which stores a null start date. |
| M2 Clarification can lose original timing intent | Open. Establish the source utterance and what a clarification can replace. A bare confirmation needs the original pending intent. |
| M3 Spoken versus ToolNotice and leaking model instructions into speech | Open. The brain handles ToolNotice as an error result. Align the contract and tests with the intended user audio and model result for refusal, partial save and success. |
| M4 Duplicate or edit refusal depends on argument order and concurrent lookup | Open contract issue. The current voice add calls `add_task` directly. Define edit refusal and use an atomic check and insert if active title uniqueness is approved as a guarantee. |
| M5 Review and voice effective day | Open. The daily task path currently has no start date. Specify the write day for a future review and pass the same captured day through resolution and storage across 04:00. |
| M6 Archived lookup order | Existing complete and reopen already refuse inactive tasks in `src/tasks_view.py`. Preserve that behavior in the new guarded error ordering and test archived IDs with both matching and stale days. |
| M7 045 prerequisite recheck | Open rollout gate. `src/database.py` registers 045 on main and no 046. Verify the eventual merge's migration order and the applied prerequisite before restart. |
| M8 Rollback mismatch and future rows | Open rollout gate. The current server actions have no guard, so a new client rolled back against this server loses stale tap protection. Pause or coordinate writes, recheck future rows and verify both versions. |
| L1 Impossible due date and JSON null coverage | Current create route already rejects impossible dates with 422. Keep a regression for it. Guarded request parsing is unwritten and explicit top level null must not become the legacy missing body. |
| L2 Rollback commit mechanics | Open rollout detail. Record the deployed history and exact revert target and mainline parent when applicable. |

## Small units and dependencies

| Unit | Dependency and minimum synthetic verification |
|---|---|
| Contract correction | First. Reconcile the guarded action check order, row day binding and the reviewed voice cases. Record owner approval before implementation. Preserve paired backend and Nova contract copies when accepted. |
| Guarded complete and reopen | After approval and roadmap gate. Backend slice can precede migration 046. Use one captured day and verify both routes around 04:00, inactive IDs, malformed or null bodies, legacy omission, no mutation on 409 and successful same day retries. Start at `src/tasks_view.py`, `src/server.py` and `src/tests/test_tasks_api.py`. |
| Nova row day | After the guarded API shape is agreed. Build separately from the selected Nova main revision. Test a tap on an old row after refresh for both actions, then verify 409 reload through the paired backend. |
| Daily storage and API | After the migration gate and accepted start date semantics. First verify additive 046 after 045, legacy rows, daily filtering and untouched prestart history in SQLite. Then verify serialization, create validation, overview and not started responses. Existing fixtures are in `src/tests/test_lists_tasks.py` and `src/tests/test_tasks_api.py`. The Nova add sheet and Starts later section depend on this response shape. |
| Voice intent and write safety | After contract closure. Test the phrase resolver and provenance using pure synthetic utterances, then batch preflight and provisional speech using fake tool calls and synthesis. Only after those pass, wire task and list write results and nightly review day handling to the storage and guarded rules. Existing suites include `test_lists_tasks.py`, `test_brain_loop.py` and `test_nightly_review.py`. Keep the wider write tool audit report only. |

**Next assignment** Ask the owner to approve a focused correction to the
revision three contract for the backend day guard and Nova's rendered row
binding. Leave the voice and migration slices gated until their reviewed
requirements are reconciled. Once the phase gate is cleared, implement
only the guarded backend complete and reopen slice first. No application
tests were run for this documentation checkpoint. Production, pushes,
deployment and new review rounds remain separately authorized.

## Focused day guard contract checkpoint

The owner authorized isolated local Tasks development while backup acceptance
remains pending. The exception leaves production rollout order and backup
verification intact. The owner approved the [focused contract](tasks-day-guard-contract.md)
at `9e1fd59`. Its backend rules are incorporated in the
[shared contract](../TASKS_CONTRACT.md).

The backend captures the task day once and derives completion and response
from that capture. A single SQLite writer transaction checks active status,
applies the guard, mutates the eligible occurrence and reads the resulting
Task. Rejected actions leave task and completion rows unchanged. Legacy
requests are accepted and counted by an info log. A client that omits the
rendered date can still act on the new task day after 04:00.

Backend verification used Python 3.13, a disposable home and synthetic
fixtures. The following command passed all 299 tests, including concurrent
archive and duplicate action races, authorization and rejected writes.

```sh
python -m pytest -q src/tests/test_tasks_api.py src/tests/test_lists_tasks.py \
  src/tests/test_lists_tasks_api.py src/tests/test_sessions_api.py \
  src/tests/test_memory_api.py src/tests/test_nightly_review.py src/tests/test_review_v2.py
```

Stop at the backend checkpoint. Nova was left untouched. When Nova
integration is authorized, sync the shared contract to the paired checkout
first, then verify complete and reopen on a rendered row held across a
refresh. Migration 046 and voice corrections remain separate. No rollout
or new independent review was initiated.

## Paired Nova checkpoint

The backend implementation is `a9d2fda` on `feature/tasks-followups`.
Nova `feature/tasks-day-guard` is at `43dde91`, forked from canonical Nova
`d9e8c9f`. The Nova Tasks contract copy matches the backend at `a9d2fda`.
In the Nova repository, see `docs/tasks-day-guard.md` for reproducible checks
and the compatibility matrix. The original Nova project file edit and the
early speech branch at `cf6a257` were preserved.

Complete and reopen send the rendered overview day through refresh and
retry. An outdated day displays the 04:00 notice and reloads without
replaying the mutation. The unsigned generic iOS build and standalone Swift
regressions passed with Xcode 26.4.1. The Swift checks use fake login and
transport and exercise stale rows, overlapping refreshes, repeated taps and
notice persistence. Four synthetic FastAPI pairings were checked against
each actual backend revision, `51676b9` and `a9d2fda`, using separate scratch
databases. Legacy apps keep working on the guarded backend without stale day
protection. The legacy backend silently ignores the day sent by updated
Nova, so release and verify the backend before updating the app. Coordinate
writes and the app version if rolling the backend back.

No production or physical device check was run. Preserve the existing rollout
gates and stop before migration 046 and voice work.

## Focused scheduling amendment checkpoint

The [proposed scheduling amendment](tasks-scheduling-amendment.md) reconciles
revision three H1 and H2 and the related title and clarification cases. It
records the migration 046 boundary and synthetic cases for ambiguous timing,
unsupported recurrence and mixed start and deadline requests. The owner
approved the amendment at `2203bbc`. The day guard remains paired with Nova
`43dde91` and backend `a9d2fda`. Sync the evolved shared contract when the daily
Tasks UI is separately authorized.

Migration 046 adds a nullable start date after Evallo 045 with no backfill.
Existing recurrence, due dates, archived state, Evallo data and completion
history are preserved. The structured API distinguishes one time due dates
from daily starts. Reads use the requested task day and writes enforce the
captured task day, including across 04:00. The existing completion tools
now surface an eligibility refusal for prestart tasks.

The scheduling resolver is offline and outside the tool registry. Voice
deadlines, including a deadline paired with a valid start, are currently
unsupported. Clarifications currently require full restatement. These
limitations may be revisited. Existing voice creation still uses the
legacy path. Wire the resolver into voice only after provisional speech
and concurrent tool mutation safeguards are approved and verified.

Local synthetic verification used Python 3.13 and a disposable home. All
495 tests passed. The migration, scheduling, Tasks, Lists, Sessions API,
Memory API, nightly review, Review v2 and Evallo filter suites covered 044 and 045 upgrades,
idempotence, concurrent starters, integrity, history, supported due dates,
and no-write conflicts. Keep the rollout gates, including checking that
045 is applied before any authorized migration in production.

## Voice safety contract checkpoint

The [proposed voice safety amendment](tasks-voice-safety-amendment.md)
reconciles H3, H4 and the outcome text issue in M3 from the recovered
revision three review. It locates the current speech, first and follow up
dispatch, speculative hosted tool and pending confirmation boundaries.
It is awaiting owner approval. The offline resolver and migration remain
at `41d1fb0`. The Nova early speech experiment remains separate at
`cf6a257`. No runtime changes or comprehensive review were started.
