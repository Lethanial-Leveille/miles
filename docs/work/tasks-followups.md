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
verification intact. [The focused contract](tasks-day-guard-contract.md)
reconciles the backend action boundary and Nova row snapshot rule. It awaits
owner approval. The next implementation unit is the guarded backend complete
and reopen pair, independent of migration 046 and voice changes. No code or
canonical shared contract was changed at this checkpoint.
