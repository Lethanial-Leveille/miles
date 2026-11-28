# Tasks request intent contract

**Tier** Large. **Status** Approved by the owner with the changes below. **Branch** `feature/tasks-followups`, rebased onto the new `main`.

## Why

The two open blockers and the dropped stream bug share one cause. Today the model acts first and code checks afterward, using word lists to guess what was asked. A request the word lists do not recognize (“in four days”, “cancel reading”) gets through, and the spoken reply can come from model text instead of from what happened.

## The rule

For a turn that could change something, the model fills in one structured intent first, then code checks it, and only matching tool calls may run.

1. **Which turns.** A cheap, wide code check decides if a turn could change something (action words, task, list, timer, reminder and calendar words). A turn that fails the check never sees any mutating tool, so a miss changes nothing, never the wrong thing. Ordinary conversation skips the step and stays as fast as now. The added time on task turns is measured and reported.
2. **Intent call.** For a flagged turn, one small model call with a forced schema and no mutating tools returns: action (add, complete, archive, skip today, list add, list check, timer, reminder, or one named other change, or none), target and timing. Nothing runs from this call.
3. **Code check.** Code validates the intent: the target exists and is one task, the timing is supported (below), and there is only one change. Failing any check means no write and a plain question or refusal. Timing must also be corroborated by the committed words, as the scheduling amendment requires.
4. **Gate.** Only tool calls matching the validated intent may run, in every round. Reads stay allowed. Anything else, including a timer or reminder riding along, is refused before any of the batch runs.
5. **Timing.** The scheduling amendment applies unchanged. An untimed one time add is a task for today's task day with no due date. Supported timing is saved as that amendment says. Anything else (“in four days”, “at dawn”) writes nothing and asks a question. It never defaults to today without his answer.
6. **Cancel.** “Cancel X today” or “tonight” skips today's occurrence only. A bare “cancel X” on a daily task asks “Skip today or stop it for good?” and writes nothing until he answers. Archive only on a clear stop request (“stop”, “archive”, “delete”, “for good”). A bare cancel on a one time task archives it. Skip today needs a small new table, `task_skips` (migration 048), because daily tasks have no way to record a skipped day. Overview and completion treat a skipped day as not shown and not due.
7. **Words.** The confirming sentence is built from the action results. A task request that ran no action says “Nothing was changed” plus the reason or question. Model text is used only for ordinary conversation and reads.
8. **Streaming and Nova.** For a flagged turn the backend sends no model text until the actions finish. Nova discards an unfinished reply when the stream drops and shows a plain error (`feature/tasks-day-guard`, same pass).
9. **Ledger 047 expiry.** Reply text is erased after 24 hours and the row is deleted after 7 days. An unfinished row is marked failed after 24 hours. A retry after 7 days is a new request. Cleanup runs once a day in the existing tick.

## Tests

Each phrase from the reviews becomes a test, with a table of near misses and the unaffected cases (“set a timer for 10 minutes”, “add eggs to my grocery list” and “remind me at 5” must still work through the intent step; “cancel reading” on a daily task asks and writes nothing): “add water plants in four days”, “add call the bank at dawn”, “cancel reading” with a timer and reminder in the same batch, “cancel reading” with no tool call, a stream that drops after a delta. Also checked: supported phrases still work, and ordinary chat and reads are untouched. All tests are offline and use fake names.

## Not changing

Evallo, calendar writes, texting, existing tables, and production. Migration 048 only adds `task_skips`. Nothing is pushed or deployed.

## What you will notice

- Timing MILES cannot handle gets a question, not a task for today.
- “Cancel reading” on a daily task asks whether to skip today or stop it. Nothing changes until you answer.
- A timer or reminder never sneaks along with a task change.
- Task turns take one extra small model call. Ordinary chat is unchanged.
- Confirmations sound more standard, because code writes them.
- No “Done” flashes on screen and then vanishes when the connection drops.
- Old chat replies expire from the live database. Copies already in backups stay until those backups age out.

## Conflicts with the approved documents

1. **Scheduling amendment, “code classifies before model arguments”.** Intent now comes from the model, with code validating and corroborating timing. This amends that sentence.
2. **Voice safety amendment, “adds no Nova protocol or code”.** The Nova stream fix amends it for this pass.
3. **Untimed one time add.** You asked for undated, not due today. The amendment and the code both save it scheduled for today's task day with no due date, because an undated task lands in the backlog, which is not on today's list. I follow the amendment. Say if you want the backlog instead.
4. **Skip today** has no storage in `TASKS_CONTRACT.md` (a daily task is only done or not). Migration 048 adds `task_skips`, which amends that contract's overview and complete rules.
5. **One change per request** stays as the voice safety amendment set it. “Skip today” counts as that one change.
6. **Resolver offline.** The scheduling resolver is wired in only as the timing validator, not as a writer.

## Amendments after the review loop (owner approved)

1. **Exact binding.** The validated intent carries the exact task id, task day and list item row, and the gate requires an exact match on the list, item, timer length and reminder content and date. The writer acts on the validated id and day and never reads the clock or resolves the words again. Completion, skip and stop verbs must be the intent's action said about the task, not a word found anywhere in the sentence.
2. **No "other change" action.** Calendar writes, remember and every other mutating tool outside task, list, timer and reminder keep their confirm first flow and are offered on a flagged turn the model calls no_change. They are not an action the intent step can authorize. An unflagged turn offers none of them (see KNOWN_LIMITS for remember).
3. **no_change.** The model picks no_change for questions, reads and chat; the ordinary reply is used. A declared change that ran nothing says "Nothing was changed." with the reason. A read question never gets that line.
4. **Uncaptured timing.** Timing words the intent did not capture (in four days, in half a dozen days, at dawn) write nothing and ask what timing he wants.

