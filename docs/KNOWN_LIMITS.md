# Known limits

Open findings that are not Blockers. Each was reviewed and deliberately left. Remove an entry when it is fixed.

- **Should fix** `src/task_intent.py:240` Permanent-stop keywords take priority even when explicitly negated. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:266` The voice path refuses supported daily first days and the changed tests encode that narrower behavior. (found by the review loop at 13424b1)

- **Should fix** `src/brain.py:1321` A skip-or-stop choice becomes active before the question has been delivered. (found by the review loop at 13424b1)

- **Should fix** `src/tests/test_task_intent.py:383` The supplied changes do not verify Nova's required behavior when a stream ends after a delta and before `done`. (found by the review loop at 13424b1)

- **Should fix** `src/database.py:1245` Backdating every legacy receipt by two days can release a recently used request ID two days early. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:392` A relative reminder with an offset-bearing ISO timestamp raises `TypeError` during validation. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:394` The clock parser rejects compact times such as `5pm`. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:483` The affirmative check rejects legitimate completions when the task title contains a hypothetical keyword. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:265` The new negation rule can interpret a request to stop permanently as a one-day skip. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py:544` The timer validator rejects compound durations that the timer tool can represent exactly. (found by the review loop at 13424b1)

- **Should fix** `src/tests/test_task_intent_followups.py:163` The rollover regression test depends on the machine’s actual task day. (found by the review loop at 13424b1)

- **Known limit** `src/task_intent.py` The intent step has no "other change" action. A calendar write, `remember` and every other mutating tool outside task, list, timer and reminder keep their own confirm first flow and are offered only on a turn the wide check flagged. A statement the wide check does not flag (for example "my cat is named Sam" with no action word) no longer reaches `remember`, because an unflagged turn offers no mutating tool. The nightly memory pass is unaffected. Accepted by the owner for this pass.

- **Known limit** `src/task_intent.py` A question turn ("what do you remember about my commute?") no longer reaches `remember`, and neither does an unprompted statement the wide check does not flag. `remember` stays on a flagged turn that is not a question, so "remember that my cat is Sam" works. The nightly memory pass is unaffected. Accepted by the owner for the final round.
