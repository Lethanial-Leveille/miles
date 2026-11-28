# Known limits

Open findings that are not Blockers. Each was reviewed and deliberately left. Remove an entry when it is fixed.

Labels compare this branch with production (`main` at 51676b9), measured by running the same scripted model through both trees. **REGRESSION** means the branch behaves worse than production for that scenario. **PRE EXISTING** means production does the same or worse.

- **Should fix** `src/task_intent.py` PRE EXISTING. Permanent-stop keywords take priority even when explicitly negated ("don't stop reading for good, just skip it today" is refused as unclear). Production archives the task for good on that sentence, so the branch is safer. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py` PRE EXISTING. A first day on a daily task is accepted by voice only when the intent copies the "starting ..." words and the call carries the matching `start_date`. Production's task tool has no first day at all and silently starts the task today. (found by the review loop at 13424b1)

- **Should fix** `src/brain.py` PRE EXISTING. A skip-or-stop choice becomes active before the question has been delivered. Production archives on the first "cancel" with no question, so the branch is safer. (found by the review loop at 13424b1)

- **Should fix** `src/tests/test_task_intent.py` PRE EXISTING. The supplied changes do not verify Nova's required behavior when a stream ends after a delta and before `done`. Production streams the same way and has no test either. (found by the review loop at 13424b1)

- **Should fix** `src/database.py` PRE EXISTING. Backdating every legacy receipt by two days can release a recently used request ID two days early. Production has no receipt table, so nothing existed to compare. (found by the review loop at 13424b1)

- **Known limit** `src/task_intent.py` REGRESSION, accepted by the owner. The intent step has no "other change" action. A calendar write, `remember` and every other mutating tool outside task, list, timer and reminder keep their own confirm first flow and are offered only on a turn the wide check flagged. A statement the wide check does not flag (for example "my cat is named Sam" with no action word) no longer reaches `remember`, because an unflagged turn offers no mutating tool; production offered it. The nightly memory pass is unaffected.

- **Known limit** `src/task_intent.py` PRE EXISTING (exposure), fixed for questions. A question turn ("what do you remember about my commute?") no longer reaches `remember`; production offered it on every turn. `remember` stays on a flagged turn that is not a question, so "remember that my cat is Sam" works. Found as F1.

- **Known limit** `src/brain.py` N1 and V1: PRE EXISTING for a reply that claims a change when the model called no tool (production speaks model text always); REGRESSION for the case where the classifier wrongly says no_change on a real change (production would have run the call, the branch hides the covered tools). Code now replaces a first person past tense claim ("I added", "Done, I archived it") with "Nothing was changed." on a flagged no_change or question turn that ran no change. The check is a wide net: a claim worded another way, and a claim on an unflagged chat turn, still gets through, the same as production. A classifier miss on a real change still changes nothing and asks him to say it again.

- **Known limit** `src/voice_action_safety.py` PRE EXISTING. F4: a successful `cancel_reminder` is still spoken from model text, which can name a different reminder than the one cancelled. It is outside the covered set by contract. Production did the same.

- **Known limit** `src/task_intent.py` PRE EXISTING. F7: the task day is read after the intent call returns, so a request spoken at 03:59:59 whose intent call returns after 04:00 is recorded on the new day. Production reads the clock at write time, after an even longer model call, so it did the same or worse.

- **Known limit** `src/task_intent.py` PRE EXISTING. V4: a recognized question opens the no_change tool set (calendar and the rest, but not `remember`) even when the intent call failed. Production offered every tool on every turn. Closing it would mean answering a read question with "I could not tell what you wanted", which is worse than today.

- **Known limit** `src/task_intent.py` PRE EXISTING. F6: only a task whose first word is a verb it knows (take, call, pay, get, buy, send, submit) is held to that verb. "I bought my vitamins" no longer completes "Take vitamins", but "Pick up vitamins" still accepts any report verb. Production completed on any call the model made.
