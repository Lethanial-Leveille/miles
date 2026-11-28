# Known limits

Open findings that are not Blockers. Each was reviewed and deliberately left. Remove an entry when it is fixed.

Labels compare this branch with production (`main` at 51676b9), measured by running the same scripted model through both trees. **REGRESSION** means the branch behaves worse than production for that scenario. **PRE EXISTING** means production does the same or worse.

## First fix after this deploy

- **W1, a false "Done" on a misclassified no_change turn.** REGRESSION, the original false success bug, now rarer but not gone. The intent model picks `no_change` for a request that really was a change, the covered tools are hidden, and the model's own text ("Done, I added milk") is spoken although nothing ran. Production would have run the action, so on this scenario the branch is worse. A safety net that replaced such replies with "Nothing was changed." was built and then removed before this deploy, because Codex showed it could also replace a correct answer to "did you add milk?" with a false denial. The next fix has to tell the two cases apart without a word list on the reply, for example by checking this turn's actual tool results, or by making the intent call itself harder to misclassify. Decided by the owner, Nov 27 2026.

## Left after the intent follow up

- **Known limit** `src/brain.py`. The forced tool round covers every validated single tool gate, but only a validated add is replaced by the plan's own call when the model's call is missing or wrong. A wrong call for a stop, skip or completion, or for a list, timer or reminder, is still refused with "Nothing was changed", because replacing a destructive or time dependent call from the plan was not approved. Forcing the tool makes a missing call rare, not impossible.
- **Known limit** `src/lists_tasks.py`. A second spoken "add reading every day" in a later turn adds a second row. A validated add never duplicates within one turn, but the task table has never refused a repeat title, and refusing one would change behavior he would notice.
- **Known limit** `src/task_intent.py`. List item names are still compared as exact words (case and a leading "the" or "my" aside). Only task titles are compared by meaning, because the observed failure was a rewritten task title.
- **Known limit** `src/task_scheduling.py`. A title is compared by content words in order, so a model title that reorders words ("Dog walk" for "walk the dog") is refused.
- **Known limit** `conversation_history.channel`. Rows written before migration 050 stay NULL. Only turns that go through `brain.ask_nova_async` and the Pi local intent paths write a channel.

## Other open items

- **Should fix** `src/task_intent.py` PRE EXISTING. Permanent-stop keywords take priority even when explicitly negated ("don't stop reading for good, just skip it today" is refused as unclear). Production archives the task for good on that sentence, so the branch is safer. (found by the review loop at 13424b1)

- **Should fix** `src/task_intent.py` PRE EXISTING. A first day on a daily task is accepted by voice only when the intent copies the "starting ..." words and the call carries the matching `start_date`. Production's task tool has no first day at all and silently starts the task today. (found by the review loop at 13424b1)

- **Should fix** `src/brain.py` PRE EXISTING. A skip-or-stop choice becomes active before the question has been delivered. Production archives on the first "cancel" with no question, so the branch is safer. (found by the review loop at 13424b1)

- **Should fix** `src/tests/test_task_intent.py` PRE EXISTING. The supplied changes do not verify Nova's required behavior when a stream ends after a delta and before `done`. Production streams the same way and has no test either. (found by the review loop at 13424b1)

- **Should fix** `src/database.py` PRE EXISTING. Backdating every legacy receipt by two days can release a recently used request ID two days early. Production has no receipt table, so nothing existed to compare. (found by the review loop at 13424b1)

- **Known limit** `src/task_intent.py` REGRESSION, accepted by the owner. The intent step has no "other change" action. A calendar write, `remember` and every other mutating tool outside task, list, timer and reminder keep their own confirm first flow and are offered only on a turn the wide check flagged. A statement the wide check does not flag (for example "my cat is named Sam" with no action word) no longer reaches `remember`, because an unflagged turn offers no mutating tool; production offered it. The nightly memory pass is unaffected.

- **Known limit** `src/task_intent.py` PRE EXISTING (exposure), fixed for questions. A question turn ("what do you remember about my commute?") no longer reaches `remember`; production offered it on every turn. `remember` stays on a flagged turn that is not a question, so "remember that my cat is Sam" works. Found as F1.

- **Known limit** `src/voice_action_safety.py` PRE EXISTING. F4: a successful `cancel_reminder` is still spoken from model text, which can name a different reminder than the one cancelled. It is outside the covered set by contract. Production did the same.

- **Known limit** `src/task_intent.py` PRE EXISTING. F7: the task day is read after the intent call returns, so a request spoken at 03:59:59 whose intent call returns after 04:00 is recorded on the new day. Production reads the clock at write time, after an even longer model call, so it did the same or worse.

- **Known limit** `src/task_intent.py` PRE EXISTING. V4: a recognized question opens the no_change tool set (calendar and the rest, but not `remember`) even when the intent call failed. Production offered every tool on every turn. Closing it would mean answering a read question with "I could not tell what you wanted", which is worse than today.

- **Known limit** `src/task_intent.py` PRE EXISTING. F6: only a task whose first word is a verb it knows (take, call, pay, get, buy, send, submit) is held to that verb. "I bought my vitamins" no longer completes "Take vitamins", but "Pick up vitamins" still accepts any report verb. Production completed on any call the model made.

- **Known limit** `src/task_intent.py` N1, the gap that remains with the net removed: on a no_change turn the model's reply is spoken as is, as in production. See the first fix above.

- **Known limit** `src/task_intent.py` PRE EXISTING. F1: a question that starts with "can you" ("Can you tell me what you remember about my commute?") still offers `remember`, because "can you" also starts real requests. Production offered it on every turn.

- **Known limit** `src/task_intent.py` REGRESSION, rare. F8: if the intent model misclassifies "show my list and check whether milk is on it" as a list change, the answer is "Please ask for one change at a time." Production answers it. Rephrasing works. When the model picks `no_change` it answers normally.

- **Known limit** `src/task_intent.py` PRE EXISTING. F11: "remind me tomorrow to call the bank" with the timing left out of the intent writes a reminder with no due date, which defaults to today. "today" and "tonight" are never bound. Production behaves the same.

- **Known limit** `src/task_intent.py` REGRESSION, safe direction. E8, E9, E10: some unusual phrasings are refused and he rephrases: a task whose stored title contains "if" ("What if analysis"), "cancel reading, I don't do it anymore", and a compound timer such as "half an hour and ten minutes". Production would have tried them.

- **Known limit** `src/task_intent.py` REGRESSION, rare. W2: the word "tomorrow" inside a reminder's own text ("call before tomorrow's meeting") can move the reminder itself to tomorrow.

- **Known limit** `src/task_intent.py` REGRESSION, rare. W3: a preposition inside a list item ("add tape to fix the shelf to my grocery list") can be read as the start of the list name, so the item may land on a wrong or new list.

- **Known limit** `src/task_intent.py` REGRESSION. W4: a `no_change` result skips the "multiple changes" finding, so the gate can offer a non covered write tool on that turn. Those tools keep their own guards and confirm first flows.

- **Known limit** `src/task_intent.py` REGRESSION, tiny. W5: the reminder time check drops the UTC offset before subtracting, so on the one day a year daylight saving ends a reminder can be an hour late.

- **Note** `files/src/brain.py:1074` If the batch holds a correct task_add plus any other mutation, nothing is substituted and preflight refuses the whole batch with ONE_CHANGE. That contradicts the comment that a validated add never depends on the model's choice. (found by the review loop at 3e8ada4)

- **Note** `files/src/task_scheduling.py:69` same_title drops 'to', 'for' and possessives, and the light stemmer merges some distinct words ('planned'/'plane' both become 'plan'), so a model title with a slightly different meaning passes and is what gets stored. (found by the review loop at 3e8ada4)

- **Note** `files/src/brain.py:1274` The new 'App turn:' journal line logs intent.target, intent.timing and up to 200 characters of refusal text for every flagged app turn. Those often hold task titles with people's names, and the target can be nearly the whole message despite the docstring. (found by the review loop at 3e8ada4)
