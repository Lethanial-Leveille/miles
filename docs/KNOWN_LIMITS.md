# Known limits

Open findings that are not Blockers. Each was reviewed and deliberately left. Remove an entry when it is fixed.

Labels compare this branch with production (`main` at 51676b9), measured by running the same scripted model through both trees. **REGRESSION** means the branch behaves worse than production for that scenario. **PRE EXISTING** means production does the same or worse.

## First fix after this deploy

- **W1, a false "Done" on a misclassified no_change turn.** Fixed for two cases, still open for a third. (1) An answer to her own question ("Stretching" after "Which one, reading or stretching?") now gets the intent step, so it can no longer fall through with no tool. It applies when her last reply for that device and actor ended in "?", was under five minutes old, and the message before it could have been a change (the wide check flagged it, or the intent step returned a change or a question). (2) On a turn where no mutating tool ran, a reply that opens with Stopped, Added, Archived, Skipped, Saved, Removed, Marked, "Set" before a, an, the, your, my, it, that, this or up, or "Done" as a whole clause is replaced by "Nothing was changed." on screen and in speech. Still not covered: a claim that does not open the reply ("Okay, I stopped it"), a claim in the middle of a reply, and a model that claims after a mutating tool refused something (that turn uses the code written reply). The follow up memory is kept in process, so it is lost on a restart, and "the second one" depends on the intent model reading the order of the options in her question correctly; a wrong pick of a one time task would archive it, a daily task still asks first. "Set reminders are listed below" is left alone on purpose, so a reply like "Set the table" is replaced. Left out, because two designs would change what he notices: replacing a claim only when the same turn also had a mutating tool offered, and a word list for claims later in the reply.

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

- **Decision** `src/task_intent.py` DUPLICATES (owner, Nov 27 2026). Adding a task that is already on the list, with the same title by meaning and the same recurrence, writes nothing and says it is already there. Production could add a second row. A daily task and a one time task with one title are not duplicates.
- **Decision** `src/task_intent.py` "I need to ..." (owner, Nov 27 2026). The code adds the task itself, when the model's call is missing or wrong, only for an explicit "add" or "create". For "I need to ..." and similar phrasings the model's own valid call still works, and a failed call is a plain refusal ("Nothing was changed. I couldn't do that. Please try again.").

- **Known limit** `src/task_scheduling.py` DUPLICATES BY MEANING. The title match ignores small endings, so "read" and "reading" count as one task: adding "reading every day" while "Read" is a daily task says it is already on the list. The reverse also applies to cancelling: "cancel reading" does not find a task titled "Read".

- **Note** `files/src/task_intent.py:792` _PRONOUN.sub(intent.target, ...) uses text from the model as a regex replacement template, so a target containing a backslash raises re.error and the turn crashes. (found by the review loop at 2d12223)

- **Note** `files/src/task_intent.py:776` _PRONOUN replaces the first of it, this, that or one in the earlier request even when that word doesn't refer to the task. The rebuilt command then fails _says_action and a valid answer is refused. (found by the review loop at 2d12223)

- **Note** `files/src/task_intent.py:715` Adding "actually", "anyway", "then" and "instead" to _REMOVE_AFTER_WORDS widens what counts as an archive or skip phrase for every request, not just follow-ups. KNOWN_LIMITS doesn't mention this. (found by the review loop at 2d12223)

- **Note** `files/src/voice_action_safety.py:38` A reply that opens with '[' but has no closing ']' keeps the gate in wait for the whole stream, so on voice nothing is spoken or shown until the reply ends. This undoes sentence streaming for that turn. (found by the review loop at 2d12223)

- **Note** `files/src/tests/test_answers_to_questions.py:578` test_an_unrelated_answer_is_answered_normally passes if the reply is either the normal answer or any "Nothing was changed" text, so it can't catch the regression described in F4. (found by the review loop at 2d12223)

- **Should fix** `files/src/voice_action_safety.py:59` A streamed chunk that ends right after 'Done ' (rest is only whitespace and final is False) now returns 'claim' instead of 'wait', so a harmless 'Done with ...' reply is swallowed. (found by the review loop at 2d12223)

- **Decision** `src/brain.py` CLAIM CHECK AND QUESTIONS (owner, Nov 28 2026). The check that replaces a no tool reply opening with Stopped, Added, Archived, Skipped, Saved, Removed, Marked, Set or Done skips a turn whose message is a question or a read, so an honest answer about an earlier change ("Did I add eggs?" answered "Added eggs at seven") is left alone. A bare statement such as "Stretching" is still checked. A claim that does not open the reply is not caught.
- **Decision** `src/task_intent.py` ANSWER TARGETS (owner, Nov 28 2026). When he answers one of MILES's own questions about a change, the target may come from MILES's question as well as his words ("the second one"). It must still resolve to exactly one existing task, a daily task still asks "Skip today or stop it for good?" first, and one change per request still holds. The risk accepted: if the model picks the wrong option for "the second one" on a one time task, that task is archived (the row stays in the database).
- **Decision** `src/task_intent.py` UNREADABLE ANSWER (owner, Nov 28 2026). If the intent call fails on an answer to a change question, the reply is "Nothing was changed. I couldn't tell what you meant. Please say it again." It is never answered as chat.

- **Known limit** `src/task_intent.py` PRE EXISTING. "Stop reading for good" when there is no task called Reading at all is ambiguous: it can mean the activity, a standing preference, or a task that is gone. The intent step files it under no change requested, and the model sometimes answers "Done." after storing a preference with `remember` (about one run in three in the Nov 28 scenario run), or asks what he means. Production offers `remember` on every turn and behaves the same way. With a task of that name the sentence is handled correctly, because the intent step is given his active task titles.

- **Note** `files/src/task_intent.py:174` Task titles go into the system prompt raw, so a title containing ';', '(daily)', a period or instruction-like text can garble the list or steer the intent call. (found by the review loop at f51e726)

- **Note** `files/docs/KNOWN_LIMITS.md:88` The doc says that with a task of that name the sentence 'is handled correctly', but the tests only check what goes into the prompt with a stubbed model, so the claim rests on a scenario run, not on a test. (found by the review loop at f51e726)

- **Note** `files/src/tests/test_intent_task_context.py:81` The tests cover only how the prompt is built. Nothing covers the target-copy refusal in F1, the bare-stop archive in F2, or what happens past the cap in F3. (found by the review loop at f51e726)
