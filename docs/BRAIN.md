# Brain: prompt, model, streaming, local intent

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Model A/B measured Aug 10 2026. Cache margin measured Aug 11 2026. Local
> intent measured Aug 12 and Aug 13 2026. The streaming section was re
> verified against source Sep 13 2026 and corrected.
>
> **Live config values are declared in [CLAUDE.md](../CLAUDE.md), not here.**
> This file carries the reasoning and the measurements behind them. A number
> quoted inside a measurement is what was recorded on the date given and is
> left exactly as measured, so it may differ from what is live now. The config
> table in CLAUDE.md is the one that tracks the present.

Up: [CLAUDE.md](../CLAUDE.md) · [SESSION_START.md](SESSION_START.md) ·
[BACKEND_TODO.md](BACKEND_TODO.md) · [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) ·
[LATENCY.md](LATENCY.md) · [VOICE_OUTPUT.md](VOICE_OUTPUT.md) ·
[BRAIN.md](BRAIN.md) · [INFRASTRUCTURE.md](INFRASTRUCTURE.md) ·
[INCIDENTS.md](INCIDENTS.md)

---

What happens between a transcript and the first spoken word: which model, what
is in the prompt, how the response is streamed into sentences, and which turns
never reach Claude at all.

## Model

Haiku won a measured A/B against Sonnet over twenty turns: 614ms faster on
median time to first token, 31 percent lower, p=0.0007 on a permutation test.
Pin the A/B off before measuring anything cache related, since prompt caches
are model scoped and alternation makes every turn a miss.

## Prompt and caching

**`HISTORY_ASSISTANT_WORDS`** trims past assistant turns before they are sent
as context. History anchors response length far more strongly than any
instruction does, which is why the response length rewrite was done this way
rather than by asking for shorter answers.

**Cache margin.** Haiku 4.5 requires a 4096 token cacheable prefix. Dropping
below it disables caching *silently*, with no error and no warning.
`cache_read_tokens` is logged per turn for exactly this reason: a **contiguous**
run of zeroes means the prefix fell under the minimum. Scattered zeroes are just
the 5 minute TTL expiring and are normal.

Measured with `count_tokens` against `claude-haiku-4-5`, Aug 11 2026:

| Prefix state | Tokens | Margin |
|---|---|---|
| Before the tool migration | 4757 | +661 |
| After, 6 tools registered | 5942 | +1846 |

An earlier version of this file claimed 4165 tokens and 69 tokens of margin.
That was stale by roughly 600 tokens and it was load bearing: a whole phase of
the tool migration was gated on it. Re measure rather than trusting the table.

The component costs, for projecting a deletion before making it: seed block 2293,
old action tag instructions 750, memory instructions 233, system prompt header
466, clock block 100.

Phase 2's deletion and the tool schemas that replace it **must land in the same
commit**. Deleting first leaves the prefix near the floor with no headroom.

## Streaming

> **Correction, Sep 13 2026.** **Most of this section is now obsolete and is
> kept because its central correction is still worth reading.** Native tool
> use removed action tag parsing from `stream_router.py` entirely. There is no
> lookahead buffer, no `ACTION_PREFIX`, no `LOOKAHEAD_CHARS`, and no
> `router.action_tags`; neither constant exists anywhere in the codebase any
> more. Tool calls arrive as their own content blocks and never appear in the
> text stream, so text is only ever text and a sentence flushes as soon as it
> is complete. The second call is no longer gated on a weather whitelist
> either: `brain.py` now asks the registry whether any called tool declares
> `returns_to_model`. Read `src/stream_router.py` and `src/brain.py` for what
> actually runs. What remains true below is the first claim, that no stop
> sequences are passed to the API, and the reason it is recorded at all.

**No stop sequences are passed to the API.** Generation runs to completion. The
previous version of this document claimed `stop_sequences=["[ACTION:"]` was sent
on every call; that was never true in code and the claim caused a full session
of work to be planned against a wrong premise. This section is now written from
`stream_router.py` and `brain.py`.

1. `brain.py` iterates `stream.text_stream` and feeds every delta to `StreamRouter`.
2. `StreamRouter` buffers `LOOKAHEAD_CHARS` (50) before emitting anything, which
   guards against a stray `[` in prose being read as a tag. Seeing `ACTION_PREFIX`
   ends that wait early, because it is unambiguous.
3. Complete `[ACTION:...]` tags are stripped from the buffer into
`router.action_tags`.
   A tag missing its closing bracket is left in place, because deltas do not
   respect tag boundaries and committing early truncated tags mid word.
4. Everything before a pending tag is flushed as sentences to the TTS queue.
5. `_tts_consumer` speaks each sentence, running `speak()` in an executor because
   it blocks on `speak_lock` and on aplay.

**An action turn makes two Claude calls, not one.** After the first stream ends,
if any action returned data the model needs to speak about, `brain.py` builds a
followup message array containing the bridge text as an assistant turn and the
tool data as a synthetic user turn, then makes a second streaming call. Today
that second call is gated by a hardcoded check:

```python
needs_data = any(r["type"] == "weather" for r in results)
```

which is a whitelist of exactly one action type. Timers, reminders, and
cancellations skip the second call entirely and the bridge sentence is the final
response.

## Nova knows the transcript is not his words

`WHAT_REACHES_YOU` in prompts.py, added Aug 13 2026 after a real incident.

"Where do you think that I live right now?" reached her as the fragment
`"live right now."` at 3.3 dB SNR. She correctly stayed quiet. Asked on the next
turn what he had just said, she did **not** report the fragment: she
reconstructed an intent for it, landed on wording from the `lower_access` tool
description, and told him he had asked to drop his own clearance so he could see
how she behaves with someone who is not him.

**Nothing had happened.** `lower_access` was never called, `effective_tier()` was
still `hokage`, and no row in `people` had changed. The tier system worked
exactly as designed. What failed was that she narrated a guess as a fact about a
security adjacent tool, which is the worst possible subject for a confabulation.

Root cause was an omission: the prompt never told her the user turn is speech
recognition output. She had no concept of a mistranscription, so a fragment was
something he had said and the only question was why. The block sits directly
after `GENERAL_KNOWLEDGE`, which is what pushes her to answer confidently from
what she has, and is present at **every tier**. `tests/test_prompts.py` pins
both the content and the adjacency.

The general rule, which `_manifest_block`'s docstring already stated one level
down: **a model handed a partial view will fill the gap rather than admit to
it.** That was solved for memory retrieval and left unsolved for the transcript.

## Channels

Requests carry a `channel` of `voice` or `text`, defaulting to `voice`. It is
distinct from `device`, which is provenance and is stored as `source_device`.
The two were one parameter until their meanings diverged: the app may want
spoken output and the Pi may one day want text.

Channel selects the response formatting fragment of the system prompt and
whether anything is played. A `text` turn never reaches the speaker:
`_consumer_for` hands it `_collect_text` instead of `_tts_consumer`, and every
line code speaks on its own (a staged question, the "Done." fallback) goes
through `_say`, which stays quiet on text. Pronunciation normalization runs
inside `speak()`, so it only ever touches voice. Tool calls and memory writes
are identical on both.

> **Correction (Sep 14 2026).** This section said channel selected the prompt
> fragment and gated normalization, and commit `b943ea0` said "the text path
> never calls speak()". By Sep 14 that was false: `ask_nova_async` started the
> TTS consumer and the direct `speak()` calls on every turn whatever the
> channel, so a message typed in the app was spoken through the room speaker by
> `miles-server`, and `/chat` returned only after playback, which the app showed
> as Nova still thinking. `tests/test_text_channel.py` pins it now, with a voice
> control so the check cannot pass by seeing no speech at all.

### Numbers on a text turn

`NUMBER_FORMAT_TEXT` asks for numerals and was losing to two other things.
Measured Sep 14 2026 on his real Oura sleep result and his real history, four
samples per condition, counting replies containing any digit:

| Condition | Numerals | Number words |
|---|---|---|
| Text prompt as written, with history | 0/4 | 4/4 |
| Numerals in the tool speech and results examples | 2/4 | 4/4 |
| Those examples, and no history at all | 4/4 | 3/4 |
| Those examples, history, and `TEXT_TURN_NOTE` | 4/4 | 1/4 |
| Shipped: both, with history | 4/4 | 0/4 |

Two findings, and they are the same finding twice. **The examples mattered more
than the rule**: `TOOL_SPEECH` said "ninety five degrees" and
`TALKING_ABOUT_RESULTS` said "Your readiness is eighty one", and those were
copied while the rule three blocks earlier was not. `_for_text` derives the text
copies from the voice ones so only the sentences that assume speech change, and
it raises at import if a swap stops matching, because the cost of a silent miss
is the spoken example staying in the text prompt.

**And the transcript beat the system prompt again**, the same way it did for
response length. Her stored replies are mostly spoken ones with every number
written out, and with history in place the numerals rule lost 4 times out of 4.
`TEXT_TURN_NOTE` wins because of where it sits, not because of its wording: the
final user turn, after the cache breakpoint, next to the clock. Its examples are
abstract (`12.75`, `15%`) rather than sleep shaped, because anything in a prompt
shaped like his real data gets treated as his real data.

The voice prompt is byte for byte unchanged by this, so voice keeps its cached
prefix and keeps spelling numbers out.

**Known and not fixed:** history is shared between channels, so text answers in
numerals now sit in what a voice turn reads. Measured after seeding one numerals
answer: 1 of 4 voice replies picked up a digit ("score of 62"). Watch it; the
fix, if it is ever needed, is the mirror of the note.

### Streaming a reply to the app

`ask_nova_async` takes an optional `on_text(kind, text)`. It is called with each
piece of the reply as Claude writes it, which is the same stream that feeds
speech, so an app can show words arriving rather than a thinking indicator.
`POST /chat/stream` is that path; the voice loop passes nothing and is
unaffected.

`kind` is `delta` for text, and `reset` to say that what was streamed so far is
being dropped. `reset` exists because on a tool turn the bridge sentence said
alongside the call is not part of the answer the tool result produces and is not
what gets saved, so a reader shown it has to be told. Lines code speaks itself,
a staged question or the `Done.` fallback, are streamed as deltas too, so what
is read matches what is said.

## Local intent

`set_timer`, `time_of_day`, `cancel_reminder`, `dismiss` and `weather` are
answered without Claude. Measured Aug 12 2026:
classify 142ms warm, and 0.0ms when the lexical gate declines, because the gate
runs before the embedding. That replaces claude_ttft 1260ms plus tts_ttfb 349ms,
so perceived on those turns goes from about 4404ms to roughly 3000ms.

**Both signals must agree, combined with AND, not fused.** Memory retrieval
fuses keyword and semantic by reciprocal rank because it ranks candidates. This
is a gate. "Set a ten minute timer" and "cancel the ten minute timer" are near
neighbours in embedding space and opposite in meaning, so any threshold loose
enough for phrasing variety is loose enough to confuse them. The lexical gate
carries the exact distinctions, the embedding carries the fuzzy ones.

Every gate needs a **positive** requirement, not only guards. Written first with
only negative checks, the dismiss gate passed anything short that was not a
question, so "stop the timer" and "set a timer for ninety minutes" both landed
on dismiss once their own intents declined. Caught by tests, not by ear.

A duration with no rendered confirmation declines rather than improvising, so
"ninety minutes" goes to Claude. A fall through costs latency and nothing else.

`local_intent.warm()` must be called at boot. Loading the model and encoding the
examples are separate lazy costs: warming only the model leaves 10.8s to land on
the first real command.

`_run_local` writes both sides to history with `save_message`. Skipping that
would leave "set a timer for ten minutes" then "make it fifteen instead" reaching
Claude as a follow up whose subject it never saw.

**Weather is the exception to every rule in this section.** Added Aug 13 2026.

It is the only local intent whose answer is composed rather than chosen, so it
is the only one that cannot use the phrase bank: temperature times condition is
not a space that can be enumerated. `_weather_sentence` builds the sentence and
`execute` returns a **null key**, which is the signal to `_run_local` to speak
it live through ElevenLabs.

**It does not work offline, and it never will.** Nothing about local intent
makes weather local; OpenWeatherMap is still a network call. What is skipped is
the language model, which for weather is *two* Claude calls, since it is the
only tool with `returns_to_model`. Measured: claude_ttft 1324ms plus
second_ttft 720ms removed, `fetch_weather` 628ms and TTS 382ms kept. First
audio goes from 5072ms to roughly 2800ms, and that first audio is the answer
rather than "let me check".

The felt improvement is **~2.3s, not the 3.6s** that "time to the answer"
suggests. The bridge sentence is already speaking while the tool and the second
call run, so most of that tail was never silence. Quote the first audio number.

**Only network errors reach `netcheck`.** Written first with a bare
`except Exception`, a wrong function name raised `AttributeError` and Nova
announced "it's the model I can't reach", which is precisely the false
statement netcheck exists to prevent. Anything that is not a
`requests.RequestException` is a bug and belongs at `run_turn`'s boundary.

The gate must decline anything about the **Pi's** temperature, which
`get_system_state` answers. Both spellings, because whisper transcribes "Pi" as
"pie" and that is the transcript the gate actually sees.

**Local intent covers roughly 10 to 12 percent of turns, not a majority.**
Measured over 322 real utterances in `conversation_history`: weather 20% (65,
and it can never be local because it needs the network), timer 4% (13), reminder
3% (11), time of day 2% (8), dismiss 2% (8), system 1% (6), cancel 1% (4).
Re-run that count before adding an intent, rather than guessing which is common.

`cancel_reminder` fires **only when exactly one reminder is outstanding**. Zero
or several both go to Claude: cancelling the wrong one is worse than spending
four seconds cancelling the right one.

`time_of_day` is enumerated to five minutes, 12 hours by 12 slots is 144 clips.
Exact minutes would be 720, and hedging with "about" on every reading grates.
The clock is read in `execute`, not in `classify`, so a slow turn cannot report
a time that has already passed.

### The ignore tool

**The `ignore` tool.** Overheard speech used to get "I'm not part of that
conversation. Let me know if you need anything.", which is itself joining the
conversation and costs him a wait to hear. `TurnResult.ignored` is distinct from
`dismissed`: dismissed ends a conversation he ended, ignored ends one he was
never having with her, and the difference is whether she says anything at all.
In the follow up loop it breaks the conversation rather than reopening the
window, so an exchange nearby cannot hold her attention turn after turn.

## Tools and the permission gate

**The model proposes, the executor decides.** A tool call is a request. Between
the request and the function, `_run_tools` in `brain.py` asks `permits()` in
`tools.py` whether the tier this turn was built for may run it. A refusal goes
back to the model as an `is_error` result, so Nova says no out loud rather than
going silent. The check lives in code rather than the prompt because a prompt
instruction is a suggestion to a probabilistic system and a conditional is a
guarantee.

Three rules the gate keeps, each pinned by a test:

- **One tier per turn.** The gate uses the tier `ask_nova_async` built the
  prompt with. It used to read `effective_tier()` itself, a second source of
  truth that would have disagreed the first time voice verification passed a
  guest's tier through: prompt gated as a guest, tools gated as hokage.
- **`min_tier` only raises.** Calendar and health data are READ in kind and
  private in content, so those tools carry one. If it could lower the floor,
  one keyword would open an external write to a guest.
- **CONTROL is never gated.** A demoted speaker still has to be able to end the
  conversation.

`tier_tool.py` used to check for hokage inside its own body. That check is gone:
the gate does it for every tool, including the ones nobody remembers to guard.

### Changes to what exists wait for the next turn; additions happen at once

Over voice there is no button. Moving, renaming and deleting are staged in
`pending_action.py` and asked as a question. Only `confirm_pending_action`,
called on the **very next** Claude turn and inside `CONFIRM_WINDOW_S`, performs
the change.

**Adding does not ask, since Sep 14 2026.** An event, or a whole planned week, is
added immediately, and code reads back exactly what was added: "Added Career Fair
today from 1 PM to 6 PM." `undo_last_addition` removes the most recent addition as
a whole for thirty minutes. He found a question before every addition too much,
and an addition is the one change undo fully reverses; a move or a delete is not.

- Confirming on the same turn as the proposal is refused, so the model cannot
  ask and answer itself. A human turn has to happen in between.
- `confirm_pending_action` takes only yes or no. It runs exactly what was read
  back, so the model cannot confirm something other than what he heard.
- Anything unrelated said in between drops the proposal. A correction ("make it
  eleven") is a new proposal and is read back again.
- **The question is spoken by code, word for word.** When a turn stages a
  proposal, `brain.py` skips the model's reply and speaks
  `pending_action.staged_question()`, also when the proposal follows a lookup.
  Until Sep 14 2026 the model read the question out, and once it said a
  different date from the one staged; his yes created the date he never heard.
  It also saves a Claude call on every confirmation.
- Several proposals on the **same** turn join one batch, asked as one question
  and confirmed or cancelled together. Until Sep 13 2026 each replaced the last,
  and his yes to the first of seven lessons created the seventh. In a batch, one
  failure is reported beside what worked rather than hiding it.
- Whether "yeah, do it" means yes is judgment and stays with the model. The code
  guarantees order and freshness, not interpretation.

State is per process, so a proposal made by voice is confirmed by voice.

**Known gap, fails safe:** a reply that local intent classifies as a dismissal,
like "yeah, that's it", never reaches Claude. The proposal expires and nothing
is written.

### Nova asks, she does not read back

A proposal returns one question built in code, like "Move LeetCode session
tomorrow from 10 AM to 4 PM?", and the model is told to ask exactly that. Left to
the model, the first real move was announced before the call and then read back in
full, both versions with their dates, which heard aloud sounded like two answers
glued together. The prompt also tells her to say nothing before calling a
proposal tool, because anything said then is spoken before the result exists.

It also forbids asking before the call. On Sep 13 2026 Nova asked "Want me to add
one at ten a.m.?", heard yes, then called the tool and asked the tool's question,
so he had to agree twice. The tool's question is the confirmation; there is no
second one.

### Live data is fetched every time

The prompt tells Nova to call the tool again whenever he asks about sleep,
readiness, activity, heart rate, the calendar, the weather or her own state, and
never to repeat a figure from earlier in the conversation. Without that rule she
answered "how did I sleep" from history that still held the wrong figure, and the
fixed tool was never called.

### Calendar times are resolved in code

The model passes phrases, and `calendar_tools.parse_when` resolves them against
the Pi's clock with dateparser. Day names prefer the future, a bare day means the
whole day, and a new event with no time of day is refused rather than invented.
Freebusy merges overlapping blocks across calendars and computes the free gaps,
because that is arithmetic. Public holiday calendars are left out of freebusy:
Google cannot report busy time for them, and a holiday is not busy time. The
failures behind these rules are in
[INCIDENTS.md](INCIDENTS.md#calendar-and-sleep-tools-answered-confidently-and-wrong-sep-13-2026).

### Edit and delete find the event in code

`update_calendar_event` and `delete_calendar_event` go through the same next turn
confirmation as create. Each asks one short question built in code, naming only
what changes.

**They take a title and a day, not an event id.** Only what Nova says reaches the
conversation history, so the ids in a listing are gone by the turn where he says
"delete it". A tool that took an id would be asking the model to invent one.
Code searches the MILES calendar for exactly one match. None, and Nova hears what
is on that day instead; two, and she hears both with their times and asks which.

**Only the MILES calendar can be changed.** His other calendars include shared
ones where a delete reaches other people or fails outright. Edit and delete never
create the MILES calendar either: with none, there is nothing of Nova's to change.

A moved time is read relative to the event's own day, so "4pm" stays on that day,
unless the phrase names today or tomorrow; a new day alone keeps the event's
time; moving keeps its length. A repeating event changes only the one occurrence,
and the read back says so. Edits use `patch`, so fields the tool never touches
are left as they were.

### Renaming a name everywhere, and spelling what cannot be heard

`rename_calendar_events` replaces a word in the title of every MILES event that
has it, across a range, with one question. On Sep 14 2026 correcting Charlie to
Charley on three lessons took five read backs and three yeses through the one
event tool, one lesson at a time.

A rename whose new word sounds like the old one says which letters changed: "Rename
Charlie to Charley, E Y instead of I E, on 3 events". Read back without that, the
two sounded identical, so he could not hear what he was confirming. Spelling the
whole name out was tried first and he found it too much; a change of more than
four letters is still spelled whole. Whether they sound
alike is decided by Soundex, which gives similar sounding names the same four
character code; a rename that can be heard, like "session" to "grind", is not
spelled, because spelling everything would bury the one case that needs it.

`brain.py` logs a turn where Nova asks to add, move, rename or delete something
herself with nothing staged, which is how "Rename Charley lesson on Wednesday to
Charley lesson?" reached him with nothing behind it.

### Oura values carry their units

Every Oura field names what it is, like `sleep_score_out_of_100`, and durations
arrive as words. A bare number is a number the model assigns a unit to, which is
how a contributor score became an hour and forty minutes of sleep. Heart rate is
summarized in code rather than handed over as raw samples.

## Nova speaks, she does not recite

Rewritten Sep 13 2026, after he described her as reading rather than speaking.
Two things in the prompt produced that, and neither was the voice.

**Nothing told her what to do with a tool result, so she read it.** Asked how he
slept, she gave every field in order and one line of meaning at the end.
`TALKING_ABOUT_RESULTS` tells her to lead with what the result means for him,
back it with one or two numbers, and connect it to his day. It sits in the
middle block, so it reaches every tier, because every tier can call a READ tool.

**The persona asked for performed composure.** "Articulate, poised... clean, well
structured sentences", "warm but never overly familiar", and humor about her own
capability. It now asks for the professional who genuinely cares about the
person in front of her: clear rather than formal, honest and kind, dry humor on
his side, the feeling answered before anything practical.

Tested before landing, on his real Oura results with the real tool result in the
conversation. Readiness and sleep answers moved from readouts to meaning first
in every sample; the calendar answer did not change, correctly, since a schedule
is information he wants read. The test also showed the costs, and each has a
guard in the prompt:

- **Interpretation can outrun the data.** One answer said he "stayed asleep the
  whole time" from 89 percent efficiency. Hence "only say what the numbers
  actually show."
- **Stock sympathy.** "That lands" came up twice in three answers. Hence "say it
  your own way each time."

Rejected on evidence first: rewriting only sentence mechanics, meaning shorter
full stops joined with commas. Over three samples it changed neither length
nor pauses.

### The calendar is read like a person, not a printout

`get_upcoming_events` starts from now, never earlier, because a bare date for
today resolves to midnight. All day events are a day, not a duration: "Monday
September 14: David's birthday". His own calendars and the ones he follows are
separated by Google's `accessRole`, owner against reader, and followed
events come with an instruction to mention them only when he asks what is going
on. He keeps club calendars as options, not as a schedule. The evidence is in
[INCIDENTS.md](INCIDENTS.md#nova-read-the-calendar-like-a-printout-sep-13-2026).

## Memory: Nova actually remembers now

**Until Sep 13 2026 the `remember` tool had never been called.** Every one of his
memories came from the seed file. Two causes, neither a wiring fault:

- **He had never asked.** In 240 turns after the tool shipped, no turn asked her
  to remember anything.
- **She almost never stored anything on her own.** The description set the bar
  at facts that "will still matter next week" and called a duplicate "worse than
  not storing it", so she played safe and stored nothing. When he did ask, she
  sometimes asked a clarifying question instead of storing. Once she said "I've
  got that down" and called nothing.

The rules now: an explicit request is stored that turn, in his words, with any
question asked after rather than instead. Things mentioned in passing that would
change what she says later are stored as inferred, for his review. She never
says she noted something unless she called the tool, and `brain.py` logs every
turn where the text claims a save without the call. Replacing a memory with
identical words is refused in code.

Measured with the real prompt, memories and tools, three samples each:

| Said | Before | After |
|---|---|---|
| "Remember that Charlie's lessons are always on Zoom." | 1/3 | 3/3, as asked |
| "I just started logging my workouts in Hevy." | 0/3 | 2/3 |
| "My friend Jalen is going to be my lifting partner now." | not tested | 3/3, inferred |
| "I'm gonna add my lifting sessions to my calendar soon." | 0/3 | 0/3, an intention rather than a fact |
| Claimed a save without calling | seen | 0 of 15 |

Still imperfect: one Hevy reply marked a passing mention as "asked", which skips
review. Inferred memories wait in a queue he can now reach by voice through
`list_pending_memories` and `review_pending_memory`; the review tool accepts
only ids that are actually waiting, so a misheard "discard that" cannot delete
an established memory.

### Conflicts are found in code

`find_schedule_conflicts` sweeps every selected calendar except holidays, drops
all day events, and groups overlapping timed events. Back to back is not a
conflict, because he tutors online and a lesson can start the minute another
ends. A chain of overlaps is one group, because choosing among three is one
decision. Nova then suggests what to keep, classes and his own commitments over
events on calendars he follows. On its first run against his real week it found
four, including his Thursday test prep session against two club events.

### Sessions are placed in code

`plan_sessions` exists because Nova placed seven tutoring lessons in her head on
Sep 13 2026 and broke nearly every rule she was given: 2 PM after "after three",
a lesson inside the class he had just described, two lessons overlapping each
other, and a confirmation for each one separately. Now she only turns what he
said into sessions and limits; `plan_sessions_on` places them.

The rules it enforces, each with a test:

- **A student's earliest start and latest end** come from what he says, per
  session. 3 PM and 8 PM are only defaults.
- **Weekends and holidays prefer mornings**, falling back to the afternoon. The
  after school rule exists only because of school.
- **Nothing lands on his own calendars.** Club events are avoided whenever any
  day has room and overlapped only when none does, and every forced overlap is
  named in the spoken question before he answers. A club event Nova knows he
  should attend goes in `blocked`, which is never overlapped. Until Sep 14 2026
  club events were ignored outright, and a plan landed lessons across a mini
  career fair, three info sessions and a workshop.
- **One session per name per day**, so one student is never back to back, while
  different students can run back to back, because the lessons are online.
- **Each further session goes on the workable day farthest from that name's other
  days**, so two lessons are early and late in the week.
- **Anything that does not fit is reported**, never squeezed in, and it is said
  before the question, along with one short note per session that had to
  overlap club events, because the whole question is spoken by code.
- **Spoken limits that are on no calendar go in `blocked`**: a career fair, a class
  not yet on the calendar. A day named in a block includes today, and an end
  given as only a time stays on the same day.

The whole plan is one question, grouped by name so it can be followed by ear, and
one yes creates every event.

A dry run on his real calendar before first use caught two bugs the rule tests
had not: fixed position spreading bunched lessons into consecutive days when the
range's last day had no usable time, and "monday 1pm" said on a Monday resolved
to the Monday after. Both are fixed and tested.

## Failure boundaries

**A failed turn no longer kills the process.** It used to. Nothing caught the
API exception, it escaped the `except KeyboardInterrupt` that is the loop's only
handler, and systemd restarted under `Restart=always`. The room saw a chime and
then silence, forever, with no way to tell anything was wrong.

The boundary is `run_turn` in `voice_main.py`, deliberately **not** in
`brain.py`. `ask_nova` has two callers with opposite needs: the server must
surface a failure as a 503 so the app can retry, and the voice loop must stay
alive. Catching inside brain would force one answer on both, and choosing the
loop's answer would hand the server a fabricated response to store as a real
assistant turn. The `TurnResult` docstring already makes this argument for
`dismissed`.

`netcheck.diagnose()` picks which phrase to speak, because
`anthropic.APIConnectionError` covers four different situations and "I've lost
wifi" is a false statement when the truth is that the API is down. Checks run
narrowest first: default route, then a raw IP probe to 1.1.1.1:443, then DNS.
The probe is by address on purpose, since that is what separates "nothing
routes out" from "DNS is dead". 60ms on a healthy machine, faster when the link
is down. Keyed on the **default route**, not link state, because Docker leaves
bridge interfaces permanently UP on this Pi.

What this does and does not do: **Nova can now say she is offline. She cannot
yet do anything offline.** Making timers and reminders work without the network
is local intent classification, which is not built. See `docs/BACKEND_TODO.md`.

## Related

- Time to first token in the turn budget: [LATENCY.md](LATENCY.md)
- What the transcript looked like before it reached here, and why a fragment is common: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#whisper)
- The tool use migration and its cache margin gate: [SESSION_START.md](SESSION_START.md)
