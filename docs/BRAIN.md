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

Channel selects the response formatting fragment of the system prompt and gates
pronunciation normalization. Tool calls and memory writes are identical on both.

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
