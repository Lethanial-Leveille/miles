# Incidents: what broke, and what the failure taught

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Each entry is dated. Every one was reconstructed from logs, the database, or
> captured audio at the time, and the evidence is kept inline.
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

Production failures and their root causes, newest first. This file is for
things that **broke**. Deliberate design choices go in the decision log in
[SESSION_START.md](SESSION_START.md) instead, and the two should not duplicate
each other: if an incident forced a decision, the incident is written here and
the decision is recorded there with a link back.

An entry earns its place by carrying a lesson that would otherwise be relearned.
Several below exist specifically because a stale or absent record cost a later
session real work.

| Date | What happened | Where |
|---|---|---|
| Sep 13 2026 | Near misses and false wakes turned out to be one population: the neighbours, sorted by level | [below](#the-near-misses-are-the-neighbours-sep-13-2026) |
| Sep 8 2026 | Nova spoke "Done." on turns where staying silent was correct | [below](#nova-said-done-out-loud-when-she-correctly-stayed-silent-sep-8-2026) |
| Sep 8 2026 | No threshold separates true from false wakes in the new room | [below](#no-threshold-separates-true-from-false-wakes-any-more-sep-8-2026) |
| Sep 6 2026 | Reminders were dropped silently by every restart | [INFRASTRUCTURE.md](INFRASTRUCTURE.md#reminders-are-fired-by-a-poller-not-by-a-thread) |
| Sep 6 2026 | The mic gain guard was reading the speakers | [below](#the-mixer-card-guard-measured-the-speakers-sep-6-2026) |
| Sep 6 2026 | One capsule recorded itself as three microphones; enrollment threw away its audio | [below](#two-guards-that-could-not-do-their-jobs-sep-6-2026) |
| Aug 13 2026 | Nova confabulated a security tool call that never happened | [BRAIN.md](BRAIN.md#nova-knows-the-transcript-is-not-his-words) |
| Aug 10 2026 | An empty room drove a runaway conversation loop | [below](#an-empty-room-drove-a-runaway-conversation-loop-aug-10-2026) |

---

## The wake word stopped separating in the new room (Sep 8 2026)

Sep 8 2026. "It responds even though I haven't said hey nova" turned out to be
two independent defects.

### Nova said "Done." out loud when she correctly stayed silent (Sep 8 2026)

`ignore` and `dismiss` carry `returns_to_model=False`, the same as timers and
reminders, so they landed in the branch holding the fallback that exists to stop
a timer being set in silence. `NOT_ADDRESSED_TO_YOU` tells Nova to say nothing
at all when speech was not aimed at her, so **correct behaviour produced an
empty `spoken_parts`, and the fallback then filled it.** The better she behaved,
the more certain she was to speak.

Caught by reading three sources for the same second:

```
01:27:38  Wake word detected! (0.50)     <- not "hey nova"
01:27:50  You: over here
01:27:51  tool_call_log: ignore          <- correct
01:27:52  journal: "Not addressed to Nova, staying quiet."
01:27:52  conversation_history: assistant "Done."   <- spoken anyway
```

`Permission.CONTROL` is the distinction, and its docstring already stated it:
a control tool changes Nova's own conversational state rather than doing work.
Work needs confirming, a state transition does not. The check is `all()` over
the called tools, not `any()`, so a turn that both sets a timer and says goodbye
still confirms the timer.

`save_message` is also skipped on an empty response now. A blank assistant row
is read back as context by the next twenty turns.

### No threshold separates true from false wakes any more (Sep 8 2026)

CLAUDE.md recorded the old room as separating cleanly, successful wakes from
0.520 up, failures topping out at 0.365, **an empty band between**. That was
measured Aug 12 in the old room and it does not hold here:

| score | what followed | real? |
|---|---|---|
| 0.75 | "Oh shoot, this is weird." | false |
| 0.64 | "(beep)" | false |
| 0.50 | "over here" | false |
| 0.46 | "We'll see you later." | false |
| 0.43 | a genuine conversation | true |

**It now overlaps in both directions.** Raising the threshold would have blocked
the real wake at 0.43 and kept the false one at 0.75. This is a model problem,
not a threshold problem, and `hey_nova.onnx` is dated Apr 8 2026, predating both
the gain tuning and this apartment. Five events is not a distribution; the
direction is unambiguous.

The capture that this failure made necessary, and the floor change that
followed, are documented as live mechanism in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#wake-capture).

### The near misses are the neighbours (Sep 13 2026)

Labelled Sep 13 2026. **All 29 wake_misses above 0.20 are `not_wake_phrase`**,
judged by ear: other people talking through the apartment wall. Not one is an
attempt Nova failed to hear.

That alone rules out the reading this capture was built to test. The misses are
not him being ignored, they are the building.

The session then produced a better result by accident. Playing the clips back
through the speakers, while `miles-voice` was still listening, made Nova wake:

| | score | peak |
|---|---|---|
| original, through the wall | 0.337 to 0.392 | -40 to -48 dBFS |
| same content replayed | **0.474, 0.586, 0.687** | -31 to -35 dBFS |

**Identical content, roughly 10 to 13 dB louder, and it crosses the threshold.**
Every one of those clips had just been labelled by ear as containing no wake
phrase at all.

So the near misses and the false wakes are **one population, not two**:
neighbour speech, sorted by how loud it happens to arrive. That reframes the
journal sample of false wakes at 0.43, 0.46, 0.50, 0.64 and 0.75, which were
read as a stale model and are better read as the wall.

Consequences worth keeping:

- **Raising `WAKE_THRESHOLD` cannot fix this.** The same content spans 0.34 to
  0.69 on level alone, and a real wake was observed at 0.43.
- **This is an argument for the array, and a strong one.** The failure is a
  second population of talkers at a fixed bearing that Nova has no other way to
  reject. A steerable beam or a null aimed at the wall addresses the cause;
  every software threshold addresses the symptom.
- The direct versus reverberant null test in `docs/BACKEND_TODO.md` stopped
  being optional. It measures exactly this path.
- Speaker verification already catches it after the fact, which is why `ignore`
  is the most called tool. The cost is that she still wakes, records,
  transcribes and burns a turn before finding out.

Artifacts from the playback are quarantined in `data/wake_playback/` with a
README. They are the measurement, not samples, and must never reach a retrain.

---

## The mixer card guard measured the speakers (Sep 6 2026)

**The mixer card is resolved by name, not hardcoded.** `MIC_MIXER_CARD` was
`"0"` until Sep 6 2026, and card 0 on this Pi is the AB13X speaker adapter, not
the Razer. So the startup check read the speakers' capture input, found 255
against an expected 23, and printed a drift warning at every single service
start, while the mic itself sat correctly at 23. The remedy it printed,
`amixer -c 0 sset Mic 23`, is nine percent on that device's 0 to 255 scale; the
microphone's own scale is 0 to 31, where 23 is 74 percent and 7.00dB. The two
devices do not share a scale, so the number was not read off the wrong card so
much as it did not mean the same thing there.

The damage was not the wrong reading. It was that a guard built to be
impossible to ignore fired on every boot and so became impossible to notice.
`config._resolve_mixer_card` now finds it by `MIC_NAME_HINT`, the same way the
mic is found by name in PyAudio and the speaker in `tts.py`, and returns `None`
on a miss rather than falling back, because a fallback silently measures
different hardware and reports it as the microphone.

---

## Two guards that could not do their jobs (Sep 6 2026)

Both fixed Sep 6 2026, both the same shape: a device reference that silently
changed meaning between boots, sitting inside the thing that was supposed to
catch exactly that.

### `MIC_NAME` carried the ALSA card number

`voiceprint_samples.mic` exists so samples gathered on one capsule are never
folded into a centroid built on another, which is the same class of mistake that
poisoned the April voiceprint. PyAudio's device name ends in the hardware
address, and the card number inside it shifts on reboot, so ten samples from one
physical Razer had recorded themselves as **three different microphones**:
`hw:0,0`, `hw:1,0`, and `hw:3,0`.

`get_voiceprint_samples` filters on an exact string, so a recompute scoped to
"this microphone" would have used a third of the samples and reported nothing
unusual. `config.capsule_name` now strips the address, and **migration 023**
collapsed the rows already written.

It lives in `config.py` rather than `audio.py` because importing `audio` opens
PyAudio and takes the exclusive mic lock, so nothing outside the voice process
can touch it, tests included.

### `enroll.py` did not keep the audio

It wrote every sample to one temp file and overwrote it, saving only embeddings
to `enrollment.npz`. **An embedding is locked to the encoder that made it.**
Resemblyzer is 256 dimensions, ECAPA is 192, and there is no conversion, so
swapping encoders meant re recording all twelve samples, and would again for the
encoder after that.

This is the April lesson one step further on. That failure was that only the
mean was saved, so a bad sample could not be identified afterwards; keeping the
individual embeddings fixed it for analysis and left the same hole for re
embedding. **Audio is the only artifact every future encoder can read.**

Recordings now go to `ENROLLMENT_AUDIO_DIR` (`models/enrollment_audio/`,
gitignored), and the npz gains `audio_files` and `encoder` so the mapping is
explicit in the file rather than implied by sort order.

**They are written at the end, in one step, from the same list the embeddings
came from.** Writing them per sample would leave a previous run's `sample_07`
beside a fresh `sample_00` through `sample_05`, which is silently wrong and
looks completely normal. Correspondence is structural, not remembered.

**This must land before the ECAPA swap**, or enrollment gets recorded twice.

---

## An empty room drove a runaway conversation loop (Aug 10 2026)

whisper.cpp does not reliably emit `[BLANK_AUDIO]` for non speech input. Fed room
noise it hallucinates a short plausible token instead. The runaway conversation
loop on Aug 10 2026 was driven by an empty room transcribing as "over." three
times. `is_noise_transcript` in parsing.py matches a frozenset of known
hallucinations against the whole normalized transcript, never as a substring.

The defence is deliberately narrow. `is_noise_transcript` matches whole
normalized transcripts against a frozenset, never substrings, and it is not
the right tool for overheard speech that happens to transcribe as plausible
English. **Do not widen the frozenset to chase those**; `verify_voice` and the
`ignore` tool are what handle them. The reasoning is in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#whisper).
