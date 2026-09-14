# Audio pipeline: microphone to transcript

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Content verified against source on the dates named inline. The Whisper and
> STT tables were measured Aug 12 and Aug 13 2026; the wake word sections were
> measured Aug 12, Sep 8 and Sep 13 2026.
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

Everything between the room and the text that reaches Nova: the capsule and
its gain, wake word detection and capture, endpointing, speech recognition and
how a change to it is scored, and speaker verification.

## Microphone and gain

Mic gain is tuned to `EXPECTED_MIC_GAIN = 23` and persisted with `alsactl store`.
It is checked and logged at startup, because a silent revert corrupts collected
data in a way that only shows up days later as inexplicably low scores.

**The mixer card is resolved by name, not hardcoded.**
`config._resolve_mixer_card` finds it by `MIC_NAME_HINT`, the same way the mic
is found by name in PyAudio and the speaker in `tts.py`, and returns `None` on
a miss rather than falling back. It was hardcoded to card 0 until Sep 6 2026
and that guard spent months measuring the wrong device. The full account is in
[INCIDENTS.md](INCIDENTS.md#the-mixer-card-guard-measured-the-speakers-sep-6-2026).

`capsule_name` in `config.py` strips the ALSA hardware address off the PyAudio
device name, so `voiceprint_samples.mic` identifies a physical capsule rather
than whatever card number it enumerated as this boot. See
[INCIDENTS.md](INCIDENTS.md#two-guards-that-could-not-do-their-jobs-sep-6-2026).

## Capture values, and what each is for

The values themselves are declared in [CLAUDE.md](../CLAUDE.md). What each one
is *for* was previously carried as a parenthetical beside it, and is kept here so
the table can stay a plain reference:

- **`CHUNK`** is 80ms frames, which openWakeWord requires.
- **`VAD_MODE`** selects webrtcvad, which replaced the amplitude threshold that
  never fired. See the decision log in [SESSION_START.md](SESSION_START.md).
- **`VAD_PREROLL_MS`** retains frames before onset so soft leading consonants
  survive.
- **`MAX_RECORD`** was raised from 15.0, which truncated 15 percent of follow ups,
  to 60 on Sep 13 2026 once long recordings stopped depending on the twenty
  second window, and down to 30 the same night after a busy room held the mic.
  See [long recordings](#long-recordings).
- **`WHISPER_AUDIO_CTX`** caps the context at 20 seconds. The default of 1500
  padded every clip to 30s and cost a flat ~2000ms regardless of input length.
  Do not lower it without rerunning the validation: 750 and 900 both corrupted
  reference speech.
- **`WHISPER_MODEL`** is quantized locally from base.en with `whisper-quantize`.
  The file is derived, so regenerate it after a whisper.cpp upgrade.
- **`SPECULATIVE_THREADS`** is capped below the core count deliberately; the
  reason is under speculative transcription below.

## Endpointing

**`SILENCE_LIMIT` 0.9 to 1.2.** It was demonstrably cutting him off: live
transcripts read `'set a timer for'` and `'Start give me start five minutes.
I'm'`. This normally costs perceived latency 1:1, but speculative transcription
changes that: 300ms more waiting is also 300ms more overlap with the 1166ms
Whisper stage, so the two very nearly cancel. That does not generalize past the
point where transcription is fully hidden.

Note the coupling documented under speculative transcription below:
`SILENCE_LIMIT` sets the ceiling on how much of transcription can be hidden,
so the two dials cannot be tuned independently.

## Whisper

Measured on 40 archived recordings, Aug 12 2026, with `scripts/compare_whisper.py`:

| Candidate | Median | Saving | Transcripts identical |
|---|---|---|---|
| base.en @ ac 1000 (current) | 1315ms | | |
| **base.en @ ac 500** | **692ms** | **630ms** | **28/40 (70%)** |
| tiny.en @ ac 1000 | 608ms | 707ms | 21/40 (52%) |

**`audio_ctx` dominates the model swap**: nearly the same saving at far higher
fidelity. Neither is safe as a blanket change. tiny.en turned "Can you set a
timer for one minute?" into "Can you say the timer for one minute?" at 24 dB SNR,
a clean recording. ac 500 turned "What is my linear algebra final?" into "One is
my linear algebra final" and made one long turn repeat a sentence and drop the
rest.

Dynamic `audio_ctx` scaled to clip length was then tested (50 frames per second
of audio, margin 100, floor 200): median 498ms, saving 760ms, 27/40 identical.
**Rejected.** It is the fastest and the least safe, breaking at every clip
length rather than only long ones:

```
2.5s  "Who are you?"                     -> "Who are you? Who are you?"
2.9s  "What is my linear algebra final?" -> "One is my linear algebra final."
8.8s  "Thanks, that's all."              -> "Uh, big vessel. No, like, if you..."
6.1s  "Thanks, Noah. Thanks for that."   -> "Thanks, Noah. Thanks for nothing."
```

**Quantization was the exception that paid, Aug 13 2026.** 40 archived clips,
3 threads, against fp16 base.en:

| Model | p50 | Identical to fp16 |
|---|---|---|
| base.en fp16 (was) | 1435ms | |
| **base.en q8_0 (now)** | **1010ms** | **36/40 (90%)** |
| base.en q5_1 | 1959ms | 32/40 (80%) |

Same weights, stored as 8 bit integers with one fp16 scale per block of 32.
Two of the four differences were mildly worse and two were garbage under both.
A real accuracy cost, and a far better trade than tiny.en at 52% or ac 500 at
70%, both rejected.

**q5_1 is smaller and slower, by 524ms.** The Cortex-A76 has native int8 dot
product, so q8_0 weights feed the hardware directly. Nothing on this CPU is
natively 5 bit, so every q5_1 weight is shifted, masked and reassembled before
it can be multiplied, and at base.en's size that unpacking costs more than the
bandwidth it saves. **Smaller is fewer bytes, not fewer cycles.** Whether a
smaller quantization wins depends on which bottleneck the model is against, and
that is a measurement, not a deduction. It was predicted to be faster and was
not.

**`--prompt` fixes "Pi" and is not enabled.** "Pi" and "pie" are perfect
homophones, so no acoustic model can separate them; the decoder's language
prior makes the call. On Aug 13 2026 "what's the temperature of the Pi"
transcribed as "pie" and Nova answered that she had no thermometer. A prompt
naming Raspberry Pi and Nova fixes that clip outright and also turned
`"7 times for 5 minutes."` into `"Set a timer for five minutes."`.

It also turned `"push my code tonight"` into `"push my coat tonight"`, and it
disagreed with base.en on 16 of 40 clips while being **frequently better when it
disagreed**. That is why the agreement score cannot decide it: base.en is not
ground truth and is often the wrong one.

**Settled Aug 13 2026 against 58 hand labelled clips.** The prompt is a wash and
stays off.

| Candidate | WER | Exact |
|---|---|---|
| base.en fp16 | 19.8% | 36/58 |
| **base.en q8_0 (live)** | **19.8%** | **36/58** |
| base.en q8_0 + prompt | 20.1% | 36/58 |
| tiny.en | 23.7% | 32/58 |

**q8_0 costs nothing measurable.** Identical WER and identical exact count to
fp16, so the 422ms is free rather than bought. The 36/40 agreement figure that
justified the switch understated it.

The prompt trades the Pi fix for losses elsewhere and lands 0.3 points worse,
inside the noise on 58 clips. It buys nothing at 64ms and one more moving part.

**19.8% WER is poor in absolute terms.** Good ASR sits under 10%. This is the
strongest argument for Deepgram in the repo and it is an accuracy argument, not
a latency one; after q8_0 there is only ~260ms of latency left to win.

Errors concentrate in **short follow ups**, where there is least context for the
decoder's language prior: "what's the date" to "was the beat", "Hey Nova" to
"Head over", "Set a timer for" to "Seven timer four".

Of 20 clips whose true content is nothing addressed to Nova, **6 produce text
that `is_noise_transcript` does not catch**, e.g. "Safe keeps handy." Mostly
other people talking in the room rather than hallucination. That filter is
deliberately conservative and matches whole transcripts only; the defences for
this case are `verify_voice` and the `ignore` tool, not the noise set. Do not
widen the frozenset to chase these.

## Scoring an STT change

`scripts/label_transcripts.py`. Every comparison before Aug 13 2026 scored
candidates against base.en as though base.en were correct, which is how a
strictly better transcript gets counted as a regression.

```bash
python3 scripts/label_transcripts.py init --n 60   # stratified by clip length
python3 scripts/label_transcripts.py               # label by ear, resumable
python3 scripts/label_transcripts.py --blind       # no candidate shown
python3 scripts/label_transcripts.py score -v      # WER per candidate
```

Labels live in `data/stt_eval/manifest.csv` and are **the** ruler: add a row to
`CANDIDATES` and it is scored against the same set, so two candidates measured
months apart are still comparable.

Labelling is seeded with what the live model heard, because typing sixty
transcripts cold is how a labelling set ends up half finished. That seeding
anchors: a plausible wrong transcript is easier to accept than to notice. Label
a dozen with `--blind` and compare, and if the two disagree that is worth
knowing before trusting any number the tool prints.

The WER function is pinned in `tests/test_stt_eval.py`, including that silence
heard as "over" scores as wrong. A scorer that called that correct would rank
the model responsible for the Aug 10 runaway loop as the best one.

**Do not trade Whisper accuracy for latency here.** Speculative transcription
(below) buys the same order of saving at zero accuracy cost, and these
comparisons score against base.en as if it were ground truth when several
baselines are themselves garbage, so the real regression rate is not knowable
from 40 of ~600 archived recordings.

## Wake word threshold

Every one of these came from one session of real speaking, not from reasoning.

**`WAKE_THRESHOLD` stays 0.4.** It was briefly lowered to 0.3 on the strength of
`wake_log` alone, which was a mistake worth recording because the table invites
it: wake_log holds only FAILURES, so it reads as a cluster of attempts sitting
just under the line. Successful wakes print their score to the journal instead.
Pulling both:

| | n | min | median | max |
|---|---|---|---|---|
| successful | 16 | 0.520 | 0.800 | 0.970 |
| failed (wake_log) | 17 | 0.161 | 0.271 | 0.365 |

**The band from 0.365 to 0.520 is empty.** The model separates cleanly, 0.4
already sits in that gap, and lowering it only moves the line into the noise
cluster. Missed wakes are a model problem, not a threshold problem.

**Never tune this threshold from `wake_log` alone.** Compare both distributions.

Root cause is unfixed. `models/hey_nova.onnx` is dated Apr 8 2026, predating the
mic gain tuning of Aug 10, and `wake_log` stores a score with no audio, so there
is nothing to retrain against and no way to confirm whether a near miss was even
a real attempt. Capturing the audio behind a near miss is the prerequisite for
any real fix. See `docs/BACKEND_TODO.md`.

> **Correction, Sep 13 2026.** The clean separation described above was
> measured in the **old room** and does not hold in the current apartment.
> Both distributions now overlap, and the cause is neighbour speech arriving
> at varying level rather than a bad threshold. Read
> [INCIDENTS.md](INCIDENTS.md#no-threshold-separates-true-from-false-wakes-any-more-sep-8-2026)
> before acting on the empty band claim.

## Wake capture

Keeps the audio behind a near miss, because a score alone cannot distinguish
"he said the phrase and the model missed it" from "that was the television", and
those need opposite fixes. Files are named by score first, so the worst sort to
the top. Captured on a lower floor than the log, since the failures that matter
most may be the ones scoring near zero, which is exactly what `WAKE_LOG_FLOOR`
hides.

This is step one of the plan in `docs/BACKEND_TODO.md`. Retraining
`hey_nova.onnx` is not possible without it: there is nothing to evaluate against.

> **Correction, Sep 13 2026.** `WAKE_MISS_FLOOR` was raised from 0.05 to 0.15
> and `WAKE_LOG_FLOOR` is also 0.15, so the capture floor is no longer *below*
> the log floor as the paragraph above states. They are equal. The reason for
> the raise is the next section.

### Wake hits are captured now, because misses were the wrong dataset

`wake_misses` only records scores **below** the threshold, so it answers "why
did she not hear me" and cannot answer "why did she answer when I said
nothing". `archive_recording` keeps the command spoken after the wake, which is
a different sound and cannot say whether the phrase was ever uttered.

`voice_main` already kept `_wake_window`, a rolling 2.5s of frames, purely for
near misses. Nothing read it on the firing path. `save_wake_hit` now does, into
`data/wake_hits`, capped at 300.

### `WAKE_MISS_FLOOR` 0.05 to 0.15

At the 400 file cap: **283 clips under 0.10, 71 percent of the buffer**, against
19 between 0.20 and 0.30 and 10 above. Pruning keeps the newest by mtime, so
empty room was evicting the band that sits near the threshold. A capture floor
low enough to record everything keeps the least interesting thing. The near zero
question is already answerable from the 283 on disk.

## Labelling wake audio

Labels either set by ear, worst first, resumable. Keyed on **filename, not
position**: both directories are ring buffers, and a verdict recorded against an
index points at a different clip after any eviction.

**It stops `miles-voice` for the duration and restarts it in a `finally`.**
Without that the tool corrupts the dataset it is labelling: the first real
session created three wake_hits and three wake_misses from its own playback,
which then sit in the capture directories indistinguishable from room audio. The
second reason is worse than the first. A clip that transcribes as a command gets
executed, and nothing in these recordings was chosen to be safe to say aloud
near an assistant that sets timers and writes memories.

**Clips are boosted to -3 dBFS for playback**, capped at 36 dB. Raw, they are 30
to 38 dB below close speech and inaudible through desk speakers, and a clip you
cannot hear cannot be labelled. The boost is written to a temp file, never back
to the capture directory, and the real peak is printed beside the gain applied.

```bash
python3 scripts/label_wake.py --status
python3 scripts/label_wake.py hits              # every false wake, loudest first
python3 scripts/label_wake.py misses --min 0.2  # 29 clips, not 400
```

## Long recordings

Sep 13 2026. Scheduling seven tutoring lessons by voice hit the 18 second cap
twice in one conversation, and the third turn ended mid thought. The cap sat under
Whisper's twenty second window because audio past it was never transcribed.

Real use says long recordings are rare: of 83 recordings since Sep 5, half were
under 2.9s, nine in ten under 8.2s, and 3 reached 18s. So the fix had to leave
normal commands exactly as fast.

**Two fixed windows, chosen by the clip's own length.** Up to 15 seconds, the fast
window. Past it, Whisper's full thirty seconds. Past 28 seconds, the recording is
cut at the quietest moment near each limit and transcribed in pieces. Not a window
scaled to each clip, which is the approach rejected above.

**Why 15 and not 19.** The fast window was only ever validated on clips of 3, 6, 10
and 14 seconds. Run on his five longest archived recordings, all 18 seconds:

| Clip | Fast | Full | Transcript |
|---|---|---|---|
| Sep 13, the scheduling request | 1299ms | 1936ms | same words; "S.A.P" against "SA-P" |
| Aug 16 | 945ms | 1438ms | identical |
| Aug 16, background talk | 1213ms | 1809ms | fast added a clause, full added a word |
| Aug 12, background talk | 1300ms | 1994ms | **fast dropped trailing words**: "I'll put it" against "I'll put it on the floor" |
| Aug 16 | 956ms | 1546ms | identical |

The full window cost 490 to 690ms on these clips. None has ground truth, and two
are not commands, so this does not prove the fast window wrong at 18 seconds. It
does show the twenty second limit is not a safe edge, so the line sits just above
what was validated. Speculative transcription is skipped for a recording that
will be cut into pieces, because one run cannot stand in for several.

**Not fixed by this:** `SILENCE_LIMIT` still ends a recording on a pause, so a long
explanation with a long thinking pause ends at the pause.

### The cap came down to 30, and the wake word interrupts a recording

The same night, guests came over and 60 seconds was the wrong cap. webrtcvad
counts anyone talking as speech, so a recording in a busy room never reaches its
silence and runs to the cap. A false wake recorded 53 seconds of their
conversation; a follow up ran the full 60. The evidence is in
[INCIDENTS.md](INCIDENTS.md#a-busy-room-held-the-microphone-sep-13-2026).

Two changes, and neither separates his voice from the room, which is still open:

- **`MAX_RECORD` 30.** His longest real request was 42 words in 18 seconds, and
  30 seconds is about 75 words.
- **The wake word is listened for during a recording.** A second copy of the
  wake model hears it, because the main model's buffer must stay frozen on "hey
  nova" for `wake_word_audio` to hand to verification. On a detection the chime
  plays and the recording starts over from that moment. `wake_listener.py` turns
  the 30ms capture frames into the 80ms chunks the model needs.

Transcripts are checked for the wake phrase anywhere, not only at the start:
"No way. Hey Nova. Hey Nova." now means he said only the wake word, and she
records again instead of sending it to Claude.

**Rejected: ending the recording as soon as the speculative transcript is ready.**
It is ready at 450ms of silence in 86 of 98 recordings, which looked like half a
second free. But across 241 turns he spoke through a pause longer than 450ms in
85 and longer than 750ms in 44. Ending early would have cut him off mid thought
in about one turn in five.

## Speculative transcription

Endpointing and transcribing used to run in sequence: 880ms waiting out silence
with the CPU idle, then 1177ms of Whisper. The audio already exists when speech
stops, so Whisper now starts during the wait. Speaking again cancels it and the
next pause tries again; anything stale or failed falls back to a full run.

**Free in accuracy terms.** Same model, same settings, same samples, started
earlier. Verified Aug 12 2026 by cutting the trailing silence off 25 archived
recordings: 19/25 identical, and all six differences benign (three are
`[ Silence ]` against `[BLANK_AUDIO]`, both of which `is_noise_transcript`
already catches; the rest are comma placement or clips that were garbage in
both).

`SPECULATIVE_SILENCE_MS` comes from `max_pause_ms` over 66 turns: p50 270, p75
450, p90 630, p99 840. At 450 about a quarter of turns speculate on a pause he
speaks through and are discarded; the rest overlap 450ms of the 900ms wait. The
overlap can never exceed `SILENCE_LIMIT` minus this value. **Re-read that
distribution before changing it.**

That same distribution says **`SILENCE_LIMIT = 0.9` has no headroom left**: p99
of pauses spoken through is 840ms against a 900ms limit. Lowering it cuts him
off mid thought.

Whisper is capped at 3 threads for the speculative run. It defaults to all four,
and starving the PyAudio read loop drops frames, which matters precisely when
speech resumes and the recording still has to be good.

`verify_voice` uses `transcript` only for logging, never for scoring, so
verification could also run concurrently with transcription. Worth ~199ms of a
4404ms budget, against torch and Whisper contending for four cores. Not done.

**It only ever covered initial turns.** Written into `record_command` and never
into `listen_for_followup`, whose phase 2 is a deliberate copy of the same loop
and stopped mirroring it the day speculation landed. Follow ups are the majority
of turns and paid full price for the entire life of the feature. Fixed Aug 13
2026 by routing both loops through `maybe_speculate` and `cancel_speculation`
instead of inlining the block in each; `tests/test_speculation.py` pins that
structurally, because the copy is what drifts.

Measured Aug 12 2026, same window and same config, split by `turn_type`:

| | n | transcribe p50 |
|---|---|---|
| initial (speculating) | 20 | 671ms |
| follow up (not) | 24 | 1187ms |

The aggregate hid it completely. Pooled, those read as one unremarkable 1083ms
median rather than as a bimodal distribution with follow ups pinned to the slow
mode. **Split timing_log by `turn_type` before trusting any stage median.**

**The head start is capped at `SILENCE_LIMIT` minus `SPECULATIVE_SILENCE_MS`**,
750ms today, and that is the number that decides whether transcription is free.
base.en costs 1237ms at four threads and 1424ms at three, measured over 12
archived clips, so even a perfect hit leaves ~670ms exposed. An STT that
finishes inside 750ms contributes nothing to perceived latency at all; one that
does not is only ever partly hidden. That, rather than raw speed, is the bar any
replacement has to clear.

Note the coupling before touching either dial: lowering `SILENCE_LIMIT` shrinks
the head start and makes transcription worse by exactly as much as it makes
endpointing better.

## Speaker verification and enrollment

Resemblyzer, 256 dimensional cosine similarity against the stored voiceprint.
Two structural fixes landed Sep 6 2026, both of which had quietly broken the
ability to analyze or rebuild a voiceprint at all: the per sample mic label
carried a shifting ALSA card number, and enrollment discarded its own audio.
Both are written up in
[INCIDENTS.md](INCIDENTS.md#two-guards-that-could-not-do-their-jobs-sep-6-2026),
and the encoder swap they unblock is in [BACKEND_TODO.md](BACKEND_TODO.md).

## Quirks

ALSA and JACK stderr suppressed via ctypes error handlers and a
`silence_stderr()` context manager in audio.py. Supercardioid mic means short
phrases at desk distance score lower on verification. OpenWeatherMap geocoder
works best with city names only. whisper.cpp compiled with NEON ARM optimizations.

The `[BLANK_AUDIO]` hallucination behaviour and the runaway loop it caused on
Aug 10 2026 are in
[INCIDENTS.md](INCIDENTS.md#an-empty-room-drove-a-runaway-conversation-loop-aug-10-2026).
