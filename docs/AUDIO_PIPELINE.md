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

### What the 19.8% is actually made of (Sep 25 2026)

**Read this before quoting it as a transcription accuracy figure**, because half
of it is not one. 19.8% is the unweighted mean of per clip WER over all 58
labelled clips, and 20 of those clips have no true content at all: their audio
was never addressed to Nova. Whisper invents words in 6 of them, each scoring a
flat 1.00.

```
'Safe keeps handy.'            'Yeah, so I think that a couple more is a good one.'
'I just know what we know about Makayana.'          '- New.'
"Come back here! No! Come back here! A bull's hand, hold on!"
'the frame was black. You took it, grimey, and you gave me these.'
```

Six clips at 1.00 out of 58 is **10.3 of the 19.8 points, 52 percent of the
total error mass.** On the 38 clips that are real speech, word weighted WER is
**12.7%**.

So the transcription number and the false wake rate are the same problem
measured twice, and the false wake is upstream. **Waking less improves this
figure without touching the STT.**

What is left after separating them is not noise either:

| | n | WER, word weighted |
|---|---|---|
| initial turns | 24 | 8.4% |
| follow ups | 14 | 21.9% |
| 4 words or fewer | 17 | 18.5% |
| more than 4 words | 21 | 10.9% |

**Noise does not explain it.** SNR computed over the same clips with
`measure_acoustics`, which already spans 3.3 to 30.3 dB in ordinary use:

| SNR | n | WER |
|---|---|---|
| under 14 dB | 29 | 14.4% |
| 14 to 18 dB | 17 | 11.1% |
| 18 to 22 dB | 5 | 17.4% |
| 22 to 26 dB | 5 | 7.1% |
| 26 dB and up | 2 | 0.0% |

Below 18 dB is 13.1%, at or above is 11.1%. Two points, non monotonic. The room
is already the noisy condition and it is not what costs the words; short
utterances with no context for the decoder's language prior are.

**Follow up clips are also where the not addressed speech is**: 27 percent of
follow ups have empty truth against 8 percent of initial commands, because the
follow up window opens into a room that is still talking. Any follow up figure
mixes two populations, which is why the two numbers above differ so much from
the per turn type means the scorer prints.

### The Aug 13 prompt verdict was hiding a split (Sep 25 2026)

`scripts/label_transcripts.py score` now prints WER per `turn_type` as well as
pooled, because the pooled number concealed this for six weeks:

| candidate | pooled | follow up | initial |
|---|---|---|---|
| base.en q8_0 (live) | 19.8% | 37.4% | 9.0% |
| q8_0 + domain prompt, every clip | 20.1% | 35.6% | 10.7% |
| **q8_0 + domain prompt, follow ups only** | **19.1%** | **35.6%** | **9.0%** |
| q8_0 + prior turn text, follow ups only | 20.5% | 39.4% | 9.0% |

(Per turn type cells are unweighted means including the empty truth clips, so
they are higher than the word weighted figures above. Same ruler across a row.)

**The prompt helps follow ups by 1.8 points and hurts initial commands by 1.7.**
Averaged, that is the "0.3 points worse, inside the noise" recorded on Aug 13,
and the conclusion drawn from it, that the prompt buys nothing, was wrong about
the mechanism. It buys something exactly where context is missing and costs
something where it is not.

**Priming with the actual prior turn is worse than no prompt at all**, 39.4%
against 37.4%. The previous turn's words bias the decoder toward reusing them,
which is the opposite of the help that was wanted.

**None of this is shipped, and the follow ups only variant should not be.** Its
entire gain is 2 clips of 22, and one of the two is the prompt leaking its own
vocabulary into the transcript:

```
truth  "Make an order to Raising Cane's ..."
was    'Make an order to raise them in the canes and tell ...'
now    'Make an order to raise them in Gainesville and tell ...'
```

`Gainesville` is a word from the prompt, not from the audio. It scores better by
edit distance and is worse by any reading that matters. The other change moved
"was the beat." to "What's the means?" against a truth of "what's the date",
which is still wrong. No clip moved from wrong to exactly right in any
candidate: `exact` is 36/58 across all four base.en rows.

**What is durable here is the split, not the candidate.** The scorer prints it
now, so the next comparison cannot average a gain and a loss into nothing.

### Beam search is the thing that was never tried (Sep 25 2026)

Every WER figure above, and every one in this repo before this date, was measured
with **greedy decoding**. `-bs 1 -bo 1` was passed at all three call sites,
including production, while whisper.cpp's own default is 5. No comparison had
ever varied it.

| candidate | pooled | follow up | initial | exact |
|---|---|---|---|---|
| base.en q8_0, greedy (live) | 19.8% | 37.4% | 9.0% | 36/58 |
| **base.en q8_0, beam 5** | **17.5%** | **33.3%** | **7.9%** | **38/58** |
| base.en fp16, beam 5 | 19.1% | 37.5% | 7.9% | 37/58 |

**Shipped Sep 25 2026** as `WHISPER_BEAM_SIZE = 5`, read by `_whisper_cmd` so the
speculative run and the full one cannot drift apart. Set it to 1 for greedy.

**It is the only change measured here that improves both populations at once.**
Every prompt variant traded one against the other. It is also the only one to move
`exact` at all: two clips went from wrong to word for word correct, where six
weeks of prompt candidates left it pinned at 36/58.

Cost is 137ms, in the table above, against a 990ms head start and a 4404ms turn.

**One caution on this set.** fp16 at beam 5 lands 1.6 points *worse* than its own
q8_0 quantization, which should not happen and is the clearest evidence in the
table that 58 clips is a small ruler. It is 22 clips deciding every follow up
cell. Treat the direction as solid and the decimals as not.

### small.en and distil-small.en, measured Sep 25 2026

Both downloaded and scored against the same 58 clips, so the rows are
comparable. Latency is p50 over 8 archived clips at three threads, against the
990ms head start.

| candidate | pooled | follow up | initial | exact | p50 | invents words in non speech |
|---|---|---|---|---|---|---|
| base.en q8_0, greedy (live) | 19.8% | 37.4% | 9.0% | 36/58 | 1024ms | 6 of 20 |
| **base.en q8_0, beam 5** | **17.5%** | 33.3% | 7.9% | 38/58 | **1137ms** | **5 of 20** |
| small.en q8_0, greedy | 18.7% | 36.3% | 7.9% | 39/58 | | 6 of 20 |
| small.en q8_0, beam 5 | 17.8% | **31.7%** | 9.3% | 38/58 | 3263ms | 6 of 20 |
| distil-small.en, greedy | 46.9% | 48.0% | 46.2% | 20/58 | | **19 of 20** |
| distil-small.en, beam 5 | 43.6% | 43.4% | 43.7% | 22/58 | | |

**Beam search on base.en beats small.en running greedy**, 17.5% against 18.7%,
for 113ms instead of roughly triple the compute. That is the second time on this
hardware that the decoder mattered more than the weights, after q8_0 turned out
to cost nothing against fp16. **Try the decoder before buying a bigger model.**

**small.en is out on latency, not on accuracy.** At beam 5 it has the best follow
up number of anything measured, 31.7%, and it costs 3263ms against a 990ms head
start. That is roughly 2.3 seconds added to a 4404ms turn, for a population that
is a quarter of turns. If follow up accuracy ever becomes the priority, this is
the ceiling and that is the bill.

### distil-small.en is rejected, and the reason is specific

46.9% pooled looks like a broken model and is not. **On real speech it is
competitive**, word weighted by clip length against base.en q8_0:

| clip length | n | base.en | distil |
|---|---|---|---|
| under 3s | 28 | 20.0% | 18.3% |
| 3 to 6s | 19 | 10.3% | 17.6% |
| 6 to 12s | 4 | 9.4% | 12.5% |

The entire deficit is one behaviour: **it invents words in 19 of 20 clips that
are not speech to Nova**, against 6 of 20 for base.en. Each of those scores a
flat 1.00, which is where 46.9% comes from.

```
truth  '[silence]'   ->  distil  'you'
truth  (room noise)  ->  distil  'Yeah, so I think that it can work.'
truth  (room noise)  ->  distil  'So you keep standing.'
```

**That is the one failure mode this system can least afford**, because 45 percent
of wakes are false and 27 percent of follow up clips are not addressed to her.
A model that fills silence with plausible sentences converts every false wake
into a turn that reaches Claude. Rejected on those grounds and not on WER.

The likely cause is upstream of the weights: whisper.cpp does not implement the
chunked long form strategy the distilled models were trained for. That is a
whisper.cpp limitation rather than a distil-whisper one, so it is worth retesting
after a whisper.cpp upgrade rather than writing distil off permanently.

**Beam search also hallucinates slightly less**, 5 of 20 against 6, which is a
small point in the same direction as everything else about it.

### Deepgram Nova 3, measured Sep 25 2026

Scored as a candidate row on the same 58 clips, via a callable in `CANDIDATES`
rather than a separate harness, so it is the same ruler. `punctuate` and
deliberately **not** `smart_format`: smart format rewrites numbers as digits and
these truths are written in words, so it would score "1 minute" against "one
minute" as an error for a formatting reason rather than a hearing one.

| | pooled | real speech only | follow up | initial | exact | invents words in non speech | p50 |
|---|---|---|---|---|---|---|---|
| base.en q8_0, beam 5 (live) | 17.5% | 11.8% | 33.3% | 7.9% | 38/58 | 5 of 20 | 1137ms |
| **deepgram nova-3** | **7.6%** | **8.8%** | **10.7%** | **5.7%** | **46/58** | **2 of 20** | **520ms** |

**It is better on every axis at once, including latency.** p50 520ms and p90
826ms are both inside the 990ms head start, so a remote transcript is *fully
hidden* where the local one at 1137ms is not. The accurate path is also the fast
one, which is not how this trade normally goes.

Read the two WER columns as different questions. Pooled 7.6% partly rewards
staying quiet on the 20 non speech clips; **8.8% word weighted on the 38 real
speech clips is the accuracy comparison**, against 11.8%. Both favour Deepgram.

**It is also better at silence, 2 of 20 against 5.** That matters more here than
the WER, for the reason distil was rejected: 45 percent of wakes are false, and
every invented sentence in an empty room becomes a turn that reaches Claude.

The errors it removes are exactly the ones that prompted the investigation:

```
truth 'what's the date'                  whisper 'was the beat.'
truth 'Set a timer for'                  whisper 'Seven timer four.'
truth 'Hey Nova'                         whisper 'Head over.'
truth 'Can you set a reminder ...'       whisper 'Can you send a reminder ...'
```

**Not integrated.** Two things have to be decided before it is, and neither is a
measurement. Whisper stays as the offline fallback regardless; `netcheck` already
knows which path is available. See
[BACKEND_TODO.md](BACKEND_TODO.md#deepgram-as-the-online-path).

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

### Settled Sep 25 2026, on 49 labelled hits: the score cannot do it

The empty band is not merely stale, it is **the opposite of what the model
does**. 49 `wake_hits` labelled by ear with `scripts/label_wake.py`, the Sep 13
guests evening excluded so that one unusual night could not set the rate:

| | n | min | p25 | median | max |
|---|---|---|---|---|---|
| real wake | 27 | 0.41 | 0.60 | 0.79 | 0.94 |
| false wake | 22 | 0.40 | 0.45 | 0.52 | 0.84 |

**22 of 49 wakes were false. 45 percent, on ordinary days.** Nearly half the
times she answered, nobody had said the phrase.

The distributions overlap end to end, so no threshold separates them:

| `WAKE_THRESHOLD` | real wakes kept | false wakes admitted |
|---|---|---|
| 0.4 (live) | 27 of 27 | 22 of 22 |
| 0.5 | 25 of 27 | 12 of 22 |
| 0.7 | 16 of 27 | 5 of 22 |
| 0.8 | 13 of 27 | 1 of 22 |

0.8 still admits a false wake and throws away half the real ones. **Stop
proposing threshold changes for this.** The Sep 8 reading was right on 5 events
and is now right on 49.

### Level separates where the score does not, and far field set the line

Peak dBFS over the 2.5 second wake window, real wakes being the 27 labelled
above plus 11 deliberate far field attempts on Sep 25 2026:

| | n | min | p25 | median | max |
|---|---|---|---|---|---|
| real wake | 38 | -40.23 | -35.55 | -30.72 | -9.48 |
| false wake | 22 | -52.36 | -43.19 | -40.47 | 0.00 |

| reject below | real wakes lost | false wakes cut |
|---|---|---|
| -44 dBFS | 0 of 38 | 5 of 22 |
| **-42 dBFS** | **0 of 38** | **7 of 22** |
| -40 dBFS | 2 of 38 | 13 of 22 |
| -38 dBFS | 4 of 38 | 16 of 22 |

This is the mechanism behind the Sep 13 playback result, measured properly: the
neighbours arrive quiet, and quiet is a thing the score does not look at.

**The far field session is why the line sits at -42 and not -40.** On the 27
desk distance wakes alone, -40 looked free: 0 of 27 real lost, 13 of 22 false
cut. Every one of those 27 was spoken from a normal position, so the sample
contained no evidence about the case that a level gate endangers. Eleven
attempts from across the room, the doorway and the next room put two genuine
wakes at **-40.23 dBFS**, just under it. The free gate is 2 dB lower and cuts
half as much.

**Two clips reading exactly -40.23 is itself the finding.** An identical peak on
two different utterances means the peak is the window's noise floor rather than
the voice, so at that distance the measure has stopped describing the speech.
Do not read a peak near -40 as a quiet voice; read it as no voice found.

**RMS was tried and is worse.** Real wakes median -50.77 against false at
-54.60, and rejecting below -54 dBFS loses 12 of 38 real wakes to cut 14 of 22
false. Peak survives because a wake phrase has a loud consonant in it even when
its average is low.

**Built Sep 25 2026**, at -42, in `wake_gate.py`. It runs before the chime, so a
rejected wake costs one peak measurement and nothing else: no recording, no
Whisper, no Claude. Replayed over the labelled clips, the shipped function
rejects 0 of 38 real wakes and 7 of 22 false ones, which is the table above.

Three properties it is not allowed to lose, each pinned in
`tests/test_wake_level_gate.py`:

- **It fails open.** No level, a window not yet full, or `WAKE_LEVEL_GATE` off
  all wake her. A partial window is the first 2.5 seconds after a restart.
- **Rejections keep their audio**, in `wake_gated`, which `label_wake.py` reads
  as a third set. A real "hey nova" landing there is the only way the gate's
  cost is ever visible, and a journal line could not answer "was that one real".
- **The level prints on every detection, not only rejections.** Recording
  failures alone is what made `wake_log` read as a cluster of attempts under the
  line and got the threshold lowered to 0.3 on bad evidence.

It lives outside `audio.py` on purpose: that module takes the microphone lock at
import, so a guard living there is only testable with `miles-voice` stopped, and
this is not the one to leave skipped on an ordinary run.

**It is a stopgap for the neighbours, not a fix.** 13 of the 22 false wakes are
quiet and the array addresses their cause; 9 arrive loud and nearby because they
are his own voice not addressed to Nova, and no acoustic measure separates
those. What a new capsule does to the number is in
[BACKEND_TODO.md](BACKEND_TODO.md#a-level-gate-on-the-wake-window-built-sep-25-2026-at--42).

### The false negative rate is not measured, and cannot be

Every one of the 29 labelled `wake_misses` is `not_wake_phrase`. **There is not
one confirmed missed attempt in the entire labelled set**, and the 11 far field
attempts on Sep 25 were all answered, including from the next room. Distance
alone does not appear to cause misses.

That is not the same as saying there are none. `WAKE_MISS_FLOOR` is 0.15, so an
attempt scoring under it **leaves no trace at all**: no clip, no `wake_log` row,
nothing. If background noise crushes the score toward zero, which is the
condition a miss is most often reported in, the capture is blind to it by
construction. The floor was raised from 0.05 to 0.15 for a good reason, that 283
of 400 clips were empty room, but the cost is that the one failure mode still
unaccounted for is the one hiding below the floor.

Answering it needs a marked attempt whose score is recorded whatever it is, not
a lower floor.

## Wake capture

Keeps the audio behind a near miss, because a score alone cannot distinguish
"he said the phrase and the model missed it" from "that was the television", and
those need opposite fixes. Files are named by score first, so the worst sort to
the top. Captured on a lower floor than the log, since the failures that matter
most may be the ones scoring near zero, which is exactly what `WAKE_LOG_FLOOR`
hides.

This is step one of the plan in `docs/BACKEND_TODO.md`. Retraining
`hey_nova.onnx` is not possible without it: there is nothing to evaluate against.

**Eviction prefers the unlabelled**, since Sep 25 2026. Pruning by age alone
destroyed 14 of the 29 labelled miss clips in eleven days, and a verdict whose
audio is gone cannot train or evaluate anything. `wake_capture.py` evicts
unlabelled clips first, oldest first within each group, and falls back to the
oldest labelled one rather than letting a full buffer stop recording. The loss
and why the fallback matters are in
[INCIDENTS.md](INCIDENTS.md#pruning-by-age-ate-the-labelling-sep-24-2026).

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

> **Changed Sep 16 2026, after re-reading it.** Over 280 turns the pauses are
> longer: p50 330, p75 600, p90 900, p99 1140. The value was set to limit
> discarded work, and that was the wrong cost to minimise. Whisper takes about a
> second, so a speculation started 450ms into the pause finished a median 287ms
> after the recording ended, and that 287ms sat on the path to the Claude call.
> Starting earlier removes it without moving the end of the recording, so there
> is no new risk of cutting him off. Simulated on 283 archived recordings:
> speculations per recording rise from a mean of 1.5 to 2.3 at 210ms, and 60% of
> turns discard at least one, against 38% before.
>
> **Ending the recording early on a finished sentence was tested and rejected
> the same day.** On the same recordings, "the speculative transcript ends in a
> question mark" fired at 45% of real endings and would have cut real words in
> 5 of 285, four of them room chatter. It cannot be built: the transcript that
> rule needs is not ready until after today's cutoff. The same limit is why the
> p99 above leaves no room to lower `SILENCE_LIMIT`.

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
and that is the number that decides whether transcription is free. An STT
finishing inside it contributes nothing to perceived latency at all; one that
does not is only ever partly hidden. That, rather than raw speed, is the bar any
replacement has to clear.

> **The figure here read 750ms until Sep 25 2026, and had been wrong since
> `5173a9c`.** It is derived from two config values, and
> `SPECULATIVE_SILENCE_MS` went 450 to 210 without this paragraph following. At
> `SILENCE_LIMIT = 1.2` it is **990ms today**. Recompute it rather than quoting
> it; a derived number written out as a literal is a number that goes stale
> silently.

When fp16 base.en was live it cost 1237ms at four threads and 1424ms at three
over 12 archived clips, which left ~670ms exposed against the head start of the
day. Measured again Sep 25 2026 on 10 archived clips at three threads, against
the 990ms available now:

| decoder | p50 | exposed |
|---|---|---|
| greedy, `-bs 1 -bo 1` (live) | 1005ms | ~15ms |
| beam 5, `-bs 5 -bo 5` | 1142ms | ~152ms |

**Beam search costs 137ms, not the multiple it looks like it should.** The
encoder runs once either way and base.en's decoder is small, so widening the
beam repeats the cheap half. Predicting this from the beam width would have been
the same mistake as predicting q5_1 was faster than q8_0.

Note the coupling before touching either dial: lowering `SILENCE_LIMIT` shrinks
the head start and makes transcription worse by exactly as much as it makes
endpointing better.

### Verification starts with the speculation (Sep 16 2026)

Verification ran after transcription, and it never needed the transcript,
which it only logs. Each speculation now also embeds the voice, on the same
frames, on one worker thread (`early_verify.py`); `verify_voice` uses that
embedding when the speculation held and computes it as before when it did not.

Measured before building, six archived recordings: whisper alone 1019ms, the
encoder alone 150ms; together, 1055ms and 212ms. The encoder fits inside
whisper's time at a cost of about 36ms to whisper.

**The audio is prepared in one place**, `early_verify.prepare`, from samples
rather than a file path, because the speculative file can be overwritten while
the early run is still reading it. Checked against the old path based version
on four real recordings: identical samples, embedding cosine 1.0. Checked end to
end with the service stopped: the same recording scored the same late and
early (0.612 and 0.688).

The wake word is prepended on a first turn exactly as before, read from the
wake buffer when the speculation starts rather than after transcription; the
buffer does not change in between.

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
