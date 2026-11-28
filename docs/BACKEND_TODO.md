# Backend backlog and research

## Approved roadmap

Use the [roadmap and its amendments](ROADMAP.md#order) for active phase order, approvals and dependencies. This document records deferred engineering work and investigation evidence.

## Archive

Finished items (shipped and working, the Sep 14 network incident follow ups, Deepgram, the wake level gate, session level verification and the memory fixes) moved to [archive/BACKEND_TODO_ARCHIVE_2026-11-28.md](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md). Move an item there when it is done, with the date.

## Deferred infrastructure follow ups

The accepted macOS setup left the following follow ups open. Schedule them through the normal approval process.

### Production WebSocket chat repair

The production `/ws` route calls the synchronous chat entry point on the
ASGI event loop. That entry point calls `run_until_complete` on its thread's
chat loop, which raises while the ASGI loop is already running in that thread. The headless development
profile can dispatch its synthetic chat on a worker thread. Production needs
a separate change to decide how chat shares application state and lifecycle
with the other production routes, and to cover full login authorization,
dispatch, disconnects, and concurrent turns. The development phase preserves
the existing production dispatch and tests that boundary.

### Headless development dependency warnings

The owner's macOS Terminal run of the development environment and label server
tests passed all 34 tests and reported three dependency deprecation warnings.
During dependency maintenance, capture the exact warning messages, identify the
affected packages, and verify compatible upgrades with the focused tests.
The warnings are deferred from development environment acceptance.

# Historical research and follow ups

These notes started with the August 2026 speaker verification investigation and grew as the system changed. Sections reflect their dates and may contain completed work, superseded assumptions or proposals that still merit investigation. Check the roadmap and relevant source before using a note to start work. Read only the investigation relevant to the task.

## State of the investigation

### The original problem

Speaker verification produced false rejections on the registered voice.
Accuracy well below 90 percent, worse with distance from the mic. Setup was a
single 256 dimension voiceprint compared by cosine similarity against a
threshold. The threshold was explicitly not the fix, since multiple enrolled
speakers are planned and lowering it trades false rejections for false
acceptances.

### What was actually found

Four separate defects, not one. They had been compounding, which is why the
symptom looked like a single hard verification problem.

**1. The energy VAD never fired.** `SILENCE_THRESHOLD = 200` on mean absolute
amplitude was never crossed by real speech, because capture level was about
-53 dBFS RMS. Zero frames in a real command clip exceeded it. Consequences:
recording length was set by the silence timeout rather than by speech, so
every recording was a near constant 3.0 to 3.4 seconds, and **any command
longer than about three seconds was truncated mid sentence**. Several garbled
transcripts blamed on Whisper were actually truncated audio.

**2. The follow up loop could not exit.** On an unintelligible transcript the
inner loop ran `continue`, which reopened the follow up window instead of
returning to wake word state. Room noise tripped capture, Whisper hallucinated
a short token ("over.", "(beep)", "Enola."), Nova responded, and the window
reopened. Self sustaining. The old guard only caught `[BLANK_AUDIO]` and the
literal word "silence", so hallucinated real words passed straight through.

**3. The enrollment centroid was poisoned.** `enroll.py` recorded a fixed five
seconds with no check on how much was speech, and the prompt list contained
`"Lock in"` (two syllables). A surviving enrollment recording,
`build/enroll_temp.wav` dated Apr 12, kept only 33 percent of its audio after
trimming and scored **0.833 against the centroid it helped build**. Derived
average pairwise similarity across enrollment samples was about 0.63. The
arithmetic fits one bad sample almost exactly: with four good samples at 0.85
and one bad at 0.30, `(6 x 0.85 + 4 x 0.30) / 10 = 0.63`.

**4. Mic gain was far too low.** 19 of 31, about -53 dBFS RMS and -37.8 dBFS
peak, roughly 7 of 16 bits unused. Fixed Aug 10 to mixer value 23 (7.00 dB),
peak -10.3 dBFS on worst case, persisted with `alsactl store`.

### What was ruled out, with evidence

**Quantization noise is not a factor.** This was a leading hypothesis and it
is wrong. Direct simulation, embedding the same signal captured at five
different levels:

```
captured at -53 dBFS -> cosine vs reference 1.0000
captured at -43 dBFS -> cosine vs reference 1.0000
captured at -33 dBFS -> cosine vs reference 1.0000
captured at -23 dBFS -> cosine vs reference 1.0000
captured at -13 dBFS -> cosine vs reference 1.0000
```

The reason is the noise floor stack in the old room:

| | level |
|---|---|
| speech (p90 frames) | -49.1 dBFS |
| room noise floor (p10 frames) | -74.2 dBFS |
| quantization floor | -101.1 dBFS |

Room noise sat 26.9 dB above the quantization floor. Converter SNR was 52 dB,
which is fine. **The binding constraint is acoustic SNR, about 25 dB at desk
distance, and it is acoustic rather than electrical.** Raising mic gain lifts
signal and room noise together and does not improve it.

**Preprocessing mismatch was ruled out.** `preprocess_wav()` is called with no
extra arguments at both enrollment and verification, so both paths run
identical resampling, normalization, and trimming. Verified by reading the
resemblyzer source, not by assumption.

**Utterance duration at verification: RULED OUT ON AUG 10, RE OPENED AUG 12.
The original conclusion was wrong.**

What was written here on Aug 10: the apparent correlation between embedded
duration and similarity (`r = +0.426`) collapsed to `+0.130` once a single
degenerate row was removed, and attempts with three or more seconds averaged
0.579, no better than the 1 to 2 second bucket at 0.582.

That held on the sample available then. It does not hold on 149 scored
attempts:

| embedded duration | n | rejected | median similarity |
|---|---|---|---|
| under 1.0s | 7 | 29% | 0.543 |
| 1.0 to 2.0s | 48 | 6% | 0.616 |
| 2.0 to 3.0s | 30 | 0% | 0.694 |
| 3.0s and over | 64 | 0% | 0.774 |

`r = +0.516`. Median similarity climbs monotonically with duration, and **every
rejection ever logged came from an utterance under two seconds. Zero failures
in 94 attempts above it.**

Duration is not a secondary factor at verification, it is the dominant one. The
earlier reading came from a small sample plus a bucket comparison that averaged
over too few long utterances to separate them.

**Consequence:** the wake word concatenation idea in "Optional, low priority"
below was deprioritised *because of this wrong conclusion*. It is not optional.
It is the fix, and it is already confirmed feasible.

### The SNR curve, which is hardware independent

Measured by adding noise at controlled SNR and embedding. This is a property
of Resemblyzer and this voice, not of any microphone, so **it transfers across
mic and room changes**:

| SNR | embedding cosine |
|---|---|
| 30 dB | 0.998 |
| 25 dB | 0.984 |
| 20 dB | 0.934 |
| 15 dB | 0.844 |
| 10 dB | 0.678 |

Speech level drops roughly 9 to 10 dB per tripling of distance while room noise
stays constant, so far field SNR lands in the 10 to 15 dB band, mapping to
0.68 through 0.84. **That is the distance failure, quantified.**

### The raised voice hypothesis, unresolved

Observation: the further from the mic, the louder the voice had to be, and the
more likely verification failed. Facing away at normal distance caused no
problems. That is a dissociation, since facing away changes mic axis without
requiring raised effort, and it points away from mic axis as the variable.

Raised vocal effort is a genuinely different production mode, not the same
voice louder. F0 rises, spectral tilt flattens, F1 rises with jaw opening,
vowel space expands. Speaker embeddings are trained for channel invariance,
not vocal effort invariance.

**Status: plausible and well grounded in phonetics, but not confirmed on this
system.** Confounded with SNR and reverberation, both of which also change
with distance. Acoustic logging (below) exists specifically to separate them
and has not yet accumulated enough data.

---

## Blocked on hardware

Moving to main campus in a few weeks. **Every acoustic measurement in this
investigation was taken in the old room, against that AC unit, those
dimensions, that reverberation.** A dorm is a different acoustic environment
and the far field problem may look entirely different. No 3D printer yet, so
the enclosure is gated.

Do not spend effort on room specific measurement until both the room and the
capsule are final.

### 1. Rebuild the voiceprint on the final capsule

**Non negotiable, and it invalidates any voiceprint built before the mic
changes.** Enrolling on mic A and verifying on mic B is channel mismatch, one
of the classic large degradations in speaker verification. Different frequency
response, self noise, and directivity.

Sequence, in order:
1. Final mic mounted in final enclosure, in the final room
2. Gain tuned with `scripts/check_gain.py` to peak -12 to -6 dBFS on worst case
   (close range, projected), persisted with `alsactl store`, and
   `EXPECTED_MIC_GAIN` updated in `config.py`
3. VAD mode rechecked (item 3 below)
4. Enrollment run
5. Collection run

### 2. Phase 3 enrollment protocol

`enroll.py` is already rewritten for this and just needs running. What it does
now:

- Twelve samples, eight seconds each, minimum **4.0 seconds of voiced audio
  after trimming**, enforced at record time. A short sample is rejected and re
  recorded rather than averaged in. This is the guard that would have caught
  `"Lock in"`.
- Every prompt is long enough to carry several seconds of speech.
- **Vocal effort crossed with distance**, because they are confounded in real
  use: near normal, near projected, far projected, far normal, plus facing
  away, casual delivery, and careful delivery. Enrolling only close and quiet
  trains on a condition far field usage never reproduces, which is the trap
  the old enrollment fell into.
- Individual embeddings saved to `models/enrollment.npz` alongside the
  centroid, with condition labels and voiced duration per sample, so the
  centroid can be recomputed or analyzed without re recording.
- Pairwise similarity matrix printed, with outliers flagged by two tests: an
  absolute floor of 0.80 and a relative test at 0.08 below the median. The
  relative test matters because Resemblyzer embeddings are non negative and
  even unrelated audio floors around 0.75, so a bad sample looks bad relative
  to the others rather than bad in absolute terms.

Sanity check after running: centroid norm should be near 1.0. The poisoned
April voiceprint had a norm of 0.862.

### 3. Recheck webrtcvad mode at new hardware

Mode 2 was chosen for the old capsule at the old gain. Mode is sensitivity
dependent and a different capsule changes it.

Reference measurements from the old mic:
- At the old low gain: mode 2 read 55 percent of frames as speech, mode 3 read
  3 percent (unusable)
- At the corrected gain: 72 to 79 percent

**Open question never answered:** whether mode 2 endpoints reliably on normal
desk speech rather than only on the loud projected sample used for tuning. Test
this at new hardware before trusting the collection run. If normal desk speech
reads much below about 50 percent of frames as speech, endpointing will run
long and clips will be padded with silence.

### 4. Baseline collection run

Deliberately deferred. Doing it before the hardware change produces data whose
absolute numbers are disposable, since the voiceprint has to be rebuilt anyway,
and it would not have changed the hardware decision.

**Acoustic logging is live, so ordinary daily use accumulates this for free.**
That is better than a staged run, because performed speech is more articulate
than real usage and biases the sample.

If a structured run is still wanted after the hardware settles:

- Four cells: near normal, near projected, far normal, far projected
- Prefer regressing similarity against logged `snr_db` and `spectral_tilt`
  over comparing block labels, since the regression does not depend on
  reproducing "projecting" consistently by feel
- **Prediction to test:** if the SNR hypothesis holds, similarity should track
  `snr_db` along the curve above, roughly 0.98 at 25 dB falling to about 0.68
  at 10 dB. Steeply nonlinear at the low end, not a straight line.
- **Spectral tilt is the discriminator between loudness and distance.** Level
  alone cannot separate them, since near and quiet can arrive at the mic at
  the same level as far and loud. Tilt is a production side signature: a close
  normal voice is loud with steep tilt, a projected voice has flat tilt no
  matter how far it travelled. If tilt correlates with similarity independently
  of SNR, the raised voice hypothesis is confirmed.
- Pause at least 15 seconds between attempts so sessions do not chain into
  follow ups and contaminate `turn_type`.

### 5. Direct versus reverberant null test

Two minutes, and it measures the **room** rather than the mic, so the result
transfers to hardware selection.

```bash
sudo systemctl stop miles-voice
python3 scripts/check_gain.py --seconds 5    # mic pointed at the noise source, no speech
python3 scripts/check_gain.py --seconds 5    # mic null pointed at it, no speech
sudo systemctl start miles-voice
```

Compare the RMS floor. Large delta means the noise is direct path and
directivity buys a lot. Small delta means it is reverberant, and only the
directivity index applies (about 5.7 dB for supercardioid) so room treatment
or distance is needed instead.

For reference: supercardioid nulls sit at roughly ±126 degrees off axis, not
180. A null gives 15 to 20 dB against a direct path point source but only the
directivity index against a reverberant field, so realistic gain from aiming
one is 5 to 10 dB, not 20. Still meaningful on the SNR curve.

### 5b. Encoder comparison, first pass done Aug 12 2026

`scripts/encoder_bench.py`. Builds a centroid from archived clips over three
seconds, scores every clip against it, and reports how far short utterances
fall. Runs on the recording archive with no new audio.

First result, 43 usable clips, 17 of them under two seconds:

| encoder | raw drop | own spread | d |
|---|---|---|---|
| ecapa | 0.190 | 0.186 | **1.02** |
| resemblyzer | 0.111 | 0.061 | 1.80 |

**Read d, never the raw drop, when comparing encoders.** The first version of
this benchmark compared raw drops and concluded Resemblyzer was more robust.
That was wrong for a reason already recorded in this document: Resemblyzer's
embeddings are non negative and unrelated audio floors around 0.75, so its
usable range is roughly a quarter as wide as an encoder whose scores can go
negative. A compressed scale shows a smaller absolute drop for free. Cohen's d
divides by each encoder's own spread and is comparable.

On that measure ECAPA loses about half as much on short utterances.

**This is one half of the picture and the weaker half.** Every archived clip is
Lethanial, so there is no false acceptance measurement at all: an encoder that
scored everything at 0.99 would top this table and be worthless. Do not swap on
the strength of it alone.

What the full comparison needs, after the move:
- Impostor audio. Fifty utterances each from at least two other people, on the
  final mic in the final room. His mother and Azarieyah are the realistic ones.
- Then EER rather than d, which is the number that actually decides it.
- CAM++ and ERes2NetV2 alongside ECAPA. Neither wespeaker nor modelscope is
  installed here; both need adding before those two can be measured.

**Ordering matters and is easy to get wrong.** Swap the encoder *before*
re-enrolling, not after. Enrolling on Resemblyzer and then swapping means
enrolling twice, because embeddings from two encoders are not comparable.

### 6. Mic selection

**Bought and live Sep 30 2026: ReSpeaker XVF3800.** Measurements and what it
changed are in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#the-respeaker-xvf3800-replaced-the-seiren-sep-30-2026).
The reasoning below is why it was chosen.

Requirements: enclosure mountable, good far field pickup across a room, Pi via
USB or I2S.

**An array is worth it over a single capsule, and the binding constraint is
exactly why.** Delay and sum beamforming gives roughly `10 log10(N)` dB against
diffuse noise, about 6 dB for four mics, comparable to a good supercardioid's
5.7 dB directivity index. The difference is that the array beam is **steerable
and adaptive**, so it tracks the talker, while a fixed directional mic only
helps when you are in front of it. The failure mode here is far field with
movement, so adaptive beats fixed at equal directivity.

**The stronger reason is echo.** An array with on chip AEC solves the enclosure
coupling problem in hardware. XMOS XVF series based boards (as used in the
ReSpeaker Mic Array v2.0 and successors) do beamforming, AEC, noise
suppression, and direction of arrival on chip and present as USB Audio Class,
so no kernel driver.

I2S HAT alternatives give raw multichannel audio, which is cheaper and more
flexible, but driver support on current Pi kernels has historically been
painful and the Pi 5 changed the audio subsystem. **Verify current driver
status and availability before buying**, since that moves faster than any
advice recorded here.

**Decision already made:** buy a USB array that works, and do beamforming as a
separate learning project against raw captures. Blocking assistant reliability
on first implementing beamforming is how the interesting project becomes a
chore.

### 7. Enclosure design

Pi, mic, and speaker in one container. This introduces a problem that does not
exist with separate devices: **mechanical coupling**, speaker vibration
travelling through shared structure into the mic capsule. Structure borne
rather than airborne, and therefore much worse.

**The first order risk is not wake word false triggers.** It is that coupling
reintroduces the runaway follow up loop, and webrtcvad cannot stop it this
time. The VAD fix works because an AC unit is not speech and a spectral
classifier rejects it. **Nova's own voice is speech.** webrtcvad will classify
it correctly as speech and open a recording on it.

`flush_input()` already ships as the software half of the defense. It drains
the queued mic buffer after Nova speaks and then discards a margin,
`TTS_FLUSH_MARGIN_MS`, currently 250ms. **Retune that margin once there is an
enclosure to measure**: record Nova speaking, find where her energy actually
ends in the mic signal relative to when `aplay` exits, and set the margin above
it.

Wake word false triggers are a secondary concern. The model is trained on the
phrase "hey nova", so Nova saying "Nova" alone is unlikely to reach the 0.4
threshold. Worth logging, not worth designing around.

**Isolation that works at this scale, in rough order of payoff:**

1. **Compliant mic mount.** Silicone grommets or a proper shock mount. This is
   a mass spring system and its resonance wants to be well below the speech
   band, under 30 to 50 Hz. Highest leverage item, and it must be designed in,
   because retrofitting compliance into a rigid mount means a new bracket.
2. **Separate structural paths.** Speaker baffle and mic mount must not share a
   rigid member. Two sub chassis joined through damping, not one frame.
3. **Speaker in its own sealed sub enclosure**, so cavity pressure does not
   drive the panel the mic is attached to.
4. **Damped panels.** Thin rigid plastic is the worst case, it rings. Mass
   loading or constrained layer damping. Do not print a thin single wall box.
5. **Maximum separation and null orientation** inside the box. Free.
6. **Mic acoustic port**: small hole aligned to the capsule with a sealed
   cavity behind. An unsealed internal cavity becomes a resonator and couples
   speaker output straight to the capsule.
7. **Plan a loopback reference channel** for AEC now even if unused. It is
   architectural, not physical.

### 8. Barge in, and why the enclosure changes it

Barge in requires listening during playback, which requires acoustic echo
cancellation. Structure borne coupling is close to the worst case for AEC:

- The path is highly resonant, so a long impulse response, so a longer filter
  and more compute
- Mechanical rattle and speaker nonlinearity are **nonlinear**, and linear AEC
  cannot cancel nonlinear components at all, which is what caps real world
  echo reduction around 20 to 30 dB regardless of algorithm quality
- The echo path shifts with temperature and mechanical settling

**If barge in matters, mechanical isolation is a prerequisite rather than a
refinement**, and on chip AEC stops being a luxury. Design for it before
building, not after.

### Barge in (closed experiments, summarized Nov 28 2026)

`BARGE_IN` is `False` in `src/config.py`; the code and its settings stay. The full record is in the [archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#barge-in-on-the-array-tried-sep-30-2026-off).

1. Failed, Sep 30 to Oct 1 2026: echo cancellation on the array with the speakers on a second USB device. The canceller removed about 1 dB, because the two USB clocks drift apart by 1 to 3 ms inside one clip and a linear filter never locks. Channel 0 chopped his voice, the wake model scored 0 of 30 over her, and direction of arrival never found him.
2. Worked, Oct 1 2026: speakers on the array's own jack (one device, one clock), canceller 14 to 15 dB, the wake model heard him 27 of 29 over her on channel 1 (`BARGE_IN_CHANNEL`), and `BARGE_IN` ran live Oct 1 to 12.
3. Speech triggers on channel 1 at minus 37 dBFS caught 29 of 29 takes with no false stops on her; an echo check (`BARGE_IN_ECHO_CHECK`) was added after her own first syllables stopped her once, and a wake floor of minus 30 after a false stop on near silence. Cut off words are kept in history as "[You were cut off here.]".
4. Still open: a long idle may leave the canceller worse, live tests with other voices and a TV, "say it slower" (v3 ignores `speed`).
5. Not recorded anywhere in these docs: why `BARGE_IN` is off now. Find out before turning it back on.

### 9. What the array still needs (Sep 30 2026)

Deferred because a few days of ordinary use measure these better than a staged
session, and the enclosure will move all of them again.

- **`WAKE_LEVEL_FLOOR`.** Set on Seiren levels; on the array it cannot fire.
  Retune from `data/wake_hits` and `data/wake_gated` captured on the array.
- **VAD mode** (item 3) and **`TTS_FLUSH_MARGIN_MS`** (item 7), both capsule
  dependent.
- **The two takes the array missed** at 14:27:19 and 14:27:24 on Sep 30, where the
  Seiren scored about 0.5. Listen before trusting the array's miss rate.
- **AEC by reference only.** Tried the same day; see
  [Barge in](#barge-in-closed-experiments-summarized-nov-28-2026).
- **`xvf_host`**: direction of arrival, the LED ring, AEC and beam parameters.
- **48kHz firmware (`_48k2ch`)**: tried Oct 1 2026, and the speakers move to it;
  see [Barge in](#barge-in-closed-experiments-summarized-nov-28-2026).

---

## Latency: where it stands and what is left

Perceived latency (speech end to first audio) went from **8070ms measured** to
roughly **5000ms estimated**, without touching the capture path in any way that
required rebuilding it. Estimated rather than measured because the last two
changes landed after the collection window; rerun `scripts/analyze_timing.py` after a
day of use to confirm.

What actually moved it, all measured rather than predicted:

| Change | Effect |
|---|---|
| `SILENCE_LIMIT` 3.0s to 0.9s | −2060 ms |
| whisper `-ac 1000` | −785 ms |
| Haiku 4.5 over Sonnet 4.5 | −614 ms |
| Prompt caching | 1995ms to 639ms TTFT on a hit |

Measured and **rejected**, so nobody spends a day rediscovering them:

- **whisper-server resident**: 40ms, not the several hundred predicted. The
  148 MB model is in page cache after first load, so reloading is nearly free.
  It also degraded one transcript into a repetition loop.
- **Quantizing the model to q5_0**: 158ms *slower*. Dequantization costs more
  than the memory bandwidth it saves on ARM.
- **Chunked whisper streaming**: would make things worse. Transcription cost is
  per invocation, not per second of audio (1s and 15s clips both cost ~2000ms),
  so chunking means several full invocations.

### Speculative endpointing is no longer worth it

Closed: measured and dropped. The record is in the [archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#speculative-endpointing-is-no-longer-worth-it).

### Remaining ideas, ranked by payoff per unit of work

1. **whisper tiny.en instead of base.en — 724ms measured.** `ggml-tiny.en.bin`
   is already downloaded. One config line. **Do not ship it on the strength of
   that number alone**: the accuracy check behind it is four clips from one
   enrollment recording plus a reference sample, and the cases that matter are
   low SNR far field turns, which already transcribe badly. Use
   `scripts/compare_whisper.py --model whisper.cpp/models/ggml-tiny.en.bin
   --worst-snr` once the recording archive has a few dozen real commands.
2. **A second prompt cache breakpoint on conversation history — 200 to 400ms,
   untested.** Only the system prompt is cached today; the twenty messages of
   history sit after the breakpoint and are re prefilled every turn. Four
   breakpoints are available and one is used. Note that trimming assistant
   turns to thirty words already cut this cost, so measure before adding more.
3. **Ask for a short opening sentence — 200 to 300ms, free.** **The 608ms
   below is stale and disputed; see the `first_sentence_ms` entry after this
   list before acting on this idea.** The sentence assembly stage is the model
   generating the first sentence before anything can be spoken. A shorter opener starts audio sooner and serves the
   brevity goal at the same time. Test it the way the length instruction was
   tested, with repeated sampling.
4. **Run verification concurrently with transcription — 275ms.** They are
   independent and both operate on the same wav. Needs care so the transcript
   still reaches the verification log, which is why it was not done inline.

### `first_sentence_ms` is unmeasured, and carries two numbers (Sep 13 2026)

Deferred out of the documentation reorganization rather than fixed, because
fixing it needs a collection run and this was a docs only session.

**The explanation attached to this stage was wrong.** Both `docs/LATENCY.md` and
idea 3 above describe it as the model generating a first sentence while
`StreamRouter` buffers `LOOKAHEAD_CHARS = 50` before anything can flush. Native
tool use removed the lookahead entirely. `stream_router.py` has no lookahead
buffer, no `ACTION_PREFIX`, and `LOOKAHEAD_CHARS` does not exist anywhere in the
codebase. A short first sentence now flushes as soon as it is complete.

So the 50 character wait, which was the *mechanical* part of this stage and the
part that looked cheap to remove, is already gone. What remains is the model
actually generating the sentence, which is not removable the same way.

**Worse, the stage has two live figures and they disagree:**

| Source | Figure |
|---|---|
| The turn budget, Aug 12 2026 | `first_sentence` **288ms** |
| Idea 3 above | "the **608ms** sentence assembly stage" |

One repo, one stage, two numbers, and this file already says that two latency
figures means nobody trusts either. Neither was taken after the lookahead was
removed, so both describe a pipeline that no longer runs.

**What to do, in order:**

1. Run `python3 scripts/analyze_timing.py` over a day of real use, split by
   `turn_type`, and read `first_sentence_ms` fresh.
2. Replace both figures with the one result. Do not add a third.
3. Only then judge idea 3. If the remaining time is mostly generation, "ask for
   a short opening sentence" is still the right lever. If it is small now, that
   idea has already been half paid by the lookahead removal and should be
   reranked or dropped.

**Do not estimate this one.** It was wrong in mechanism for weeks precisely
because nobody reread it against `stream_router.py`.

## Accountability: she notices, she does not ask (planned Oct 2 2026)

**Why this, of everything.** Asked what would make Nova more useful, his own
data answered: calendar work is about 87 of her tool calls, more than anything
else, and that night he asked her for "a daily LeetCode session, because I need
to be more consistent". Jarvis's usefulness is closing loops, and Nova only
speaks when spoken to. **The design rule is his:** she should know, not ask. "Did
you do X?" every day is a survey he would tune out within a week. Where a fact
can be read from a service, it is read; only what has no source is ever asked,
and then only once.

**His targets:** gym 4 days a week (judged on weekly pace, mentioned only when
behind), LeetCode daily, sleep 8 to 9 hours (trends, "third night under six",
never one bad night). Classes are on Zoom, so travel time and maps are dropped.

**Build in this order, each live and used before the next:**

1. **The evening check in.** Around midnight, the first time he is in the room
   and not mid conversation, once a night. Says only what is off; on a good day
   one line or nothing. Once he answers a point ("skipping the gym today") it is
   dropped, not raised again tomorrow. Sources, all but one connected today:
   Hevy (gym pace), Oura (sleep trend, steps), the calendar (tomorrow's first
   class), and **Google Tasks** for Bible time, the one thing with no other
   record. Tasks is a separate API from Calendar, so `google_auth.py` is run once
   more with its scope; she can also tick a task off when he tells her ("done with
   Bible time"). Details to settle in the design:
   - **"Today" is the day he just lived.** At 12:30am the new date has barely
     started; a check in reading it would say he has not trained "today".
   - **Spoken through `alerts.py`**, the way reminders are: queued, and the voice
     loop announces it at the next safe moment, never over a conversation.
   - **"In the room" from the radar, as "seen in the last 15 minutes".** Measured
     Oct 2 00:57 to 01:40, while he sat at the desk talking to her: the radar
     said present only about 30 percent of the time, with absent stretches up to
     about 10 minutes, but never longer. Instantaneous presence is unreliable as
     mounted; recent presence is enough for this. Fall back to the first wake
     after 11:30pm if it is not.
2. **LeetCode, known without asking.** He solves on NeetCode, which can commit
   each solution to a GitHub repo, so "a commit landed today" is a supported
   signal through GitHub's official API (no token for a public repo, a read only
   token for a private one). Better than LeetCode's own submissions lookup,
   which is unofficial. **Ask him for the repo name.**
3. **"Your class starts in five minutes."** Zoom classes, from the calendar, only
   when the radar has seen him recently. **Ask whether his classes are on his
   calendar**; if not, that comes first.
4. **Class tutor, with an iPad canvas.** Drawing on the iPad (PencilKit, in the
   Swift app repo) and Claude reading and adding to it. Cheap per use (a sketch
   read costs a fraction of a cent); the cost is weeks of app work. He was unsure
   how much he would use it, so **name a weekly use before building**, and test
   whether Claude reads his handwriting well enough by photographing a page of
   notes first.
   **Named Oct 20 2026:** Linear Algebra, then DSA, then Computer Organization,
   later his other CpE classes and his own projects. Modelled on Solvely (photo
   or drawing in, steps out, follow ups on any step), but **hint first by
   default**: the next step, with the full solution one tap away. His call, made
   because he is building this to learn.
   **Teach mode, the main use (same day):** he hands it the material for a test
   or lesson (study guide, slides as PDF, or just the topics), it teaches one
   concept at a time, then gives practice problems he solves on the canvas, and
   it checks his working hint first and brings back what he missed. So the
   canvas is mostly where he answers, and what Claude must read is his working
   mid solve. He keeps no paper notes, so the handwriting test is a screenshot
   of homework worked in Notes or Freeform with the Pencil, the same input the
   canvas will produce.
   **Handwriting test, Oct 20 2026: passes on current models only.** One dense,
   cut off Freeform scratch page of linear algebra, transcribed to LaTeX, graded
   by hand against the page. Haiku 4.5 and Sonnet 4.5 garbled every dense matrix
   and guessed without marking it, so neither can check work. Sonnet 5.5 read
   nearly all of it with two misreads (one, minus 1 as minus 5, changes an
   answer). Opus 5.5 had none found and marked unsure entries `[?]` instead of
   guessing. About 4,650 input tokens and 25 seconds per page with thinking on.
   Haiku 5.5, run after, misread five, so reading his working needs Opus 5.5.
   **Designed Oct 20 2026:** [STUDY.md](STUDY.md) holds the design and the
   order of work.
5. **Assignments, through Notion, next semester.** Canvas would not cooperate;
   Notion's API is solid, but moving his coursework there is the slow part.

**Gmail (he wants it in, use not yet known).** The candidate that fits this
plan: **Canvas emails its notifications**, so assignments and announcements may
already be in his inbox, which could cover item 5 this semester without Notion
or Canvas's API. Second: recruiter replies, beside the Prowl job search tools.
Either way, read only (`gmail.readonly`, another `google_auth.py` run), filtered
by sender on the Pi, and only the subjects of what matched sent to Claude. His
mail is the most private thing she would read, and the rest of it has no reason
to leave the Pi. Look at what Canvas actually sends before deciding.

## Not blocked, and worth doing anytime

### The transcript eval has no audio left (found Oct 9 2026)

`scripts/label_transcripts.py` is the only ruler for STT accuracy, and it can no
longer measure anything. Its manifest (`data/stt_eval/manifest.csv`) names 60
clips, 51 of them hand labelled, all recorded Aug 11 to 13 2026. It stores only
file names and reads the audio out of `ARCHIVE_DIR`, and the archive keeps the
newest `ARCHIVE_MAX_FILES` (600). The clips were present for the Deepgram
measurement on Sep 25 and have since been pruned; on Oct 9 the oldest file in the
archive was from Aug 18, and no copy exists anywhere under `~`. `score` crashes
on the first missing file.

The labels survive and are worthless without the audio. Every WER figure in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) still stands as a record of what was
measured, but none of them can be rerun, and nothing since has been scored:
`DEEPGRAM_KEYTERMS` went live Oct 9 2026 unmeasured, with a "deepgram nova-3
keyterms" row added to the candidate table for when there is audio again.

**The fix is two parts, and both are needed.**

1. `init` copies each sampled clip into `data/stt_eval/clips/` and the manifest
   points there, the way `speaker_eval` and `wake_eval` already keep their own
   audio. Nothing the archive prunes can then touch the eval. About 60 clips is
   roughly 15MB.
2. A new labelling pass, because the old audio is gone. Sample from the current
   archive, which is all on the XVF3800 and so is also the first eval of the mic
   that is actually running; the lost set predates it. Include some clips that
   say family names, so the keyterms have something to be scored on.

Until then, treat any claim of an STT improvement as unmeasured.

### Resuming a reply he cut off (partly built, Oct 30 2026)

**Why:** anything that makes a sound stops her, and a dropped object or a tap
costs him the rest of an answer. Raising the sensitivity floor can only be tuned
on the Pi, against `data/barge_in_stops`. Making a false stop harmless needs no
Pi at all, and a "go on" is wanted anyway.

**Built:** the unsaid part of a cut reply is kept for two minutes in `resume.py`.
The rule for what it holds and what lets go of it is in the
[decision log](ENGINEERING_HISTORY.md#what-a-cut-reply-had-left-to-say-is-kept-oct-30-2026-done).

**Also built:** what to say, and saying it after a stop that came to nothing.
`resume.plan()` restarts the sentence she was cut in (past twenty words, its last
comma whose clause does not open on and, but or so), with a lead in, "Anyway," or
"So,", only from ten seconds; `voice_main` speaks it when the recording after a
stop is noise or too short to be speech. **Reworked Oct 30 2026** after the room
said "Anyway, but most granola is heavy on sugar and oil": the old rules backed
up a few words to a clause and always led in. The constants carry the why.
"A stop that came to nothing resumes by itself" in the decision log still holds.

**Heard on the Pi Oct 30 2026**, with the test paragraph ("Hey Nova, read the
test paragraph"): read whole, a cough resume, a quick go on, a second cut, and
the long sentence rule all passed, and the amended history row matched the
paragraph exactly, said once. Not yet heard in the room: the lead in after a real
gap, and the two minute window; both are covered by the tests.
`scripts/replay_resume.py` runs the rules over every real cut.

**Also built:** "go on" as a local intent, resuming only when it is all he said and
only for him, not a guest. Decision log: "Go on is a local intent". And her
history after a resume: the cut reply's row is extended so the next turn knows she
finished. Decision log: "Her history after a resume".

**Not built, in order:**
1. **Tune the barge in floor from labelled stops.** Needs the Pi: listen to
   `data/barge_in_stops`, label the false ones, replay them through
   `SpeechOnset`. Do it after the resume is heard working on the Pi, so the
   number being tuned is a cost of a short pause and no longer a lost answer.

### Wakes while he is away (planned, Oct 30 2026)

**Not a plain "ignore wakes when I am out."** He wants guests to be able to use
her without him there, so a veto on his phone's location alone would make her
deaf to them. Chosen with him: the phone says whether he is home, the radar says
whether anyone is.

| Him (phone) | Room (radar) | A wake |
|---|---|---|
| Home | any | Normal |
| Away | Someone seen recently | Normal, answered as a guest |
| Away | Nobody seen for about 30 minutes | Ignored |

- **Long window.** The radar is good at saying someone is there and bad at saying
  nobody is; it has gone about 10 minutes without seeing him sit still (see
  [Presence detection](#presence-detection)). A guest sitting quietly must not be
  ignored.
- **Fail open.** A silent radar, a stale geofence, or any missing signal wakes
  her, as `wake_gate.py` already does. A dropped real wake is invisible; a false
  one costs a chime.
- **A manual "guests are over"** that turns the veto off for a set time.
- Guests already get the guest tier, so nothing private is newly exposed.

**Depends on, in order:** a presence fusion function that turns timestamped
events (radar, wake, SSH idle, phone on Tailscale, geofence) into `away`,
`home_idle` or `in_room` with a confidence and expiry each; then a geofence
endpoint in `server.py`. **Test it for a few days on the phone before relying on
it**: how reliably iOS reports a crossing in the background is not established.

> **Update Nov 2 2026:** the geofence endpoint is built, and in the app rather
> than a Shortcuts automation: CLMonitor, a device token for background writes,
> and `POST /presence/phone` logging to `phone_presence`
> ([INFRASTRUCTURE.md](INFRASTRUCTURE.md#the-device-token-and-the-geofence-nov-2-2026)).
> **Next:** let it run a few days, read `at` against `received_at` and look for
> crossings that never came, then build the fusion function and the wake rule. This does not
help with false wakes while he is home, which is the retraining below.

### Retrain the wake model on his false wakes (asked Oct 1 2026)

**Why:** on Oct 1 2026 at 14:57 `hey_nova.onnx` woke at 0.73 during his phone
call, loud enough (minus 7.8 dBFS) to pass the level gate, and she answered
three stretches of the call. The prompt now catches most of that after the fact
(`scripts/replay_ignore.py`: silent on room talk 44 of 63, from 16), but a
false wake still chimes, records 30 seconds and sends it to Deepgram and Claude.
Only the wake model can stop it before that. The model is dated Apr 8 2026,
trained before this apartment, this mic and the array.

**Why not a local filter instead:** measured Sep 25 2026 and rejected, see
[Rejecting a false wake before the Claude call](#rejecting-a-false-wake-before-the-claude-call-tested-do-not-build).
Sentence case and recording length both would have rejected real commands.

**What exists, Oct 1 2026:** `data/wake_hits` 220 clips, 49 labelled (27 his
wake phrase, 22 not); `data/wake_misses` 400, 29 labelled (all not the phrase);
`data/wake_gated` 15, unlabelled. Most hits predate the array (Sep 30), so the
array's own false wake rate is not yet known. `data/wake_eval/` is the test set
and **must never be trained on**.

**Before that, a cheaper candidate, not live (Oct 2 2026).** He labelled the
26 array wakes: 18 his, 8 not. No threshold separates them (0.55 cuts 6 false
wakes and loses 3 of his). A score under 0.6 needing a peak above minus 9 dBFS
cut 6 of 8 and lost none of his, but it was drawn from those same clips, his 4
low scoring wakes were all from his seat, and one real wake of his scored 0.656
at minus 27 dBFS. `scripts/wake_rule.py` scores the rule on every labelled clip,
from the audio already kept, so nothing new runs live. **Label a few more days
of wakes, his other spots included, and run it before the rule acts on
anything.**

**Built Oct 12 2026, at 0.5 and minus 9** (`WAKE_RULE`), **live only from the
Oct 26 17:32 restart**: the commit landed nine minutes after the Oct 12 restart,
and the journal has no "Wake ignored" by the rule until Oct 27, though 17 wakes
in between met it. He labelled 82 more:
106 array wakes since Oct 1, 36 his and 70 not, so two of every three chimes
were false. Swept with `wake_rule.py`: 0.5 at minus 9 cut 20 of 70 and lost
none of his; 0.6 at minus 9 cut 27 and lost one; 0.8 at minus 6 cut 48 and
lost 7. The live function agrees with the script on all 103 clear labels. Still
not shown: his far wakes, which is what it would drop. Next, in order: far
field takes with `record_wake_eval.py`, then the retrain below, training on
features computed on the Pi (`openwakeword.utils.AudioFeatures.embed_clips`,
22 x 96 per 2.5s clip in 0.1s) so the room's audio never leaves it; then
rescore this rule against the new model and keep it only if it still pays.

**In order:**
1. Label the array era hits with `scripts/label_wake.py`, his voice in
   conversation especially. That is the false wake the level gate cannot see,
   since it arrives loud.
2. Train with openWakeWord's pipeline (synthetic positives, his labelled clips,
   room and conversation negatives), on a laptop or GPU, not the Pi.
3. Score the candidate against the current model on `data/wake_eval/` (quiet
   wakes, his wake over her) and on held out labelled hits. Ship only if it
   wakes for him as often and falsely less; barge in reads the same model on
   channel 1, so check that column too.

**Oct 27 to 30 2026: two cheaper fixes measured and set aside, and a bench to
choose an engine with.** He reported her waking unasked. Oct 12 to 26 she woke
76 times, nearly all room talk (his roommates by name); 39 recorded 30 seconds
to an empty transcript, and the rest she mostly answered instead of calling
ignore. Measured offline, nothing live:

- **openWakeWord's custom verifier** (`openwakeword.custom_verifier_model`, a
  logistic regression on the model's features at its peak frame, trained on his
  154 labelled hits). Five fold cross validation, array era, verifier under 0.3:
  cut 57 of 68 false wakes, lost 1 of his 37; kept all 31 woken `wake_eval`
  quiet cues. **Set aside: it is speaker specific, and he wants guests to wake
  her.** Every positive it learns from is his voice.
- **Whisper on the 2.5s wake window**, chime only if it hears the name. 0 of 90
  false wakes contain "nova", but base.en hears his phrase as "in over" or
  "ain't over" and only 21 of 64 as "nova"; accepting "over" too kept 53 of 64
  and let 2 false through, so it would lose one wake in six, and it costs about
  1.1s before the chime. **Parked.** Deepgram with "Nova" as a keyterm is the
  untested version; it sends the archived clips online, so ask first.

`scripts/wake_bench.py` scores any engine on the same audio: `wake_eval` quiet
takes (catches, and false wakes between cues), labelled hits (biased: every
clip already fired the current model), and room audio under `data/room_audio`
as false wakes per hour. Baseline, Oct 30: the live model caught 31 of 33 cued
wakes with or without the gates, 0 false in 4.4 minutes; on hits the gates cut
34 of 90 false and lost 2 of 64 his, both Seiren era. **Next:** a Porcupine
"hey nova" from Picovoice's console through `--porcupine` (speaker independent,
so guests still work); if it does not clearly win, the retrain above. Room
audio is the only set that can show what a new engine falsely wakes on, and it
holds his roommates' voices, so keeping it is his call. It needs capturing from
inside `miles-voice`, which holds the mic.

### A log of her own gaps, for him to review (asked Sep 30 2026)

He wants her to see what she could be better at. She cannot change her code and
should not, but she can record where she fell short, so improvements come from
evidence instead of memory. The same shape as `label_wake.py` for wake misses:
she collects, he reviews.

What to record, each with the turn's message and her reply:
- **She said she can't.** A reply declining for want of a tool or access ("I
  can't check that"), which is a feature request in his own words.
- **A tool failed.** Already in `log_tool_call`; surface the failures, do not
  duplicate them.
- **He corrected her.** A supersede by voice, or "no, that's wrong", right after
  her reply.
- **She guessed where code should decide.** Today's example: `remember` called
  `asked` when he asked nothing. Anything the executor overrides gets logged.

Detection in code, not by asking her to notice herself: a model judging its own
turn is the self report this repo keeps measuring as unreliable. A review script,
`scripts/gaps.py`, lists them newest first and marks each done or ignored.
Nothing here touches the reply path; it runs after the turn like the memory pass.

### Whether a confusion reply should be held for a go on (found Oct 30 2026)

`scripts/replay_resume.py` over every real cut: of 14 replies he cut off, 10
were her confusion replies to a false wake or room talk ("I didn't catch that
clearly. Could you say it again?", "I'm catching fragments…", "I'm listening.").
Four were answers. Cutting one of those off most likely means "not you", but
the remainder is held all the same, and a cough or silence after the stop makes
her come back with "I didn't catch that clearly. Could you say it again?"

Not decided, and not part of the resume wording work: whether she should hold
those at all, and how to tell one apart without matching her wording, which
Claude varies. Start from examples, per [TESTING.md](TESTING.md). Rejecting
false wakes earlier ([Wake word misses](#wake-word-misses-aug-12-2026)) makes it
rarer either way.

### Roll back a deploy that will not start (asked Oct 30 2026)

`scripts/auto_deploy.sh` (not installed, and kept that way; deploys are by hand
with `scripts/deploy.sh`) tests a commit in a worktree before it touches the
repo, so a failing suite never restarts anything. What it cannot catch is a
commit that passes every test and still will not start: a missing package, a
hardware path, an import only the live service reaches. Today that leaves
production down. The commit is written to `data/deploy_failed_commit` so the
timer does not restart it every two minutes, and it stays down until he
pushes a fix or deploys an older commit by hand.

The rollback: when `deploy.sh` reports a service not active, move main back to
the commit in `data/deployed_commit` (the last one that started) with
`git reset --hard`, restart, and check again. The tree is already known clean
by then, so the reset destroys nothing. The failed marker stays, so the next
run does not pull the bad commit straight back; a newer push fast forwards
from the old commit as usual.

Two things to settle before building it:
- **The database does not roll back.** Migrations in `database.py` run forward
  at startup. If the bad commit added a schema version, the old code meets a
  newer schema. Added columns are probably harmless to old code; check what
  `init_db` does with a version it does not know before trusting this.
- **A rollback that also fails must stop.** One attempt, then leave it down and
  say so loudly, or it becomes the restart loop the failed marker exists to
  prevent.

### Rejecting a false wake before the Claude call: tested, do not build

Closed: tested, not built. The record is in the [archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#rejecting-a-false-wake-before-the-claude-call-tested-do-not-build).

### Multi speaker verification architecture

Originally Phase 4. Untouched, still wanted, and independent of hardware.

- **Schema**: a `speakers` table with name, tier, centroid, and individual
  embeddings.
- **Verification**: compute similarity against every enrolled speaker, take
  the argmax, then apply two checks. The top score must clear the threshold,
  **and** the margin between first and second place must exceed a minimum.
  Otherwise the result is ambiguous and should be treated as unknown rather
  than guessed.
- **Why the margin check matters:** without it, argmax always names somebody.
  Two speakers who both score just above threshold, say 0.72 and 0.71, produce
  a confident sounding identification that is essentially a coin flip. The
  failure it prevents is silent misattribution, which is worse than a
  rejection because nothing downstream can tell it happened. Family members
  and anyone sharing vocal characteristics are exactly the case where scores
  cluster. Prefer answering "I am not sure which of you that was" over naming
  the wrong person with full confidence.
- The `tier` column also gives a natural place to require fresh per turn
  verification for a high consequence action even inside an otherwise trusted
  session.

### Optional, low priority

- Consider concatenating wake word audio with command audio for verification
  only, to increase embedded duration. **Confirmed feasible**: the openWakeWord
  ring buffer (`wake_model.preprocessor.raw_data_buffer`, a
  `deque(maxlen=sr*10)`) still holds the "hey nova" utterance when
  `verify_voice()` runs, because nothing calls `predict()` between detection
  and verification, and `wake_model.reset()` clears only the prediction
  smoothing buffer. Lower priority now that duration was ruled out as the
  verification side problem.
- `enroll.py` duplicates speaker device resolution from `tts.py` on purpose,
  since importing `tts` constructs an ElevenLabs client at module scope and
  enrollment should not require a TTS API key. If a third caller ever needs
  it, move it to a shared module rather than duplicating again.

---

## Memory system: correction before automation

### `asked` only when he asks

Sep 30 2026: replaying "I will be going to that event", `remember` came back
`asked` in 6 of 9 even after the description said a stated plan is inferred.
`asked` skips his review queue, so a guess becomes permanent record at once.
Enforce it in code: downgrade `asked` to `inferred` unless his message contains
a request word (remember, note, don't forget, keep in mind, save). A miss costs a
pending memory he approves, never a wrong one stored. Deferred because the tool
does not see his message today; `_run_tools` has to pass it through. Numbers in
[BRAIN.md](BRAIN.md#a-plan-for-one-day-expires-that-day-measured-sep-30-2026).

Three items, in dependency order. The third is blocked on the first two and
must not be started before them.

## Outbound channels: phone calls and iPhone messages

Raised Aug 12 2026. Neither is started. Recorded here so the constraints are
known before either becomes a session's plan.

### Phone calls (inbound, outbound, hold, hangup)

Feasible, and half the infrastructure already exists: a telephony provider
needs a public HTTPS webhook target, and `miles.lethanial.com` behind the
Cloudflare Tunnel already is one.

Shape, with Twilio as the reference provider (Telnyx and SignalWire are
equivalent):

- Inbound call POSTs to a FastAPI endpoint. Answer with TwiML.
  `<Connect><Stream>` opens a bidirectional WebSocket carrying live call audio.
- Outbound is a REST POST to the Calls resource.
- Hangup, hold, and transfer are TwiML verbs plus REST modification of a live
  call. All four asks are covered by the provider API.
- Audio format: Media Streams carries 8kHz mulaw base64. ElevenLabs can emit
  `ulaw_8000` directly, so the outbound leg needs no conversion. The inbound
  leg must be resampled to 16kHz for whisper.cpp.

Four things gate it, in rough order of difficulty:

1. **Latency.** Median perceived is 4298ms (`timing_log`, Aug 12 2026). That is
   comfortable in a room and unacceptable on a call, where silence reads as a
   dropped connection. Phone conversation wants under 1.5 seconds. There is no
   Anthropic realtime speech to speech API, so the whisper to Claude to
   ElevenLabs chain stays as it is. Fast mode does not help: it is Opus 5 and
   Opus 4.8 only, and production is Haiku 4.5.
2. **Barge in is mandatory, not optional.** People interrupt on the phone. See
   "Barge in, and why the enclosure changes it" above; this shares that
   dependency and should not be started before it.
3. **Telephony audio is band limited** to roughly 300 to 3400 Hz. `base.en`
   will do measurably worse than it does on the Seiren V3. Expect the
   `scripts/compare_whisper.py` accuracy question to reopen on a different distribution.
4. **The audio layer assumes one session.** A global `speak_lock` and a single
   aplay process is one speaker and one mic. A call is a second concurrent
   audio session, which is an architecture change rather than a feature.

**Legal:** Florida is a two party consent state for recording. `ARCHIVE_RECORDINGS`
currently captures a person in a room who knows about it. Capturing a caller who
does not is a different thing, and the archive path would need a consent gate
before any call audio reaches `ARCHIVE_DIR`.

### iPhone messages

Harder than calls, and for a different reason: the constraint is Apple's, not
latency.

- **There is no iOS API for reading or sending arbitrary iMessages.** Apple's
  Messages framework covers iMessage app extensions, meaning stickers and mini
  apps rendered inside a conversation. It does not expose the message store and
  does not send on the user's behalf.
- **macOS is the only real path.** The Messages database is a SQLite file at
  `~/Library/Messages/chat.db`, readable with Full Disk Access, and Messages.app
  accepts AppleScript to send. This is the standard approach and it works, but
  it requires a Mac that is powered on and logged in. The Pi cannot do it. This
  is the same dependency as "Mac control" already on the v0.8+ list, so the two
  should be planned as one piece of work.
- **iOS Shortcuts is the thin alternative.** A personal automation can trigger
  on a received message and call a webhook, and the Send Message action can run
  from an automation. Fragile, per trigger, and not a general read path, but it
  needs no Mac.
- **Twilio SMS is not iMessage.** It is a different number and a different
  thread. Fine for notifications, useless for participating in existing
  conversations.

**Privacy, and this one is not a footnote.** `chat.db` holds every conversation
with everyone, including people who never agreed to any of this. Reading it
wholesale to answer "what did my mom text me" ingests messages from people who
are not users of this system. If this gets built, scope the read to specific
threads rather than opening the database, and treat the result the way the
recording archive is treated.

## Ground truth: the only things Nova can know without being told

Raised Aug 12 2026, out of the accountability question. Nova reads a
transcript. Whisper strips affect, so she cannot tell effort from excuse, or
tired from fine. Everything she believes about how Lethanial is doing is
self reported, and self report is exactly what fails on the days it matters.

Three APIs would give her facts she did not have to ask for. They are not
about features, they are about whether the accountability work has anything
underneath it.

**Oura Ring 4.** Already owned, already worn. Sleep and activity, which covers
the two things he actually said he cared about: sleep sitting at 6 to 7 hours
against a stated target of 8 to 9, and whether a session happened. This is the
highest value of the three because it is the only one that reports on a day he
would rather not talk about. Check what the current API exposes before
planning around specific endpoints.

**Hevy.** Workout logging. His written program says "Log every session. Phone
notes app is fine," and a notes app is not queryable. Hevy would turn the
training block into structured sets and reps, which makes "did the muscle up
work happen on Tuesday" answerable rather than a question. Confirm whether API
access needs a paid tier before committing to it.

*Reads built Sep 28 2026* (`hevy_tools.py`): recent workouts, one lift's
history with an estimated one rep max, and routines with when each was last
done. API access does need Hevy Pro. `get_session_plan` (Sep 28 2026) gives
each lift of a routine its last session, a target, and a stall flag (two
sessions without beating the one before, his deload trigger). Sessions are
judged at the working weight, the weight with the most sets, so a heavy single
is neither the target nor a false stall.

*Routines rebuilt Sep 28 2026* for the apartment gym, from this repo, not by
MILES: Upper Strength, Upper Volume, Lower Strength, Lower Volume, skills first
(the programme is in his memories). Rep ranges live on the routine sets in
Hevy (`rep_range`), which the API supports though his routines had none, so
`target` now decides between a rep and a jump: the full planned count of
working sets at the top of the range earns 10 lb on squat, deadlift and their
variants, 5 lb on everything else, and "the next pin" on machines and cables.
The pre rebuild routines are in `data/hevy_backups/`. Three custom exercises
were created: Muscle Up Negative, Straight Bar Dip, Handstand Push Up
Negative. Front squat reads as stalled until a few sessions are logged with
the ramp sets marked as warm ups; the old sessions logged them as working sets.

Hevy's API differs from its own spec in two places: creating a custom
exercise returns the id as plain text, not JSON, and a routine set rejects
`rep_range: null`, so leave the key out. Next: MILES routine edits from the
app chat, for ongoing tweaks, read back before sending.

*Automatic progression, Oct 1 2026* (`src/hevy_progress.py`, `miles-hevy.timer`):
he asked not to have to think about adding weight. Every fifteen minutes,
workouts from a routine finished in the last three days and not yet judged
(`hevy_progressions`, migration 30) are judged with the same `earned_jump` the
session plan speaks from, and an earned lift's working sets are raised in the
routine, backed up first. It only raises. Machines and cables now go up 10 lb
("or the nearest pin"); he picks the closest and the log keeps what he used.
He keeps the routine as is at Hevy's end of workout prompt, since "update"
copies an off day's lighter weights and nothing raises them back. First live
raise: Lower Strength leg extension 150 to 160 after 15, 15.

*Dumbbell cap, Oct 2 2026:* the apartment's dumbbells stop at 50 a hand. He
logs one dumbbell lifts per hand and pairs as the two added together, so a
dumbbell lift logged above 50 is a pair, capped at 100 and stepped by 10 (5 a
hand). At the cap nothing is raised and the target says to slow the lowering
or add a set. Upper Volume incline went from 55 to 50 that day; split squat
is already at the cap. A pair logged at 50 or less in total would be misread
as one hand; none is today.

*The phone overwrites the routines, Oct 2 2026.* Saving a workout in his Hevy
app also saves the app's own copy of the routine, and when that copy is stale
it undoes every change made from here: Upper Volume went back to incline 55,
the ab wheel and the negatives at 7:58:38 PM, 0.2 seconds before the workout
itself saved, and Lower Strength's warm up reps were cleared the same way on
Sep 30. He refreshes routines before a session and keeps the routine at the
end. As a guard, `hevy_progress.run` now also restores any raise found below
the weight it set, but only when the routine last changed within two minutes
of a workout save from it (migration 32 marks those rows `restore`), so a
weight he lowers on purpose in the app stays lowered. Only raises are
guarded: a stale copy can still undo a structural change made from here, so
check the routine after a session that follows one. Hevy replaces a routine whole on update and has no undo, so an edit
must keep the old routine locally first, and an exercise name has to resolve
to one of 451 templates, so a change is read back before it is sent. Since Nov 14
2026 he can also change routines by voice, asked first (`hevy_edit.py`, BRAIN.md). Logging workouts by voice is not
planned: he logs in the app during the session. No bridge line yet;
`bridge_health` names Oura, so lifting needs its own rendered phrases.

**Canvas.** UF runs Canvas, and Canvas LMS has a REST API with courses,
assignments, due dates, submissions, and grades, reached with a personal access
token from account settings. **UF appears to block personal access tokens**
(Lethanial, Aug 12 2026), so treat the REST API as unavailable unless that is
re confirmed. Institutional OAuth developer keys exist but need admin approval,
which is slow and probably not worth chasing.

The workaround covers half of it and needs no API at all. **Canvas exposes an
ICS calendar feed** from the calendar sidebar, which works without a token and
carries assignments and due dates. Subscribe Google Calendar to it and the
deadline half arrives for free, through an integration already planned. One
integration instead of two.

Grades stay self reported, and that is survivable. The grade conversation was
always the one where the follow up question mattered, because a grade never
told you whether he studied anyway.

**Google Calendar.** He said plainly he is going to use it, so it becomes the
schedule source of truth: classes, pledging, shifts, deadlines. Also the thing
that makes a morning briefing possible at all, which is already on the v0.8+
list.

**What none of them give you.** Every one of these reports what happened, not
why. Canvas can say he failed the Physics exam; it cannot say whether he
studied. That distinction is the whole basis of the accountability design, so
the follow up question does not disappear when the APIs land, it just gets
better: instead of "how did the test go", it becomes "I saw the Physics grade,
what happened". She opens with the fact and asks only the part that is
genuinely invisible.

**Rank them by what they need from him.** Oura is passive: it reports whether
he trained and slept with no action on his part. Hevy needs him to log. Canvas
needs an instructor to post. The one that requires his participation is the one
that fails on a bad week, and a bad week is exactly when the signal matters,
because not logging and not going correlate. That argues for Oura as the
primary training signal and Hevy as the detail on top, not the reverse.

**Decide this deliberately.** Connecting these means he loses the ability to
not mention something. That is the point of accountability and it is also a
real change in the relationship. Worth choosing on purpose rather than drifting
into it one integration at a time.

**Do the OAuth once.** Life OS already runs n8n in Docker on this same Pi and
already holds OAuth2 credentials for Google. Google Calendar should ride that
rather than growing a second credential path, and the tunnel is already shared.
Check the Life OS setup before writing any auth code.

**Related and unbuilt: a commitment is not a memory or a reminder.** A memory
is a durable fact. A reminder fires once and is finished. A commitment is
something he said he would do, with a when, that stays open until it resolves
either way. Accountability needs that third thing, and so do birthdays, which
are commitments that recur annually. Probably one migration rather than two.

### Presence detection

Raised alongside the accountability idea (Nova speaking unprompted only when
Lethanial is present and idle). Options on the Pi, cheapest first: BLE or wifi
presence of a known phone MAC (unreliable, randomized MACs); a PIR sensor
(motion, not presence, so it misses someone sitting still); an LD2410 or
similar mmWave module over UART, which detects a stationary person and is the
right answer if this is built. Mic activity alone is not presence.

**Hardware installed Sep 27 2026. Logged since then, not yet acted on.**
`presence.py` reads it inside miles-voice and writes each change to
`presence_log`; nothing Nova does reads that yet. An LD2410C sits on
the desk, wired to the GPIO header: VCC to pin 2 (5V), GND to pin 6, sensor TX
to pin 10 (GPIO15, RXD), sensor RX to pin 8 (GPIO14, TXD), OUT unused. The UART
on those pins is off by default on a Pi 5, so `/boot/firmware/config.txt` now
ends with `dtparam=uart0=on`, which gives `/dev/ttyAMA0`. Do not use
raspi-config's serial option for this; on a Pi 5 it configures the separate
debug port (`ttyAMA10`, where the console lives). `python3-serial` is installed
from apt, because the services run on system Python. Protocol: 256000 baud,
8N1, about ten frames a second, header `f4 f3 f2 f1`, footer `f8 f7 f6 f5`,
little endian. Report type `0x02` is normal, `0x01` is engineering mode with
energy per gate appended. A throwaway decoder lives at `~/ld2410_test.py`,
outside the repo. The port is opened exclusively, so stop miles-voice to use it.

**Its settings live in the module's flash, set over the UART with
`scripts/radar.py`.** `show` reads what the module holds, `record` prints the
energy per gate over a stretch of time, `set` and `range` write. Each needs
miles-voice stopped, since the service holds the port. Leave the HLKRadarTool
app closed: its Detection range slider writes the same max gate setting and put
it back to gate 8 on Sep 27 2026.

| Setting | Value | Why |
|---|---|---|
| Gate 0 moving and still threshold | 80 (default 50) | The empty room read 51 to 64 in gate 0 with the fan unplugged and 100 with it on. After this, 3 to 19 with the fan on |
| Gate 2 still threshold | 10 (default 40) | The bed. Lying still read 8 to 34, median 12, aimed at the door, and was never detected. Aimed at the bed: 18 to 100, median 42. Empty room max 4 |
| Gate 3 still threshold | 10 (default 40) | The far half of the bed. Lying still: min 7, median 18. Empty room max 3 |
| Gate 5 moving threshold | 35 (default 15) | Overnight Sep 27 to 28, small movements at 360 to 375 cm (16 to 30) set off most of the false presence while he slept. Walking in reads 100 |
| Range | gate 5 (default 8), about 3.75 m | Stops short of the door and hallway |
| Unmanned duration | 5 s | Default. How long after the last detection it reports nobody |
| Aim | toward the bed | Not a module setting, and the biggest single change. Aimed at the door, the bed was at the edge of the beam |
| Everything else | Factory default | Gate size stays 0.75 m; 0.2 m would shrink the range under 2 m |

Validated Sep 27 2026 20:08 to 20:15: present on the bed for 2 minutes 24
seconds without a gap, absent within seconds of leaving and for the 3.5 minutes
out, present again at 375 cm on the way in. Before the gate 2 change the bed
flickered every 20 seconds and once went 19 minutes unseen.

**Sleep is not solved.** Against Oura (in bed 01:59 to 10:08, Sep 28 2026) the
module held him 30% of the night: 20% in deep sleep, 28% while not moving, 71%
while tossing. Awake in bed afterwards it held 96%, and a 30 minute nap the
evening before, deep sleep included, held 100%, so the sensor can see him
asleep from where it is and the difference is probably how he lies. At each
overnight loss gate 2 read a still energy of 6 to 10 against a threshold of 10.
Since Sep 28 the service keeps engineering mode on and writes the energy per
gate once a minute to `presence_energy` (`PRESENCE_ENERGY_LOG`), so a night can
be lined up with Oura's sleep stages. Decide from that whether a lower threshold
has room above the empty room's level, or whether the sensor has to move higher
and look down at the bed.

**Sep 29 2026: a threshold cannot fix sleep from the desk.** Against Oura (in
bed 01:02 to 10:08) the module held him 16% of the night. Gate 2's still
energy, minute medians, was 10 to 13 from 01:00 to 04:00, 6 to 8 from 04:00 to
08:00, and 4 from 09:00 to 10:00 while still asleep, which is the empty room's
level. A tenth of sleep minutes read 4 or less, so no threshold separates him
from an empty room. 0.2m gates (`radar.py resolution 0.2`) were tried the same
morning and did not help: lying still read no higher than the empty room at
80 to 100cm, and his movement showed at 140 to 160cm, so the chest is about
1.5m away, not inside the 0 to 1.5m band where this module never reports still
energy (gates 0 and 1 read 0 in every recording). Reverted to 0.75m. Oura is not
live either: its newest heart rate sample was 35 minutes old while he was
awake with the phone. Sleep samples arrive labelled `rest`, after sync.

**Corrected.** An earlier version of this table said the range was 3.75 m
from the app and that a doorway test at 15:00 proved it. The first UART read
showed the range was still gate 8: the app setting had not saved, and the
doorway passed for some other reason, probably low energy at that distance.

**What is left.** During tuning the UART went silent whenever the app was
connected to the module, so any reader must tolerate long silences. The service
should write these settings over the UART at startup, from config.py, so a
module reset or replacement does not silently undo them; the command code for
that is already in `ld2410.py`. Before writing that
service, design what present means over time (5 s is the module's opinion;
Nova likely wants minutes) and how it combines with speaker verification,
because radar sees a person, not Lethanial.

---

## Quick reference

```bash
# Analyze verification data
python3 scripts/analyze_verification.py
python3 scripts/analyze_verification.py --after 2026-09-01 --label "dorm, new mic"

# Check capture level (voice service holds the mic, so stop it first)
sudo systemctl stop miles-voice
python3 scripts/check_gain.py --seconds 5
sudo systemctl start miles-voice
python3 scripts/check_gain.py --file build/command.wav   # no service stop needed

# Enrollment (only after mic, room, and gain are final)
sudo systemctl stop miles-voice
python3 src/enroll.py
sudo systemctl start miles-voice
```

Key config in `config.py`: `VERIFY_THRESHOLD`, `VAD_MODE`, `VAD_PREROLL_MS`,
`VAD_ONSET_FRAMES`, `TTS_FLUSH_MARGIN_MS`, `EXPECTED_MIC_GAIN`,
`MIN_VOICED_SECONDS`, `MAX_FOLLOWUP_TURNS`.

---

## Wake word misses (Aug 12 2026)

### The symptom

"Hey nova" does not fire every time. Transcription hears him fine on the same
turns, so it is not capture level and not the mic.

### What the data says

`wake_log` records only FAILED wakes, at or above `WAKE_LOG_FLOOR = 0.15`.
Successful wakes are not in it; they print their score to the journal instead.
Reading wake_log alone makes it look like a cluster of attempts sitting just
under the threshold, and that reading led to lowering `WAKE_THRESHOLD` from 0.4
to 0.3, which was then reverted. Pull both:

```bash
journalctl -u miles-voice --since "3 hours ago" \
  | grep -oP "Wake word detected! \(\K[0-9.]+" | sort -n
sqlite3 data/miles.db "select score from wake_log order by score"
```

| | n | min | median | max |
|---|---|---|---|---|
| successful | 16 | 0.520 | 0.800 | 0.970 |
| failed | 17 | 0.161 | 0.271 | 0.365 |

The band 0.365 to 0.520 is empty. The model separates cleanly and 0.4 already
sits in that gap, so **the threshold is not the binding constraint** and moving
it only trades one error for the other.

### Why it cannot be diagnosed further today

Two gaps, and the first blocks everything:

**1. `wake_log` stores a score and no audio.** There is no way to tell whether a
0.27 was a real "hey nova" spoken quietly, a fragment of unrelated speech, or
the television. Without that, "the model is weak" and "those were not attempts"
are indistinguishable, and there is nothing to retrain against.

**2. A missed attempt may produce no row at all.** Anything scoring under 0.15
is not logged, so the failures that matter most may be entirely invisible.

### The actual fix, in order

**Step 1, and nothing else is possible before it: capture the audio behind a
near miss.** The wake loop already reads 80ms frames continuously. Keep a rolling
deque of roughly the last two seconds, and on a near miss write it next to the
score. Cheap, and it turns wake_log from a number into a dataset. Drop
`WAKE_LOG_FLOOR` while collecting, or the interesting failures stay invisible.

**Step 2: label what comes back.** If the clips are clearly him saying the phrase
and still scoring 0.27, the model is the problem. If they are the room, the
model is fine and the misses are elsewhere (frame alignment, or attempts that
never reached the mic at usable level).

**Step 3, only if step 2 says so: retrain `hey_nova.onnx`.** The current file is
dated Apr 8 2026, which predates the mic gain tuning of Aug 10 and possibly the
capsule itself. openWakeWord trains custom models from synthetic speech; the
collected real clips from step 1 are what makes an evaluation set possible, so
the retrain can be measured rather than hoped at.

### Do not

Tune `WAKE_THRESHOLD` from `wake_log` alone. It is a table of failures and will
always argue for lowering.

---

## Mute (planned, not built, Aug 13 2026)

Pinned mid design. The plan is settled apart from one open problem; write it
down rather than rediscover it.

### Why it exists

Speaker verification cannot currently separate Lethanial from his sister. Her
scores ran 0.463 to 0.606 against his median 0.683, and no threshold splits
them: 0.50 accepts her 89 percent of the time, 0.65 excludes her entirely and
rejects him 43 percent of the time. Mute is the only deterministic control
available while that is true.

### Settled

- State in `data/mute.state`, a file rather than a DB row, because the CLI, the
  voice loop and the server all read it and a file needs no migration. Survives
  restart by construction.
- **Mute spares the app.** JWT authentication already proves it is him, so
  muting the room must not lock him out of his own phone.
- Blocks the voice channel entirely: no chime, no ack, no Claude call, no
  speech. Timer and reminder alerts are queued through the existing `alerts`
  deferral and spoken on unmute, so muting never silently loses a timer.
- **Default variant, not strict.** The wake word keeps running while muted and
  the only reachable outcome is an unmute: wake silently, transcribe, check
  local intent, do nothing otherwise. Strict (wake word ignored entirely,
  unmute only by CLI) stays available as a config flag.
- Fail **closed** if the state file is unreadable, with a loud journal line.
  This is a security control, so ambiguity must not resolve to "listening".
- Audible confirmation both directions from the phrase bank, so it works
  offline and so a mute that did not take is obvious.

### The asymmetry that drives the design

Muting is fail safe and unmuting is not. Anyone saying "mute" is harmless;
anyone saying "unmute" defeats the feature.

So unmute by voice requires a much higher verification score, around 0.75. Her
measured maximum was 0.606, so 0.75 excludes her outright. It also rejects him
about 61 percent of the time, and that is the correct place to spend a false
rejection: the cost is one repeat on a rare action. Unmute by CLI or app needs
no verification, since shell access or a valid JWT is a stronger credential
than a voice.

### The open problem that stopped it

Turning it on by voice is the hard half, not turning it off.

"Be quiet" or "go to sleep" said to a person in the room could mute Nova by
accident, and because unmuting is deliberately hard, a false mute is expensive.
Two rules were proposed:

1. Mute fires only on an **initial** turn, never on a follow up. An initial turn
   requires the wake word, and the follow up window is exactly where overheard
   speech leaks in (see the `ignore` tool).
2. Only unambiguous phrasing, so "mute yourself" and "stop listening" but never
   "be quiet" or "go to sleep".

He found rule 1 too limiting and pinned the feature there. **That is the thing
to solve before building.** Options not yet explored: a confirmation step, a
short grace window after muting where unmute does not need the high threshold,
or a GPIO button making the voice path unnecessary.

### Worth knowing

The Pi is headless, so "keyboard" means the CLI over SSH. The real version of
what he described is a **GPIO button**, which becomes trivial once the state
file exists: it only has to toggle one file.

---

## Encoder swap: Resemblyzer to ECAPA (measured Aug 13 2026, not yet done)

### The measurement that finally became possible

`encoder_bench.py` always said it could not measure false acceptance because
every archived clip was Lethanial. On Aug 12 2026 his sister used Nova for
about twenty minutes, and 23 of those clips are now labelled by ear in
`data/speaker_eval/manifest.csv`. `--eer` uses them.

    python3 scripts/encoder_bench.py --eer --models resemblyzer ecapa

| | resemblyzer | ecapa |
|---|---|---|
| genuine median | 0.764 | 0.542 |
| impostor median | 0.653 | 0.090 |
| impostor max | 0.843 | 0.179 |
| EER | 22.3% | 4.8% |
| threshold to exclude every impostor | >0.843 | >0.179 |
| ...which rejects him | 82.5% | 5.3% |

**Resemblyzer's impostor max exceeds its genuine median.** Her best clip scored
above half of his. No threshold fixes that; it is a resolution problem, and
same family voices are exactly where a 2019 GE2E model is weakest.

### Migration hazards, in the order they will bite

**1. The scales are different and are NOT interchangeable.** ECAPA cosines run
much lower: his median is 0.542 where Resemblyzer's is 0.764. Carrying
`VERIFY_THRESHOLD = 0.5` across would sit near his median and reject about half
his turns. The threshold must be re-derived, not migrated. Measured landmark:
0.179 excluded all 23 impostors at a 5.3% cost to him. Pick above that with
margin, then re-measure.

**2. The voiceprint has to be rebuilt, not converted.** `models/voiceprint.npy`
is a 256 dimension Resemblyzer embedding; ECAPA is 192. Re-embed the enrollment
audio with the new encoder.

**3. `voiceprint_samples` already has a `model` column.** Use it. Samples
embedded by different encoders must never be averaged into one centroid, which
is the same class of mistake that poisoned the April voiceprint.

**4. `verification_log` history becomes incomparable across the switch.** Every
tuning argument in the earlier Claude guide that quotes a similarity number is about
Resemblyzer and stops applying the moment this lands.

### What this evidence is and is not

23 impostor clips, one person, one evening. Enough to decide between two
encoders, which is the decision in front of us. Not enough for a precise error
rate, and the genuine set is *presumed* his rather than labelled, so the false
acceptance figures are optimistic. Do not quote the EER as a property of the
system.


## Tools: calendar and Oura follow ups (Sep 13 2026)

Left over from landing the permission gate, the calendar and Oura tools, and
next turn confirmation. The design is in `docs/BRAIN.md`; this is what waited,
and why.

### Calendar listing costs one request per selected calendar

Measured Sep 13 2026 by hand, not through `timing_log`: "tomorrow" across nine
selected calendars took 2.4s, on the voice path, plus the `calendarList` call in
front of it. A batch request or a small thread pool would collapse it. Not done
yet because correctness came first, and a speed change here should be measured in
`timing_log` before and after rather than quoted from one run.

### A "yes" that sounds like a goodbye never reaches Claude

`_DISMISS_WORD` in `local_intent.py` matches "that's it", "all set" and "i'm
good". "Yeah, that's it" as the answer to a read back is classified as dismiss,
so the proposal expires and nothing is written. That fails safe. The fix, if it
is ever observed, is to skip the dismiss gate while `pending_action` holds a
proposal from the previous turn. Not done because it couples local intent to
confirmation state for a case nobody has hit yet.

### Google OAuth publishing status is unverified

It cannot be read from the Pi. If the Google Cloud app is still in Testing,
refresh tokens expire seven days after consent. The token was created Sep 13
2026, so every calendar tool would start failing around Sep 20. Check the
console.

### Pending confirmations do not survive a restart

They live in memory, per process. A restart between the question and the answer
loses the proposal, and the "yes" gets "nothing is waiting". Accepted, because
the confirmation window is two minutes and restarts are rare.

### A correction Nova notices on her own skips his review

Found Sep 13 2026. `supersede_memory` always inserts the replacement as `active`,
so when `remember` is called with `certainty: inferred` and `supersedes`, the
correction replaces the old memory immediately instead of waiting in the review
queue the way a new inferred fact does. It predates the memory fix; the fix only
made inferred calls common enough to notice.

Not fixed in the same change because doing it properly needs the queue to hold a
pending replacement: a pending row that records which memory it would retire, and
an approval that performs the supersede. That is a schema migration. Applying an
inferred correction as a plain new pending row would leave both facts active once
approved, and superseding into a pending row would hide the old fact from Nova
until he reviewed it, both worse than today. Corrections he states himself are
"asked" and should apply at once, so the gap is only in corrections she infers.

### A loud room: telling his voice apart from everyone else's

Sep 13 2026, after guests came over. The wake word now interrupts a recording and
the cap is 30, which stops the room holding the microphone. Nothing yet separates
his voice from the room, and that is the harder half:

- **webrtcvad detects speech, not his speech.** Anyone talking keeps a recording
  open until the cap.
- **The wake model was trained before this apartment** and already separates
  poorly here (see INCIDENTS.md, Sep 8 2026). Retraining with room noise and his
  archived wake hits is the software route.
- **Hardware is the strongest route:** a microphone array with beamforming and
  echo cancellation listens toward where he is. It decides the enclosure's
  openings, so choose it before the enclosure.
- **Online speech to text** would transcribe a noisy room better than base.en,
  at a monthly cost and with room audio leaving the Pi. Not faster; see the
  decision log.

## Text channel follow ups (Sep 15 2026)

The backend half is done: a text turn is silent, answers in numerals, and can be
streamed over `/chat/stream`. See
[engineering decision log](ENGINEERING_HISTORY.md#a-typed-message-is-read-not-spoken-sep-15-2026-done).
What is left, and why each was left.

### The app has to send `channel: "text"`

**Nothing above works until it does.** `ChatRequest` defaults to `voice`, on
purpose, because the field was added after the app shipped. The public app repo
posts `{"message": text}` and nothing else, so on that code every typed message
still takes the voice prompt and the speaker. The build on his phone is ahead of
that repo (it posts to `/chat`, while the public code uses the socket), so this
has to be read in the app session rather than guessed at from here.

### The app has to read the stream

`URLSession.bytes` needs no dependency. Append each `delta` to the bubble, clear
it on `reset`, replace it with `done`. The thinking indicator then ends at the
first word instead of at the end of the turn.

### `/ws` has never worked

The handler is `async` and calls `ask_nova`, which calls `asyncio.run` inside the
already running loop. Every message raises. Nothing uses it. Either delete it or
make it `await ask_nova_async`; deleting is the honest option unless the app
wants a socket, since `/chat/stream` now covers streaming. Not touched in the
same change that added the streaming route, because it is a separate decision and
the route it would compete with had not been used yet.

### `WHAT_REACHES_YOU` is false on a text turn

It tells Nova every message is speech recognition output, wrong somewhere in more
than a third of turns. When he types, it is exactly what he wrote. The risk is
the mirror of the incident that created the block: she may treat a typo or a
terse message as a mishearing and hand it back instead of answering it. A text
copy is the same `_for_text` pattern used for the other two blocks. Left out of
the numbers change because it changes how she reads his messages, which deserves
its own before and after.

### Digits in shared history may reach a voice turn

History is shared between channels, so numerals written on text turns now sit in
what voice turns read back. Seeded with one numerals answer, 1 of 4 voice replies
picked up a digit. Whether that matters depends on how `eleven_v3` reads "62",
which has not been tested; the old claim that the synthesizer reads digits badly
predates v3. Measure before building anything: render one reply containing digits
and listen.

### She still does not notice anything on her own

> **Addressed Sep 16 2026** by the background memory pass; see
> [BRAIN.md](BRAIN.md#a-separate-pass-notices-what-he-mentions). The replay this
> entry asked for is what chose the fix. Left here for the history.

Sep 16 2026: in real use since the Sep 13 memory change, `remember` was called
once, for a fact he stated outright, and **no memory has ever been stored as
inferred**, so the approval queue in the app has never had anything in it. The
Sep 13 probes said she would store passing mentions two times in three; his
real conversations say otherwise. Start by replaying a week of his real app and
voice turns through the current prompt and counting how many she would store,
before changing any wording.

### A memory's end date expires it a day early

`get_episodic_memories` hides a temporary memory when `references_date < now`,
comparing ISO text. A bare date like `2026-09-19` sorts before
`2026-09-19T08:00`, so the fact disappears at the start of the day it is about.
The `remember` tool asks for a date, so any date Nova passes has this off by
one. Found Sep 16 2026 while storing Andrew's test date by hand, which was
written as end of day to avoid it. Fix: store a bare date as the end of that day
in `memory_tool`, with a test.

### Deleting a memory from the app breaks its history

`DELETE /memories/{id}` removes the row, while the voice review tool marks rows
`'deleted'`. Deleting a memory that superseded another leaves the older row's
`superseded_by` pointing at nothing. Soft delete would match the rest of the
memory design; not changed yet because the app already calls this route.

### The app side of Sep 16

Backend endpoints exist for: memories (add, edit, history), reminders and
timers (list, cancel), status details, the calendar week with tap edits, and
stage events on `/chat/stream`. None of it shows until the app uses it; the
routes and their rules are in INFRASTRUCTURE.md.

### Prowl name filter and limit

`list_job_applications` and `list_discovered_jobs` cut Prowl's lists on the Pi
(15 and 10 rows), because whole they were 60 KB and 47 KB on Sep 25 2026. A job
past the cut cannot be opened or accepted by voice, and in the discovery inbox
that is most of it: 84 of 120 jobs were unscored and sort last. The fix is on
Prowl's side, since filtering belongs there: `?organization=` (a name match)
and `?limit=` on `/integrations/miles/applications` and `/discovered`. Then
"the Anduril one" is resolved by Prowl rather than by Haiku scanning a cut
list, and the Pi caps become Prowl's `limit`. Not done yet because the caps are
enough to ship and it touches the job tracker repo, not this one.

## Apple Developer Program follow ups (Oct 30 2026)

Push and Health landed the same day the paid program did. What was left out, and
why, in the order it is worth doing.

### Snooze and Done on a reminder notification

A push carries `reminder_id`, so the app can offer actions. Snooze needs a new
route (`POST /reminders/{id}/snooze`: look up the delivered row, write a new one
for now plus ten minutes) and a notification category in the app. **Not built
because** the action runs with the phone locked and the app in the background,
and it needs a login. The access token lasts an hour and the only way back in is
Face ID. See the next entry, which is the same problem.

### Health sync in the background

> **Update Nov 2 2026:** the credential this needed now exists. `/health/sync`
> accepts the device token (`get_device_user`), built for the geofence. What is
> left is the app side: background delivery, and sending with the device token.

HealthKit can wake the app when new data arrives (background delivery). **Not
built because** the wake cannot authenticate: the app's token expires after an
hour and renewing without the app open needs the password, which sits behind Face
ID. The fix is a second credential, a long lived JWT scoped to health writes
only, issued at login and kept in the Keychain readable after first unlock.
`get_current_user` would have to reject any token carrying a scope, or a leaked
health token would work on every route. The same credential would let Snooze work
from a locked phone. Until then Health is as fresh as the last time the app was
opened, and the tool says so.

### Environment is a compile time guess

The app reports `sandbox` for a Debug build and `production` otherwise. A Release
build run straight from Xcode is signed for sandbox but says production, and its
token is refused as `BadDeviceToken`. The robust fix is reading `aps-environment`
out of the embedded provisioning profile. Not done because Debug from Xcode and
Release from TestFlight are the only two ways he builds today.

### Not tried on a phone

Both features were written in a cloud session with no Xcode and no Apple
credentials. The Python is tested against a fake Apple; the Swift has never been
compiled. The first run on the phone is the real test of both.

### Other things the program unlocks, unbuilt

A Lock Screen timer (Live Activity), an Action Button or Siri shortcut that opens
Talk Mode, widgets, and WeatherKit in place of OpenWeatherMap. Ranked by
value: the shortcut is cheap and the most used, the Live Activity needs the push
plumbing that now exists, and WeatherKit only matters if OpenWeatherMap annoys
him.

### Real model scenario harness (queued Nov 28 2026)

A reusable script, kept in `scripts/`, that does what the standing rule in AGENTS.md asks. It exports the code to a temporary folder on the Pi, copies the live database through SQLite's backup, sets `MILES_DB_PATH` and a redirected `HOME`, and runs a scenario list through `brain.ask_nova` on the text channel and on the voice channel. The voice run stubs the speech output so Claude is called and ElevenLabs is not. Each scenario states its setup rows, the phrases, the expected reply pattern and the expected database change. A generic check fails any reply that claims a change the database does not show. The list starts with the Nov 27 2026 night: daily adds, duplicates, cancel then answer, "the second one", "yeah stretching", "both", the claim cases, "did I add eggs?", the timing refusal, the eggs list. The model is not deterministic, so ambiguous phrases run several times and report a count. It prints counts and pass or fail only, never secrets, and deletes the copy and the temporary folder when it ends.
