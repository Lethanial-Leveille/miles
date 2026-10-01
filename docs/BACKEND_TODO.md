# M.I.L.E.S. Backend TODO

Deferred work from the speaker verification investigation, Aug 10 2026.

Written to be picked up cold. If you are a fresh session reading this, start
with "State of the investigation" below, then check "Blocked on hardware"
before proposing anything, because most of what is left is gated on hardware
that did not exist when this was written.

---

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

## Shipped and working

All live in production as of Aug 10 2026. All mic independent unless noted.

- `verification_log` table with per attempt logging: similarity, accept
  decision, threshold in effect, transcript, recorded duration, embedded
  duration, wake confidence, turn type, outcome, and acoustic measures.
  Migrations 2 through 4 in `database.py`.
- `is_noise_transcript()` in `parsing.py`. Catches bracketed and parenthesized
  sound annotations plus an exact match list of known Whisper hallucinations.
  Exact match on the whole normalized transcript, never substring, so "over."
  is noise but "Read that over for me" is speech. Twelve tests.
- Follow up loop exits on noise instead of reopening the window.
- `MAX_FOLLOWUP_TURNS = 6` hard cap regardless of what the VAD decides.
- webrtcvad mode 2 replaces the energy threshold for both onset and endpoint.
- 300ms pre roll ring buffer plus two frame onset debounce, so soft leading
  consonants are no longer clipped. This fixed transcripts like "year do I
  graduate?" losing their first word.
- `verify_voice()` returns `VERIFIED` / `REJECTED` / `NO_AUDIO`. No voiced
  audio no longer produces a meaningless similarity score or accuses the user
  of being an intruder.
- `measure_acoustics()` logs RMS dBFS, SNR dB, and spectral tilt per attempt.
- `flush_input()` drains the mic buffer after Nova speaks. See "Enclosure"
  below for why this matters more once mic and speaker share a housing.
- Mic gain checked and logged at every service start, with a loud warning and
  the restore command if it drifts.
- `scripts/analyze_verification.py`, with `--after` / `--before` / `--label` /
  `--include-degenerate`.
- `scripts/check_gain.py`, records or analyzes a clip and reports level against
  targets. **Mic dependent**: `MIC_MIXER_CARD` and `MIC_MIXER_CONTROL` in
  `config.py` will need updating for a new capsule.
- `enroll.py` rewritten (see below). **Not yet run.**

---

## Done since the Sep 14 2026 network incident

### Tailscale, installed Sep 15 2026

Was deferred here for part of a day because it had been reported as applied
when it had only been installed on the MacBook. Now real on the Pi: tailscale
1.102.4, `tailscaled` enabled and active, joined as `miles` at
`100.99.248.127` on the tailnet.

Full reasoning is in the decision log in
[SESSION_START.md](SESSION_START.md). Kept here only as a pointer, because the
same item was listed as deferred and a future session searching this file for
Tailscale should not find a stale "not done".

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

### Barge in on the array (tried Sep 30 2026, off)

A day of testing "hey nova" said over her, ending with `BARGE_IN` and
`AEC_REFERENCE` both off. What was learned, so the next attempt starts here.

**The array's two channels behave very differently while she talks.** Read with
`xvf_host.py` (Seeed's `python_control`, pyusb, no root with
`udev/99-respeaker-xvf3800.rules`): channel 0 is the processed output (category
8, AGC, noise and echo suppression), channel 1 the recognition beam (category 7
source 3).

| | Her voice | His "hey nova" | Wake score |
|---|---|---|---|
| Ch0, no reference | full level | buried | none |
| Ch0, reference, `PP_DTSENSITIVE` 0 | gone entirely | "Hello?" (chopped) | none |
| Ch0, reference, `PP_DTSENSITIVE` 15 | leaks at first, then gone | "I know that" | none |
| Ch1, no reference | about minus 20 to minus 25 dBFS | "Take over" | 0.00 |
| Ch1, reference, DT 15, arecord test | present | "Hey, Nova" twice | **0.88, 0.98** |
| Ch1, production path, speakers moved | about minus 28 dBFS | "Hey, Nova", clear | 0.00 |

The one success was an `arecord` capture where he likely spoke louder; it never
reproduced through `play()`. The last row is the deciding test: a fixed clip
through the real `_play_with_barge_in`, his wake transcribed clearly by
Deepgram, scored 0.00 by `hey_nova.onnx`.

**The reference over a second USB device is sound on timing.** Her echo reaches
the raw mic 18.5ms after the chip's view of the reference (routed to USB with
`AUDIO_MGR_OP_R 4 0`), stable across the clip, inside the canceller's 3072
sample (192ms) filter. Speakers stay on the AB13X at full quality.

**The three second lag was never a lag (solved Sep 30 2026).** ALSA's own
status shows the array triggering 4.7 to 9.7ms *before* the speaker and playing
the 16.88s clip in real time. The drop rule was wrong: it counted what was queued
for the reference thread, and a synthesis already fetched is queued all at once
(5.85s in the first 4ms, written through by 0.32s). So every such sentence was
dropped before any audio played, and its stderr went to `/dev/null`, so it looked
like a slow array. `_Reference` now drops only when the array has taken nothing
for a second (`_STALL_S`). Three runs through the real `play()`: never dropped,
all 16.88s delivered.

**Fixed along the way, kept and off:** `play()` now stops mid sentence (it used
to check once, after writing everything, and missed every wake in a sentence's
last second); `_Reference` can never block the speaker, and waits for the
array to be free before the next sentence opens it; the watcher logs its peak
and level per sentence and keeps what it heard in `build/barge_in_*.wav`.

**Any speech on channel 0 does not work (measured Sep 30 2026, evening).** With
the reference running for the whole reply, `PP_DTSENSITIVE` 0, nobody talking,
the same 16.88s clip:

| Reference | runs | channel 0 median while she plays | webrtcvad mode 2 calls her speech |
|---|---|---|---|
| none | 2 | minus 32 dBFS | 80 percent |
| plain `aplay` of the file | 5 | minus 39 to minus 65 | 46 to 74 percent |
| `play()`, the production path | 8 | minus 24 to minus 61 | 58 to 94 percent |

Cancellation comes and goes second by second (minus 67 one second, minus 25 the
next), and her loudest frames reach minus 3 dBFS, as loud as his speech over
her (minus 3 to minus 7). Six talk over takes agreed. The morning's minus 63 to
minus 72 described her quiet stretches, not a reply. A watcher firing on
sustained speech would stop her within a second of nearly every reply, so it was
not built. Why the file beat `play()` in every pairing is unexplained: routing
the canceller's own inputs out (`AUDIO_MGR_OP_L 3 0`, `AUDIO_MGR_OP_R 12 0`, not
saved, restored after) showed her echo 14 to 20ms after the reference for both,
steady, inside the filter, at identical levels. At the best lag, reference and
echo correlate only 0.3 to 0.5, which points at speaker distortion or coupling a
linear canceller cannot remove, and the chip's AGC then runs at about 44 times
on whatever is left. A small book under the speaker changed nothing measurable.

**Direction does not find him while she talks.** `DOA_VALUE` puts her at 280 to
290 degrees on about 90 percent of reads, with a reflection near 150. His seat
alone read 98 degrees, steady. Talking over her, the reading left her direction
on 16 percent of reads from his seat and 3 percent from another spot, against 4
to 14 percent with nobody talking: it never heard him.

**The wake model over her, scored on a recorded set (Oct 1 2026).**
`scripts/record_wake_eval.py` records him saying "hey nova" at cues over her
replies (reference on), her alone, and quiet wakes, both channels, into
`data/wake_eval/` (evaluation only, never trained on). Each take is labelled by
his confirmation, not the cue. `hey_nova.onnx`, fed 80ms frames like the watcher:

| Session | Speaker | Over her, heard (ch0, ch1) | Quiet room | Her alone, loud moments ch0 | Him over her vs her, ch0 |
|---|---|---|---|---|---|
| 003125 | 192 | 0 of 30, 0 of 30 | 9 of 9 | minus 10.6 dBFS | minus 0.9 dB |
| 004958 | 108, mic moved | 2 of 30, 4 of 30 | 8 of 9 | minus 49.9 dBFS | plus 14.1 dB |
| 010635 | 120 | 0 of 18, 1 of 18 | 6 of 6 | minus 16.2 dBFS | plus 3.8 dB |

No false fires in any session. The model hears him in a quiet room and is deaf
over her, on both channels, from every spot.

**The 108 session was the canceller, not the volume.** Six silent runs
alternating 108 and 120 the same night gave her loud moments at minus 16 to
minus 17 dBFS and minus 13 to minus 14: the 3dB the volume step should give,
steady. During session 004958 the canceller was about 30dB better than in any
run before or after, with the same mic, code and volume, and nothing logged
says why. **Every single run comparison is suspect until that swing is
explained**, including the volume conclusion first drawn from it. The chip's AGC
gain drifts too (44 in the afternoon, 54 to 62 across the six runs), which is a
few dB, not thirty.

**Even in the good session he was missed.** Plus 9 to 14dB over her, and still
2 to 4 of 30. On channel 0 his phrase over her carried about 10dB less of its
energy between 1.5 and 3 kHz than in the quiet room, consistent with double talk
suppression by frequency (`PP_FMIN_SPEINDEX`); channel 1 showed no such loss and
missed as often, so that is a lead, not the explanation.

**Whisper is no detector of "hey nova".** On the quiet takes, which the wake
model scores 0.9, `base.en` writes "Ain't over", "Game over". A keyword search of
its transcripts as evidence of absence was wrong and is retracted; what it does
show is whose voice dominates a window (her sentences at 192, his fragments in
the good session).

**Speaker volume is 120 of 255** (was 192), saved with `alsactl store`. Kept for
everyday use; he chose it by ear. **The array's own playback volume, through
which the echo reference reaches the canceller, is 40 of 60, minus 20 dB**, and
the reference measured 12dB under her echo at the canceller's input. Unexamined.

**Root cause: two USB clocks (Oct 1 2026, late).** Measured inside the chip by
routing the canceller's input (`AUDIO_MGR_OP_L 3 0`) and its residual for the
same mic (`AUDIO_MGR_OP_R 7 0` with `AEC_ASROUTONOFF` 0; all unsaved, restored)
during silent runs. With speakers on the AB13X and the reference sent to the
array, the canceller removed **about 1 dB**, the same with the reference as
without it, across ten runs. At sample resolution her echo slid against the
reference by 17 to 52 samples (1 to 3ms) within one 16s clip, in every run: two
independent clocks and two independent playback buffers, so the echo path never
holds still long enough for a linear filter to lock. Every channel 0 result
before this, the 30dB swings included, was post processing suppression working
around a canceller that was not cancelling.

With the speaker's 3.5mm plug in **the array's own jack**, one device and one
clock, array `PCM` at 60 for the test: minus 7 dB at first (a filter learned on
the old path), climbing over about two minutes of her speech to **14 to 15 dB**
over the last four runs, which is the ceiling of this measurement (her echo was
only 9 to 15dB above the room). It cancelled everything it could see. On the
2.0.6 firmware the jack is quiet at the array's `PCM` 40 and the `AIC3104`
level commands do not exist (status 65); they arrive in 2.1.

**The 48kHz firmware, tried and rolled back the same night (Oct 1 2026).**
`v2.1.1_48k2ch` flashed cleanly with `sudo dfu-util -R -e -a 1 -D <image>` and
came back as version 2.1.1, same USB IDs, chip control and the udev rule still
working without root, routing defaults unchanged, `AIC3104_HP_LEVEL` 8. **It
offers 48kHz only, for capture as well as playback**, two channels each, so
miles-voice, which opens the mic at `RATE` 16000, could not run on it. Through
the jack she **sounded like the AB13X by his ear**, and there were no clicks in
30s of quiet room or during a 17s playback (issue 39 not reproduced, small
sample). Rolled back to 2.0.6 so Nova worked overnight. Both images are kept in
`models/xvf3800_firmware/` (gitignored) for flashing either way.

**Next, in order:**
1. **Resample the mic (LIVE Oct 1 2026 11:12, on 2.1.1_48k2ch).**
   `mic_channel.ChannelPick` reads three times the frames, filters every channel
   on every read with its state carried across reads (121 tap Kaiser FIR, cut at
   7.5kHz: flat to 6.5k, 82dB down from 8.8k, 0.6ms per read on the Pi), and
   keeps every third sample, so readers still get 16kHz mono. Turned on by
   `MIC_CAPTURE_RATE` alone: 16000 today, which passes audio through exactly as
   before, 48000 after the flash; `audio.py` and `enroll.py` both read it. Tests
   break on the two traps: a filter restarted per read, and channel 0 going
   stale while barge in reads channel 1. Checked offline on archived 16kHz audio
   raised to 48kHz and brought back down: 60 wake hits moved by a median
   minus 0.005 (worst minus 0.148, none fell below 0.4), 30 of his recordings
   verified at a median 0.491 against 0.483. Live, first turns after the flash:
   five wakes at 0.66 to 0.96 (median 0.844, against 0.705 for 25 on 2.0.6),
   short commands verified at 0.42 to 0.51 (0.30 to 0.58 before), a 15s turn at
   0.641, every Deepgram transcript right, 13 percent of a core while
   listening. Empty follow ups (3 of 4) are not new: 10 of 15 on 2.0.6. **Not
   yet checked:** that channel 1 is still the recognition beam on 2.1.1; the
   wake evaluation in step 3 reads both channels and will show it.
2. **Move the speakers to the jack for good.** Flash 2.1.1 48k, speaker plug in
   the array, `SPEAKER_NAME_HINT` to the array, array `PCM` raised from its
   saved 40 (it was quiet), and `_Reference` with `AEC_REFERENCE` retired: the
   array's own playback is the reference. Allow two minutes of her speech for
   the canceller to learn the new path.
3. Rerun `record_wake_eval.py` with the canceller locked, before touching any
   chip setting. The wake model was deaf over her only ever while the canceller
   was not working.
4. Only then speech detection on channel 0, `PP_DTSENSITIVE`, or a retrain,
   whichever the evaluation points at.
5. A stop button: on 2.1.1, `MUTE_FUNCTION_ENABLE` 0 frees the mute button and
   `GPI_EVENT_PENDING` reads presses; persisting it needs `SAVE_CONFIGURATION`,
   ask first.

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
  [Barge in on the array](#barge-in-on-the-array-tried-sep-30-2026-off).
- **`xvf_host`**: direction of arrival, the LED ring, AEC and beam parameters.
- **48kHz firmware (`_48k2ch`)**: tried Oct 1 2026, and the speakers move to it;
  see [Barge in on the array](#barge-in-on-the-array-tried-sep-30-2026-off).

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

Its ceiling is the endpoint delay, because all it does is overlap downstream
work with time already spent waiting. At the old 2960ms endpoint that ceiling
was ~2400ms and it was the single largest lever available. At 900ms it is
**~500ms**, since triggering below 300 to 400ms of silence produces constant
false fires.

High complexity, touches the capture path, and it creates a problem it cannot
solve on its own: firing early means Nova sometimes starts talking mid thought,
which needs barge in to recover from. Build barge in because interrupting her
is worth having, not to chase 500ms.

**Barge in and speculative endpointing share their hard part**, which is
cancelling in flight pipeline work, including a TTS stream already writing to
aplay. Build that cancellation once and both become straightforward. That is
the right unit of work if either is wanted.

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

## Not blocked, and worth doing anytime

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

### Deepgram as the online path (BUILT Sep 25 2026)

Live in `src/deepgram_stt.py`; the mechanism is in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026) and
the privacy decision is in the decision log. What stays here is the reasoning.

Measured Sep 25 2026 and it wins on accuracy, on latency and on staying quiet in
an empty room. The numbers are in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026).
Cost is not a factor: $200 of credit, then $0.0043/min prerecorded, against a
median clip of 2.9 seconds and a few minutes of audio a day.

**Whisper stays.** This is an online upgrade with a local fallback, not a
replacement. `netcheck` already diagnoses which path exists, the phrase bank
already covers what Nova can say offline, and a remote engine failing is exactly
what the local one is for. A failed or slow call must fall through to whisper
rather than producing an empty transcript, which `deepgram_nova3` in the eval
script deliberately does not do, because there an empty string is the honest
score.

Two open questions, neither of them a measurement:

**1. Where it sits relative to speculation.** Speculative transcription starts a
whisper run 210ms into the endpoint silence and has 990ms before the recording
ends. A Deepgram call on those same frames returns in a median 520ms, so it fits
inside the same window and the existing `maybe_speculate` and
`cancel_speculation` structure is the natural home. The open part is what happens
when the remote call is still in flight at the end of the recording: wait on it,
or start whisper alongside and take whichever lands first. The second is more
robust and doubles the CPU cost of a turn.

**2. It sends the room to a third party, continuously.** This is the one that is
not technical. The eval was 58 clips uploaded once, with consent asked for. Live,
**every recording goes out, including the 45 percent of wakes that are false**,
which is how the neighbours and anyone visiting end up transcribed by a company
they have never heard of. The recordings are already treated as sensitive in this
repo and that line in CLAUDE.md exists for a reason. Options worth weighing
before wiring it up: gate the remote call behind `verify_voice` so only confirmed
speech from him leaves the Pi, which costs the speculation overlap; or accept it
and say so explicitly here.


### Rejecting a false wake before the Claude call: tested, do not build

Raised Sep 25 2026 on the grounds that `ignore` costs a full Claude round trip
per false wake, and 13 fired in 30 days. **Two candidate local signals were
measured against the labelled set and both fail. Do not reach for either again
without reading this.**

**Sentence initial case.** The idea was that room speech is caught mid
conversation, so whisper transcribes it starting lowercase, while a command to
Nova starts a sentence. Measured on the 58 labelled clips plus 28 real ignored
turns: most room speech starts **capitalised**, because whisper capitalises
whatever it hears first ("Oh, that's cool.", "Yeah.", "I mean she found the
little tree."). Worse, real commands start lowercase, including `'cancel the
timer.'` at 0.79 verification, which is the Sep 16 2026 incident. **The signal
would have rejected a real command.**

**Recording length.** The idea was that a recording running to `MAX_RECORD` is a
room that never went quiet rather than a request. Only 2 recordings have ever
exceeded 29 seconds, both from the Sep 13 guests evening under the old 60 second
cap, and the 25 to 29 second bucket is **empty**. Since `MAX_RECORD` came down to
30 and the wake word interrupts a recording, this guard would never fire.

**The cost was also overstated when this was raised.** `ignore` produces no
speech, so a false wake reaching Claude is inaudible to him. What he actually
experiences is the chime and the recording, and the level gate addresses a third
of those before either happens. What is left is roughly 13 Claude calls a month.

**What would actually separate them is semantic**, and the thing already doing it
correctly is the model. Any local version needs a labelled corpus of addressed
against not addressed transcripts, which does not exist; the 20 empty truth clips
in `data/stt_eval` are the start of one. Until it does, this is guesswork with an
asymmetric cost, and `local_intent.py` already sets the rule for that case:
decline when unsure, because falling through to Claude is merely expensive while
firing wrongly hangs up on him mid thought.


### A level gate on the wake window (BUILT Sep 25 2026, at -42)

Shipped in `src/wake_gate.py`; the live mechanism and its three safety
properties are in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).
What stays here is why the floor is where it is, and what the hardware change
does to it.

**It is not free.**
Reject a wake whose 2.5 second window never peaks above a floor, before the
chime, before recording, before Whisper, before Claude. The full measurement,
including why peak beats RMS and why the line is not -40, is in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).

The trade, on 38 real wakes and 22 false, Sep 25 2026:

| reject below | real wakes lost | false wakes cut |
|---|---|---|
| -44 dBFS | 0 of 38 | 5 of 22 |
| **-42 dBFS** | **0 of 38** | **7 of 22** |
| -40 dBFS | 2 of 38 | 13 of 22 |

**Start at -42.** It is the lowest line that costs nothing measured, and it
still removes about a third of false wakes, each of which currently costs a
chime, a recording, a transcription and a Claude call before `ignore` catches
it. -40 is tempting at nearly double the yield and it is where two genuine far
field wakes actually landed.

Three things to hold onto before building it:

- **A gate that rejects is a gate that can deafen her, and silently.** It must
  log every rejection with the level it measured, or the first far field
  regression is invisible. Prefer logging the level on every wake from the
  start, gate or no gate, so the floor can be moved on evidence later.
- **It is a stopgap for the neighbours, which is the array's job.** 13 of the 22
  false wakes are quiet and the array addresses their cause; the other 9 arrive
  loud and in the room, and nothing acoustic separates those. Do not let this
  gate stand in for the work in
  [INCIDENTS.md](INCIDENTS.md#the-near-misses-are-the-neighbours-sep-13-2026).
- **Mic dependent in the number, not the mechanism.** A new capsule and gain
  invalidate -42, and recalibrating is one afternoon of labelled clips rather
  than a retrain. That is what makes it reasonable to build before the hardware
  rather than after.

Only worth it if the level is already to hand at the decision point. It is:
`voice_main` keeps `_wake_window` as a rolling 2.5s of frames, which is what
`save_wake_hit` already writes.


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

### Session level verification (DONE Aug 11 2026)

Verify once per conversation session rather than once per turn. Follow ups are
short by nature ("yeah", "what about tomorrow") and will never embed reliably.

Shipped in `9c95893`, and **narrower than the proposal that sat here until
Sep 19 2026**. Session trust covers only follow ups under
`MIN_TRUSTWORTHY_SECONDS = 2.0` of embedded audio (`audio.py`). A follow up
longer than that is scored exactly as an initial command is, and a bad score
ends the conversation rather than just the turn (`voice_main.py`). A wake word
restart inside a recording is never session trusted, because it is a fresh
request and long enough to score.

**Those turns log `outcome='session_trust'` with `similarity = 0.0`**, which is
a placeholder and not a measurement. Filter on `outcome` before averaging
similarity or the zeros drag the mean down: over the 30 days to Sep 19 2026,
25 of 148 attempts were session trust.

**Security tradeoff, accepted deliberately:** the exposure is someone speaking
into the mic within ten seconds of the authenticated user, in the same room.
Current actions are weather, timers, reminders, and conversation, nothing
touching money, physical access, or outbound messages. **Revisit this if any
action with real consequence is added**, and use the `tier` column above rather
than reopening the whole decision.

**Note from the data:** the original justification, that follow up turns embed
poorly, is not what the logs showed. Follow ups kept 98.9 percent of their
audio after trimming versus 38.4 percent for initial commands, because
`listen_for_followup()` waits for speech onset while `record_command()` starts
capturing immediately. The security argument stands on its own, the embedding
quality argument does not.

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

### 1. SUPERSEDE: nothing can update a memory (DONE Aug 11 2026)

`memories.superseded_by INTEGER` is declared at `database.py:29` and appears
exactly once in the entire codebase. Nothing writes it. Verified Aug 11 2026:
zero rows have it set.

The consequence is that the memory store is append only in practice. A memory
recorded wrong stays wrong, and the only remedy is deleting the row by hand.

### 2. Expiry: volatile and references_date are write only (DONE Aug 11 2026)

`volatile` and `references_date` are both written by `save_memory` and read by
nothing. Fifteen seed rows are flagged volatile. Nothing expires them, nothing
filters on them, and nothing acts on a referenced date passing.

Also unranked: `get_episodic_memories` is `ORDER BY id DESC LIMIT 20`, with no
notion of importance or last use. Once more than twenty active memories exist,
older ones fall out of the prompt silently, by id, regardless of value.

Not yet urgent. There are currently zero active episodic memories, so nothing is
being displaced today. That is runway to build this properly rather than
retrofitting it under pressure, not a reason to skip it.

### 3. `remember` as a tool (DONE Aug 11 2026)

During the native tool use migration (Aug 11 2026) the question came up of
whether `[MEMORY:]` and `[MEMORY-EXPLICIT:]` should become a `remember` tool
alongside the action tags. Decision: **not yet**, and the reasoning is recorded
here so it is not relitigated from scratch.

The reason is **not** round trip cost. A `remember` tool with
`returns_to_model=False` costs zero extra Claude calls, which is the entire
purpose of that field. That argument was raised during the migration and it was
wrong.

The real reason is that **automating writes into a store that cannot be
corrected is worse than manual capture**. Manual capture produces errors a human
notices and fixes. Automated capture into an append only store produces errors
that are permanent, accumulate silently, and are only removable by hand editing
SQLite. The failure gets worse the better the tool works.

Built. `src/memory_tool.py`. The bracket tags are gone from the prompt and
`brain.py` strips any stray tag without saving it, so there is one write path.

Still open regardless: `get_episodic_memories` is `ORDER BY id DESC LIMIT 20`
with no ranking, so past twenty active episodic memories the oldest fall out of
the prompt silently, by id, regardless of value.

---

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
raise: Lower Strength leg extension 150 to 160 after 15, 15. Hevy replaces a routine whole on update and has no undo, so an edit
must keep the old routine locally first, and an exercise name has to resolve
to one of 451 templates, so a change is read back before it is sent. He builds
and edits routines in the app, not by voice. Logging workouts by voice is not
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
tuning argument in CLAUDE.md that quotes a similarity number is about
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
[SESSION_START.md](SESSION_START.md#a-typed-message-is-read-not-spoken-sep-15-2026-done).
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
