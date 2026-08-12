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

**Utterance duration at verification was ruled out.** The apparent correlation
between embedded duration and similarity (`r = +0.426`) collapsed to `+0.130`
once a single degenerate row was removed, and attempts with three or more
seconds of embedded audio averaged 0.579, no better than the 1 to 2 second
bucket at 0.582. Duration mattered at **enrollment**, not verification.

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
- `analyze_verification.py`, with `--after` / `--before` / `--label` /
  `--include-degenerate`.
- `check_gain.py`, records or analyzes a clip and reports level against
  targets. **Mic dependent**: `MIC_MIXER_CARD` and `MIC_MIXER_CONTROL` in
  `config.py` will need updating for a new capsule.
- `enroll.py` rewritten (see below). **Not yet run.**

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
2. Gain tuned with `check_gain.py` to peak -12 to -6 dBFS on worst case
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
python3 check_gain.py --seconds 5    # mic pointed at the noise source, no speech
python3 check_gain.py --seconds 5    # mic null pointed at it, no speech
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

### 6. Mic selection

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

---

## Latency: where it stands and what is left

Perceived latency (speech end to first audio) went from **8070ms measured** to
roughly **5000ms estimated**, without touching the capture path in any way that
required rebuilding it. Estimated rather than measured because the last two
changes landed after the collection window; rerun `analyze_timing.py` after a
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
   `compare_whisper.py --model ../whisper.cpp/models/ggml-tiny.en.bin
   --worst-snr` once the recording archive has a few dozen real commands.
2. **A second prompt cache breakpoint on conversation history — 200 to 400ms,
   untested.** Only the system prompt is cached today; the twenty messages of
   history sit after the breakpoint and are re prefilled every turn. Four
   breakpoints are available and one is used. Note that trimming assistant
   turns to thirty words already cut this cost, so measure before adding more.
3. **Ask for a short opening sentence — 200 to 300ms, free.** The 608ms
   sentence assembly stage is the model generating the first sentence before
   anything can be spoken. A shorter opener starts audio sooner and serves the
   brevity goal at the same time. Test it the way the length instruction was
   tested, with repeated sampling.
4. **Run verification concurrently with transcription — 275ms.** They are
   independent and both operate on the same wav. Needs care so the transcript
   still reaches the verification log, which is why it was not done inline.

## Not blocked, and worth doing anytime

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

### Session level verification

Verify once per conversation session rather than once per turn. Follow ups are
short by nature ("yeah", "what about tomorrow") and will never embed reliably.

Proposal: verify against the first command, then trust session state for
follow ups inside the window. If a follow up is long enough to embed well and
scores badly, drop the session. If it is too short to embed, accept on session
state.

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

### 3. `remember` as a tool (UNBLOCKED, not built)

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

SUPERSEDE and expiry both landed Aug 11 2026, so this is no longer blocked. It
is simply not built yet: bracket tags still capture memories and
`extract_memories` in `parsing.py` is still what strips them.

What building it now involves: a `remember` tool with `returns_to_model=False`,
carrying content plus the classification the model is already choosing badly by
implication (volatile or not, and a date when volatile). The reason to give the
tool an explicit expiry argument is that the current implicit path cannot set
one, which is why fifteen seed rows are flagged volatile with no date and can
never expire.

Still open regardless: `get_episodic_memories` is `ORDER BY id DESC LIMIT 20`
with no ranking, so past twenty active episodic memories the oldest fall out of
the prompt silently, by id, regardless of value.

---

## Quick reference

```bash
# Analyze verification data
python3 analyze_verification.py
python3 analyze_verification.py --after 2026-09-01 --label "dorm, new mic"

# Check capture level (voice service holds the mic, so stop it first)
sudo systemctl stop miles-voice
python3 check_gain.py --seconds 5
sudo systemctl start miles-voice
python3 check_gain.py --file ../build/command.wav   # no service stop needed

# Enrollment (only after mic, room, and gain are final)
sudo systemctl stop miles-voice
python3 enroll.py
sudo systemctl start miles-voice
```

Key config in `config.py`: `VERIFY_THRESHOLD`, `VAD_MODE`, `VAD_PREROLL_MS`,
`VAD_ONSET_FRAMES`, `TTS_FLUSH_MARGIN_MS`, `EXPECTED_MIC_GAIN`,
`MIN_VOICED_SECONDS`, `MAX_FOLLOWUP_TURNS`.
