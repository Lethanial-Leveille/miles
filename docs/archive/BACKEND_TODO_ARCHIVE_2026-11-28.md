# Backend backlog archive, Nov 28 2026

Finished items moved out of [BACKEND_TODO.md](../BACKEND_TODO.md) on Nov 28 2026, text unchanged.

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
`192.0.2.10` on the tailnet.

Full reasoning is in the decision log in
[engineering decision log](../ENGINEERING_HISTORY.md#decision-log). Kept here only as a pointer, because the
same item was listed as deferred and a future session searching this file for
Tailscale should not find a stale "not done".

---

### Deepgram as the online path (BUILT Sep 25 2026)

Live in `src/deepgram_stt.py`; the mechanism is in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026) and
the privacy decision is in the decision log. What stays here is the reasoning.

Measured Sep 25 2026 and it wins on accuracy, on latency and on staying quiet in
an empty room. The numbers are in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026).
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
repo, as recorded in the earlier Claude guide. Options worth weighing
before wiring it up: gate the remote call behind `verify_voice` so only confirmed
speech from him leaves the Pi, which costs the speculation overlap; or accept it
and say so explicitly here.

### A level gate on the wake window (BUILT Sep 25 2026, at -42)

Shipped in `src/wake_gate.py`; the live mechanism and its three safety
properties are in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).
What stays here is why the floor is where it is, and what the hardware change
does to it.

**It is not free.**
Reject a wake whose 2.5 second window never peaks above a floor, before the
chime, before recording, before Whisper, before Claude. The full measurement,
including why peak beats RMS and why the line is not -40, is in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).

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
  [INCIDENTS.md](../INCIDENTS.md#the-near-misses-are-the-neighbours-sep-13-2026).
- **Mic dependent in the number, not the mechanism.** A new capsule and gain
  invalidate -42, and recalibrating is one afternoon of labelled clips rather
  than a retrain. That is what makes it reasonable to build before the hardware
  rather than after.

Only worth it if the level is already to hand at the decision point. It is:
`voice_main` keeps `_wake_window` as a rolling 2.5s of frames, which is what
`save_wake_hit` already writes.

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

## Closed experiments, moved Nov 28 2026

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
2. **Move the speakers to the jack (DONE Oct 1 2026).** Speaker plug in the
   array, AB13X unplugged, `SPEAKER_NAME_HINT` to "XVF3800". Array `PCM` 54 of 60:
   at 60 her loudest syllables sounded rough by his ear, so the level comes from
   the speakers' knob instead. `_Reference` and `AEC_REFERENCE` removed from
   `tts.py` and `config.py`; the array's own playback is the canceller's
   reference now. Playback still goes through ALSA's linear converter, 22050 to
   48000; by his ear that was not what sounded off. The canceller needs about two
   minutes of her speech on the new path before any measurement.
3. **Rerun the wake evaluation (DONE Oct 1 2026, session 114604).** After about
   two minutes of her speech on the jack. Scored with the same script over all
   four sessions, which reproduces every earlier figure in the table above:

   | Session | Over her, ch0 | Over her, ch1 | Quiet, ch0 / ch1 | Her alone, highest |
   |---|---|---|---|---|
   | 003125, two clocks | 0/30 | 0/30 | 9/9, 9/9 | 0.00 |
   | 004958, two clocks | 2/30 | 4/30 | 8/9, 8/9 | 0.00 |
   | 010635, two clocks | 0/18 | 1/18 | 6/6, 5/6 | 0.00 |
   | **114604, her on the jack** | 3/29 | **27/29** (28 at 0.30), median 0.99 | 8/9, 9/9 | 0.07 |

   Every spot: 9/10, 9/9, 9/10; the one real miss scored 0.11. **The wake model
   was never deaf over her; the canceller was not cancelling.** Channel 1 is
   still the recognition beam on 2.1.1. Channel 0 stays deaf over her (double
   talk suppression), so barge in listens on channel 1, as `BARGE_IN_CHANNEL`
   already says.

   **`BARGE_IN` on, Oct 1 2026 12:06.** First live use: two interrupts, both
   stopped her (1.0s and 1.2s into a piece); a reply left alone played to the
   end, watcher peak 0.00 throughout. **The logged peak on a HEARD line is not a
   margin.** The watcher stops at the first frame over `BARGE_IN_THRESHOLD`, so
   the peak is wherever the score was when it crossed (0.31, 0.40), not how high
   it would have gone; the evaluation, which runs on, has a median of 0.99. The
   watcher's fresh model per piece costs a little: the 0.40 frame scored 0.49
   with the model warmed on the 8.9s before it. Worth keeping one warm watcher
   only if live misses show up.
4. **Barge in on any speech of his ("wait wait wait... actually"), asked Oct 1
   2026.** He wants an interruption to be a conversation, not only the wake
   word. Measured the same day on session 114604, webrtcvad mode 2 on 30ms
   frames, over 326s of her playing:

   | Trigger | False stops on her | Caught him over her |
   |---|---|---|
   | ch1, level over minus 40 dBFS, speech for 300ms | **0** | **29 of 29** |
   | ch1, over minus 35 | 0 | 24 of 29 |
   | ch0, over minus 45 | 0 | 28 of 29 |

   With the canceller working her echo is speech on 1 percent of ch0 frames
   (46 to 94 percent on Sep 30, when it was not), median minus 55 dBFS; his
   voice over her peaks at minus 15 (ch0) and minus 26 (ch1). **Not shown:**
   one session, one room, and he said "hey nova", not "wait"; nothing here says
   whose speech it is, so someone else, a TV, a cough or a laugh would stop her.

   The build, in order, each tried before the next:
   1. Detect: `watch_for_interrupt` adds the rule above on `BARGE_IN_CHANNEL`,
      beside the wake word, with the floor and duration in `config.py`.
      **LIVE Oct 1 2026, at -37 dBFS** (`interrupt_speech.py`). Two sessions
      were added and all three replayed through the shipped code, a fresh VAD
      per take as the watcher makes one per sentence:

      | Floor, 300ms | Him, 29 takes | Claps, 30 (`barge_in_eval/20261001T134904`) | Her echo | Voices outside the room (`..._silence`, about 150s) |
      |---|---|---|---|---|
      | minus 40 | 29 | 1 | 1 | 7 |
      | minus 38 | 29 | 0 | 0 | 3 |
      | **minus 37** | **29** | **0** | **0** | **0** |
      | minus 36 | 28 | 0 | 0 | 0 |
      | minus 35 | 23 | 0 | 0 | 0 |

      About 1dB of margin each way; his farthest spot ("wherever you would
      naturally be") drops first. **webrtcvad is stateful**: shown only frames
      over the floor (the first version, to save CPU) it learned her loudest
      moments as the background and missed him 2 of 29, so it hears every frame
      (0.05ms per 80ms read). Every stop keeps its audio in
      `data/barge_in_stops`; retune from there.

      **Open: does a long silence leave the canceller worse?** On both eval
      sessions the take right after he changed spots was the worst in its
      session for echo (her clip's envelope on channel 1 at r 0.56 to 0.83
      against about 0.3), recovering by the next take; one of the four stopped
      her. Body moving or the longer quiet: the test meant to separate them
      (`..._silence`, her clips after 5s to 300s of quiet) was spoiled by people
      talking outside the room. Rerun it with the house quiet.

      **Answered by live use, Oct 5 2026: yes, after hours.** Of three speech
      stops Oct 2 to 5, two were her own first syllables (20261004T145558,
      20261005T001408), about 0.85s into the first reply after hours idle,
      at minus 30 dBFS on channel 1. Their envelopes follow her reply at r 0.94
      and 0.97; the real interruptions score 0.07 to 0.37. Raising the floor
      past minus 30 would lose him, so instead **a run that follows what she is
      playing is ignored** (`BARGE_IN_ECHO_CHECK`, live Oct 5): `play()` fills
      an `interrupt_speech.HerAudio`, placed on the clock from when aplay's
      buffer fills, and the watcher scores the last 1.2s of mic levels against
      hers at 30 to 300ms delay. His 28 takes over her score 0.69 at most and
      all still stop her through the shipped code; searching 1.5s of delay
      instead let one reach 0.91, which is why the window is narrow. **Not
      shown:** that live timing puts her delay inside that window. The archive
      does not record when aplay started, so the first live echo is the test;
      every sentence logs `echo score`, and every ignored stretch keeps its
      audio as `barge_in_stops/echo_*.wav`. A miss fails safe (she stops as
      before); a false match means she talked over him, so check those by ear.

      **First week live, Oct 5 to 12 2026:** no speech stops at all, so the
      echo check never had to act. One false stop, from the wake word instead:
      0.43 on near silence 14.9s into a reply (channel 1 about minus 50 dBFS,
      her echo at r 0.10, Whisper heard nothing), and he said "keep going".
      Its 2.5s window peaked at minus 35.1; his 37 takes over her peaked at
      minus 22.5 or louder. **A barge in wake now needs that window above
      minus 30** (`BARGE_IN_WAKE_FLOOR`, live Oct 12), or she keeps talking
      and the audio is kept as `wakegated_*`. One false stop and one session:
      check `wakegated_*` by ear after a week. On Oct 12 he heard her stop near
      "you need to either fall asleep faster", but the log, her archived reply
      and the watcher's own channel 1 audio all show her playing to the end;
      unexplained, and a clue to look for if it recurs.
   2. Keep what he said: the watcher's frames from his onset, with a preroll,
      seed the next recording, which endpoints as a command does. Channel 0
      chops his voice while she plays, so the interrupting stretch comes from
      channel 1; check verification and transcription on that mix.
      **LIVE Oct 2 2026.** Measured first (`record_wake_eval.py --sentence`,
      `barge_in_eval/20261002T004951`, a sentence shown at each cue, scored
      per channel with the pipeline's own Deepgram call and encoder):

      | | Words wrong, quiet | Over her | Verify, quiet | Over her (median, lowest) |
      |---|---|---|---|---|
      | channel 0 | 1.4% | 64.4% | 0.454 | 0.280, 0.229 |
      | channel 1 | 1.4% | 5.2% | 0.447 | 0.505, 0.408 |

      So the whole interruption is recorded on channel 1. Pessimistic: she
      played through his whole sentence there, and live she stops 0.34s in.
      On a speech stop the watcher hands off to `interrupt_speech.Carrier`,
      which keeps reading until `audio.claim_interruption()`; `flush_input`
      claims first, so there is one reader at a time on every path. The voice
      loop claims instead of flushing after a reply and seeds
      `listen_for_followup`. An interruption never teaches the voiceprint
      (`learn=False`). First live test: every interruption arrived whole
      ("Wait. I'm sorry. Can you say that a little bit slower?"). "Wait." then
      a pause was 0.48s of voice, under what verification takes, and ended
      the conversation; a short or noise interruption now reopens the window.

      **Not possible yet: "say it slower".** v3 ignores `speed` (VOICE_OUTPUT.md);
      she repeats at the same pace. Worth trying: told to slow down, she
      rewrites in short sentences with ellipses, which v3 does pause on. By
      ear before keeping it.
   3. A conversation: history keeps only what she said before he cut in,
      marked as interrupted, so "actually" reaches Claude in context.
      **LIVE Oct 2 2026.** `play()` records how long a sentence played before
      the stop (`heard_s`, wall time since the first write, capped at what was
      written); `interrupt_speech.words_heard` keeps that share of the words
      when the sentence's whole audio had arrived, or her pace (3.2 words a
      second, median of 73 replies) when it had not. Saved with "[You were cut
      off here.]", which the 30 word history trim keeps. First test: cut 4.5s
      into a pull ups answer, saved up to "cleanly on a dead", and "Wait.
      Sorry. I meant push ups." was answered as a correction; before this she
      argued "I didn't say anything about push ups." **Fixed with it:** cut
      during a tool turn's first line, the rest of the turn used to play
      through a second player, whose watcher read the mic beside the carrier.
      Once cut, the rest is collected silently and no bridge plays.
   4. Live tests, including other voices and a TV, to decide whether the
      interruption needs a voiceprint check. **DEFERRED to ordinary use (Oct 2
      2026, his call):** he runs it naturally and comes back if it goes badly.
      Every stop keeps its audio in `data/barge_in_stops`, named by where the
      speech began. When it is picked up: a `stops` set in `label_wake.py`
      (him, someone else, TV, her own voice, noise), then false stops per hour
      of her talking. Raise the floor first, measured on the recordings; a
      voiceprint check on the interruption is the last resort, because it
      needs about two seconds of speech and she would talk over him for that
      long. Also open: miles-server still runs the Oct 1 `ignore` wording;
      services restart only when he asks, and typed turns carry neither note.

5. A stop button: on 2.1.1, `MUTE_FUNCTION_ENABLE` 0 frees the mute button and
   `GPI_EVENT_PENDING` reads presses; persisting it needs `SAVE_CONFIGURATION`,
   ask first.

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
