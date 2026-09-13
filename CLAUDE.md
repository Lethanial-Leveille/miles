# M.I.L.E.S. — Modular Intelligent Learning & Execution System

> **Precedence rule.** This file describes the repo. The repo is the authority.
> If anything here conflicts with source, **the source wins**, and whoever finds
> the conflict fixes this file in the same session. Every claim below was
> verified against source or a live command on Aug 11 2026. Claims that cannot
> be checked cheaply are marked UNVERIFIED.
>
> Read `docs/SESSION_START.md` before proposing changes. It carries the preflight
> that catches drift and the decision log that explains why things are the way
> they are.

## Current Status

v0.7.1 shipped. Unreleased work has landed on top of it since (latency
instrumentation, webrtcvad endpointing, recording archive, dismiss action,
clock injection). All three systemd services enabled and active. Tunnel live at
miles.lethanial.com, dashboard managed.

DO NOT modify existing services unless explicitly asked. They are running in production.
DO NOT edit ~/.cloudflared/config.yml. Tunnel config is dashboard managed.
DO NOT touch the Swift companion app (separate repo at github.com/Lethanial-Leveille/miles-app).

## What This Project Is

A personal AI voice assistant running on a Raspberry Pi 5 (8GB). The voice
personality is named Nova. M.I.L.E.S. is the system name on the resume and GitHub.

Built by Lethanial Leveille, CpE student at University of Florida, Class of 2029.
Primary portfolio project targeting embedded/firmware and full stack hardware to
cloud engineering roles.

The companion iOS app (Nova) is a separate repo at github.com/Lethanial-Leveille/miles-app.

## Hardware

Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite 64 bit. Razer Seiren V3 Mini
USB microphone, found dynamically by name in PyAudio rather than hardcoded to a
card number, since ALSA card numbers shift on reboot. Amazon Basics USB speakers
via QianLink USB to 3.5mm adapter, also resolved by name at runtime in `tts.py`
using `SPEAKER_NAME_HINT` rather than a fixed device string. Development machine
is a MacBook connected via VS Code Remote SSH at theycallmelee@miles.local, local
network only. Remote SSH via Cloudflare Tunnel is planned and not yet configured.

Mic gain is tuned to `EXPECTED_MIC_GAIN = 23` and persisted with `alsactl store`.
It is checked and logged at startup, because a silent revert corrupts collected
data in a way that only shows up days later as inexplicably low scores.

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

## Repo Structure

```
~/miles/
├── src/
│   ├── config.py              # Constants, paths, thresholds, env vars
│   ├── prompts.py             # System prompt assembly
│   ├── database.py            # SQLite operations + schema migrations
│   ├── parsing.py             # Memory/action tag extraction, noise transcripts
│   ├── actions.py             # Weather, timer, reminder handlers
│   ├── audio.py               # Mic, wake word, VAD, Whisper, Resemblyzer
│   ├── tts.py                 # ElevenLabs synthesis, speaker resolution, aplay
│   ├── phrasebank.py          # Pre rendered Victoria, played from disk, no network
│   ├── netcheck.py            # Why a network call failed, answered locally
│   ├── local_intent.py        # Answer timers and dismiss without Claude
│   ├── stream_router.py       # Delta buffering, sentence flush, action tag detect
│   ├── tools.py               # Tool registry, @tool decorator, schema/prompt generation
│   ├── system_state.py        # get_system_state tool: uptime, temp, latency, commit
│   ├── tier_tool.py           # lower_access tool: demotion by voice, never escalation
│   ├── timing.py              # Per turn latency instrumentation
│   ├── brain.py               # ask_nova orchestrator
│   ├── auth.py                # JWT + bcrypt password hashing
│   ├── server.py              # FastAPI REST + WebSocket
│   ├── voice_main.py          # Audio loop entry point
│   ├── setup_auth.py          # One time password and JWT secret setup
│   ├── enroll.py              # Voice enrollment
│   ├── seed_memories.py       # Seed corpus loader
│   ├── analyze_timing.py      # Latency analysis over timing_log
│   ├── analyze_verification.py # Speaker verification analysis
│   ├── check_gain.py          # Mic gain verification
│   ├── compare_whisper.py     # Whisper model comparison harness
│   ├── profile_turn.py        # Times the turn stages no stopwatch covers
│   └── tests/                 # pytest suite (367 tests as of Aug 12 2026)
├── docs/
│   ├── SESSION_START.md       # Preflight, drift rules, decision log
│   └── BACKEND_TODO.md        # Deferred work, written to be picked up cold
├── assets/                    # wake_chime.wav
├── scripts/
│   ├── memory.py              # list/fix/chain/temporary on stored memories
│   ├── retrieval.py           # review and label what retrieval returned
│   ├── people.py              # people, tiers, birthdays
│   ├── voiceprint.py          # improve the voiceprint from real use
│   ├── encoder_bench.py       # compare speaker encoders on archived clips
│   ├── pronounce.py           # pronunciation aliases, live with no restart
│   ├── render_phrases.py      # render/trim the phrase bank, run while online
│   ├── label_transcripts.py   # hand labelled truth + word error rate per STT
│   ├── label_speakers.py      # label who is speaking in the eval clips, by ear
│   └── healthcheck.py         # what broke, on a timer, not by trying to use it
├── systemd/                   # miles-health units, versioned (the other three are not)
├── README.md                  # Public facing project description
├── .env                       # Password hash, JWT secret, API keys (gitignored)
├── models/                    # Wake word + voiceprint (gitignored contents)
├── build/                     # Temp WAV files (gitignored)
├── data/                      # SQLite database, seed corpus, recordings (gitignored)
├── whisper.cpp/               # Compiled from source (gitignored)
└── CLAUDE.md
```

## Production Infrastructure

### systemd Services (all enabled, running on boot)

- `miles-voice.service` runs voice_main.py (room mic pipeline)
- `miles-server.service` runs uvicorn server:app on port 8000 (FastAPI)
- `miles-tunnel.service` runs cloudflared tunnel (Cloudflare Tunnel)

```bash
sudo systemctl status miles-voice miles-server miles-tunnel
sudo systemctl restart miles-voice
journalctl -u miles-voice -f
journalctl -u miles-voice -n 50
```

### Cloudflare Tunnel (Dashboard Managed)

Tunnel name: miles. Routing changes happen at https://one.dash.cloudflare.com
under Networks > Tunnels > miles > Public Hostnames.

Active routes:
- miles.lethanial.com to http://localhost:8000 (primary)
- api.lethanial.com to http://localhost:8000 (legacy fallback)

Domain lethanial.com registered through Cloudflare. Free Zero Trust tier.

### FastAPI Endpoints (miles.lethanial.com)

REST: /auth/login, /auth/refresh, /chat, /memories, /memories/{id}, /history, /status, /docs
WebSocket: /ws

Auth: JWT, HS256, Authorization Bearer header. Access tokens 7 day expiry.

## Key Config Values (config.py)

Verified against source Aug 11 2026. When these drift, this table is the first
thing to fix.

### Audio capture
- `CHUNK = 1280` (80ms frames, required by openWakeWord)
- `RATE = 16000`, `CHANNELS = 1`
- `WAKE_THRESHOLD = 0.4`
- `VERIFY_THRESHOLD = 0.5`
- `VAD_MODE = 2` (webrtcvad, replaced the amplitude threshold that never fired)
- `VAD_PREROLL_MS = 300` (retains frames before onset so soft leading consonants survive)
- `VAD_ONSET_FRAMES = 2`
- `SILENCE_LIMIT = 1.2` (3.0 to 0.9 was the largest latency win; 0.9 to 1.2 bought back truncation)
- `MAX_RECORD = 18.0` (was 15.0, which truncated 15 percent of follow ups)
- `TTS_FLUSH_MARGIN_MS = 250`
- `EXPECTED_MIC_GAIN = 23`
- `MIC_NAME_HINT = "Seiren"`, and `MIC_MIXER_CARD` is **derived** from it at
  import by `_resolve_mixer_card`, never written literally. It is `None` when no
  card matches, which callers report as "cannot check" rather than passing.

### Whisper
- `WHISPER_MODEL = ggml-base.en-q8_0.bin` (quantized locally from base.en with
  `whisper-quantize`; the file is derived, so regenerate it after a whisper.cpp
  upgrade)
- `WHISPER_INITIAL_PROMPT = None` — see below, this is a decision not an omission
- `WHISPER_AUDIO_CTX = 1000` (20 seconds; default 1500 padded every clip to 30s
  and cost a flat ~2000ms regardless of input length. Do not lower without
  rerunning the validation, 750 and 900 both corrupted reference speech.)

Measured on 40 archived recordings, Aug 12 2026, with `compare_whisper.py`:

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

### Scoring an STT change

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

### Wake miss capture
- `CAPTURE_WAKE_MISSES = True`, `WAKE_MISS_DIR = ~/miles/data/wake_misses`
- `WAKE_MISS_FLOOR = 0.05` (below `WAKE_LOG_FLOOR`, deliberately)
- `WAKE_MISS_MAX_FILES = 400`, `WAKE_MISS_PREROLL_MS = 2500`

Keeps the audio behind a near miss, because a score alone cannot distinguish
"he said the phrase and the model missed it" from "that was the television", and
those need opposite fixes. Files are named by score first, so the worst sort to
the top. Captured on a lower floor than the log, since the failures that matter
most may be the ones scoring near zero, which is exactly what `WAKE_LOG_FLOOR`
hides.

This is step one of the plan in `docs/BACKEND_TODO.md`. Retraining
`hey_nova.onnx` is not possible without it: there is nothing to evaluate against.

### Speculative transcription
- `SPECULATIVE_TRANSCRIBE = True`, `SPECULATIVE_SILENCE_MS = 450`,
  `SPECULATIVE_THREADS = 3`

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

### Model
- `MODEL_A = "claude-haiku-4-5"` — **this is the production model**
- `MODEL_B = "claude-sonnet-4-5-20250929"`
- `MODEL_AB_TEST = False`

Haiku won a measured A/B against Sonnet over twenty turns: 614ms faster on
median time to first token, 31 percent lower, p=0.0007 on a permutation test.
Pin the A/B off before measuring anything cache related, since prompt caches
are model scoped and alternation makes every turn a miss.

### Prompt and caching
- `PROMPT_CACHING = True`
- `HISTORY_ASSISTANT_WORDS = 30` (past assistant turns trimmed before being sent
  as context; history anchors response length far more strongly than any
  instruction does)
- `LOOKAHEAD_CHARS = 50`
- `ACTION_PREFIX = "[ACTION:"` — **consumed only by StreamRouter.** It is not
  passed to the API as a stop sequence. See the streaming section below.


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

### Conversation loop
- `MAX_FOLLOWUP_TURNS = 6`
- Exit is intent based, not a phrase list. Nova emits `[ACTION: dismiss]`.

### TTS
- `TTS_VOICE_ID` = Victoria (`qSeXEcewz7tA0Q0qk9fH`), defined in config.py. The
  single place a voice is named; no call site references one.
- `DEFAULT_TTS_MODEL = "eleven_flash_v2"`
- `EXPRESSIVE_TTS_MODEL = "eleven_v3"` (HTTP only, no WebSocket)
- `TTS_OUTPUT_FORMAT = "pcm_22050"` (raw S16_LE mono, piped to aplay)
- `SPEAKER_NAME_HINT = "AB13X"` (resolved by name at runtime in tts.py)

### Recording archive
- `ARCHIVE_RECORDINGS = True`, `ARCHIVE_DIR = ~/miles/data/recordings`
- `ARCHIVE_MAX_FILES = 600` (~150MB, a few weeks of normal use, oldest pruned first)

Recordings are of a real person in a real room. Treat them accordingly.

### Phrase bank (offline speech)
- `PHRASE_DIR = ~/miles/data/phrases`, gitignored. `PHRASES` in `phrasebank.py`
  is the versioned source of truth; the WAVs are derived from it.
- `ACK_SPOKEN_CHANCE = 0.5` — how often a wake gets a spoken ack instead of the
  chime.

Rendered Victoria played by aplay. `TTS_OUTPUT_FORMAT` is `pcm_22050` and the
chime is 22050Hz mono 16 bit, so the two are the same format and nothing
resamples.

Measured Aug 12 2026, after trimming, blocking wall time through the same aplay
path: chime 391ms, shortest ack ("Yep?", 0.358s) 388ms. **Indistinguishable in
mechanism, but not free in production**: the chime is fire and forget and
overlaps recording, while an ack must block, so an ack costs roughly 390ms that
the chime costs nothing.

An ack cannot overlap capture. webrtcvad at mode 2 does not read a tone as
speech, which is why the chime is safe, but an ack is speech: overlapped, it
endpoints the recording on Nova's own voice and lands at the head of the clip
Resemblyzer scores against his voiceprint.

**`play()` returns the text it played, not a bool.** The caller writes that to
history, and a bool meant the caller had to guess which variant ran. It guessed
index zero, so history could record "Goodnight, Lethanial." on a turn where she
actually said "Talk soon."

**`NIGHT_ONLY` filters variants by clock.** A conversation ends at any hour, and
"goodnight" at two in the afternoon is worse than no farewell. Kept as an index
filter rather than a separate key so a variant already auditioned and committed
keeps its file.

**Pinning the render seed freezes bad draws.** CLAUDE.md already notes that at
`stability = 0.80` the name is the unstable part: 0.75 "occasionally missed the
name", 0.90 "held the name but read flatter". Live, a bad reading of it passes
in one turn. Rendered, it is permanent. Use `render_phrases.py audition <key>
<index>` then `pick` for anything containing his name. Auditioning makes the
bank strictly better than live synthesis here, because live re-rolls the name
every time and can always draw badly.

ElevenLabs pads each clip with dead air, up to 232ms measured. `render_phrases.py
trim` strips it in place with no API call, kept separate from `render` because
ElevenLabs treats `seed` as best effort, so re-rendering risks redrawing the voice.

### Tuned from live use, Aug 12 2026

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

**`SILENCE_LIMIT` 0.9 to 1.2.** It was demonstrably cutting him off: live
transcripts read `'set a timer for'` and `'Start give me start five minutes.
I'm'`. This normally costs perceived latency 1:1, but speculative transcription
changes that: 300ms more waiting is also 300ms more overlap with the 1166ms
Whisper stage, so the two very nearly cancel. That does not generalize past the
point where transcription is fully hidden.

**`ACK_SPOKEN_CHANCE` 0.5 to 0.75**, and **`FOLLOWUP_TIMEOUT` 2.5 to 3.0**.

**Timer alerts said "your 5 minutes timer is up".** `_plural` is right for
"Timer set for five minutes", a quantity, and wrong before a noun, where the
unit is a modifier and must be singular. `_attributive` handles that role and
`_spoken_amount` spells the digit. Two grammatical roles, two helpers.

**Farewells now answer what he said.** `THANKS_ONLY` restricts "Anytime." and
"Any time at all." to dismissals that actually thanked her. "Never mind" is a
retraction, and answering it with "Any time at all" reads as not having
listened. Same index filter mechanism as `NIGHT_ONLY`, chosen so already
rendered files keep their numbering.

**The `ignore` tool.** Overheard speech used to get "I'm not part of that
conversation. Let me know if you need anything.", which is itself joining the
conversation and costs him a wait to hear. `TurnResult.ignored` is distinct from
`dismissed`: dismissed ends a conversation he ended, ignored ends one he was
never having with her, and the difference is whether she says anything at all.
In the follow up loop it breaks the conversation rather than reopening the
window, so an exchange nearby cannot hold her attention turn after turn.

### Local intent
- `INTENT_SIMILARITY_THRESHOLD = 0.55`, `MAX_DISMISS_WORDS = 6`

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

### The turn budget, measured Aug 12 2026

| Stage | Median |
|---|---|
| endpoint | 880ms |
| transcribe | 1166ms |
| verify | 193ms |
| claude_ttft | 1253ms |
| first_sentence | 288ms |
| tts_ttfb | 349ms |
| genuinely unaccounted | 32ms |
| **perceived** | **4325ms** |

Re-read on the Aug 12 window, after `SILENCE_LIMIT` went to 1.2 (n=44, Haiku,
non action): endpoint 1200, transcribe 1089, verify 190, claude_ttft 1350,
first_sentence 230, tts 382, perceived p50 4324. **The sum of the stage medians
exceeds the total by 117ms and that is arithmetic, not a missing stage**: each
stage's median comes from a different turn, and medians do not add. Do not go
looking for overlap to explain a residual of this size in either direction.

**Local intent turns carried no perceived latency until Aug 13 2026.** They
never reach `tts.speak`, so nothing closed out the measurement and every one of
them logged a null total. They are also the fastest turns the system has, so
the reported median described the Claude path rather than the room, and it did
so while local intent was answering a fifth of all turns. `phrasebank.play` now
calls `timing.note_local_audio`, migration 22 adds `local_intent` to
`timing_log`, and `analyze_timing.py` reports the two paths separately in
section 1 and gives local turns their own stage table in section 2b.

A number that excludes the fast cases is worse than no number, because it looks
like a measurement. Any turn that produces audio has to close out
`total_perceived_ms`, whoever produced it.

**There is no missing time.** An earlier reading of this table left
`first_sentence_ms` out of the sum and reported a 325ms hole; counting it as its
own stage closes the residual to 32ms. `profile_turn.py` confirmed the other
suspects are noise: `_write_wav`, `archive_recording` and the entire prompt
assembly including hybrid memory search total 3.1ms.

`first_sentence_ms` at 288ms is real optimizable time: Claude's first token has
arrived, but StreamRouter is still buffering `LOOKAHEAD_CHARS = 50` and waiting
for a sentence boundary before anything can reach TTS.

### Other
- `DEFAULT_LOCATION = "Gainesville"`
- `DB_PATH = ~/miles/data/miles.db`
- `MIN_VOICED_SECONDS = 4.0`, `ENROLL_RECORD_SECONDS = 10`

## Environment Variables

`~/.bashrc`:
- ANTHROPIC_API_KEY
- WEATHER_API_KEY
- FISH_API_KEY (retained for rollback only)

`~/miles/.env` (gitignored):
- MILES_PASSWORD_HASH
- MILES_JWT_SECRET
- ELEVENLABS_API_KEY

The voice id used to live here. It moved to config.py: a voice id is neither
secret nor deployment specific, and keeping it in a gitignored file meant voice
changes carried no history. `ELEVENLABS_VOICE_ID` is now unused and can be
deleted from .env.

## Tech Stack

Wake word: openWakeWord (hey_nova.onnx).
VAD: webrtcvad, mode 2.
STT: whisper.cpp (base.en, greedy decoding, NEON ARM optimizations, capped audio context).
LLM: Claude API, `claude-haiku-4-5`, streaming, prompt caching on the system prompt.
TTS: ElevenLabs (Victoria, eleven_flash_v2), pcm_22050 piped to aplay.
Voice auth: Resemblyzer (256 dim cosine similarity).
Memory: SQLite WAL mode, schema migrations to version 21.
Weather: OpenWeatherMap.
Backend: FastAPI + uvicorn + python-jose + passlib (bcrypt) + python-dotenv.
Tunnel: cloudflared (dashboard managed).
Process management: systemd.
Language: Python 3.13.5.

## Voice Settings (config.py)

```python
TTS_VOICE_ID = "qSeXEcewz7tA0Q0qk9fH"        # Victoria

TTS_VOICE_SETTINGS = VoiceSettings(
    stability=0.90, similarity_boost=0.75, style=0.00,
    use_speaker_boost=True, speed=1.00,
)
```

`VOICE_WITTY` and `VOICE_SERIOUS` are also defined and currently inert:
`speak()` falls back to `TTS_VOICE_SETTINGS` unless a caller passes an override,
and nothing selects one yet.

`stability` was tuned by ear, not chosen: 0.45 no, 0.60 eh, 0.75 pretty good,
0.90 good, 1.00 good, then the band between 0.75 and 0.90 searched directly.
0.60 was live during the whole period the voice sounded inconsistent.

Raised to 0.90 on Aug 12 2026 from live use: 0.80 read as too expressive and was
mispronouncing his name, which is the exact pair 0.90 is documented to address.

**The phrase bank renders at whatever this is.** Changing it means
`render_phrases.py render --force` over all of data/phrases, or the cached clips
and live speech drift apart. It also discards any variant already auditioned, so
re-pick those after.

It is one dial with two failure modes. Stability buys consistency by reducing
variation, and that same variation is what makes delivery sound alive, so 0.75
read better but occasionally missed the name and 0.90 held the name but read
flatter. Change it only after listening.

`use_speaker_boost` is NOT supported on eleven_v3. Drop it when targeting v3.

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

## Pronunciation

`pronunciations` table, one row per grapheme, seeded with Lethanial to
Luhthanyul. `tts.normalize_pronunciation` runs inside `speak()`, immediately
before the ElevenLabs call, so an alias reaches the synthesizer and nothing
else. Whole word, case insensitive, longest grapheme first.

`database.upsert_pronunciation()` adds entries at runtime with no migration.

`arpabet` is usable once `TTS_PHONEME_TAGS` is on, which requires a model that
honors phoneme tags. Measured Aug 11 2026: flash v2 honors them, flash v2.5
**drops the tagged word entirely**. Plain "Lethanial" gave 0.79s of audio;
wrapped in a phoneme tag on v2.5 it gave 0.23s, and deliberately absurd
phonemes gave the same 0.23s. On v2, correct phonemes matched plain at 0.74s
and absurd phonemes stretched to 1.21s.

Model is `eleven_flash_v2` for that reason. The switch is free: median time to
first byte over five runs was 349ms on v2 against 347ms on v2.5.

### Changing a pronunciation

`scripts/pronounce.py`. Aliases live in the database and `speak()` reads them
per sentence, so a change is live on the next thing Nova says with **no
restart**.

```bash
python3 scripts/pronounce.py list
python3 scripts/pronounce.py try Lethanial Luthanyull Lah-than-yull
python3 scripts/pronounce.py set Lethanial Luthanyull
```

## Streaming and Action Tags (how it actually works)

**No stop sequences are passed to the API.** Generation runs to completion. The
previous version of this document claimed `stop_sequences=["[ACTION:"]` was sent
on every call; that was never true in code and the claim caused a full session
of work to be planned against a wrong premise. This section is now written from
`stream_router.py` and `brain.py`.

1. `brain.py` iterates `stream.text_stream` and feeds every delta to `StreamRouter`.
2. `StreamRouter` buffers `LOOKAHEAD_CHARS` (50) before emitting anything, which
   guards against a stray `[` in prose being read as a tag. Seeing `ACTION_PREFIX`
   ends that wait early, because it is unambiguous.
3. Complete `[ACTION:...]` tags are stripped from the buffer into `router.action_tags`.
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

## Latency (measured, not estimated)

From `timing_log`, 48 rows, Aug 10 to Aug 11 2026. Segmented by endpoint delay,
which reveals which `SILENCE_LIMIT` was live:

| Era | n | Median perceived | Median endpoint |
|---|---|---|---|
| Before `SILENCE_LIMIT = 0.9` | 20 | 8088ms | 2960ms |
| After | 28 | **4938ms** | 920ms |

Haiku, post change, non action turns: n=19, median perceived 4977ms, median
time to first token 1318ms.

Perceived latency is measured from the moment the user stopped talking to the
moment the first audio chunk is flushed to aplay. ALSA adds buffer delay after
that point, so the true figure at the speaker is slightly higher and
consistently so.

Do not quote a latency figure in this file that was not read out of `timing_log`.

## What Is Completed

- v0.1: Wake word detection + Whisper transcription
- v0.2: Claude API integration + Fish Audio TTS
- v0.3: Resemblyzer speaker verification + energy based VAD
- v0.4: SQLite persistent memory + conversation history
- v0.5: Action tag system (weather, timers, reminders, cancellations)
- v0.6: Multi turn follow up conversation loop + exit phrases
- v0.7: Module refactor + JWT auth + FastAPI server + Cloudflare Tunnel +
  systemd services + miles.lethanial.com + Nova iOS app on physical device
- v0.7.1: ElevenLabs TTS migration + Claude streaming with StreamRouter +
  concurrent TTS queue + wake chime + Miles pronunciation normalization

Landed since v0.7.1, unreleased:
- Per turn latency instrumentation (`timing.py`, `timing_log`)
- webrtcvad endpointing, `SILENCE_LIMIT` 3.0 to 0.9
- Whisper audio context cap, Haiku migration, prompt caching
- Response length rewrite + history trimming (median 76 words to 54)
- Exit phrase list replaced by intent based `[ACTION: dismiss]`
- Command recording archive
- Clock injected into the last user turn (`_with_current_time`)
- Offline resilience: `run_turn` exception boundary in voice_main, phrase bank,
  `netcheck` cause diagnosis, spoken wake acks
- Local intent for `set_timer`, `time_of_day`, `cancel_reminder`, `dismiss`
- Speculative transcription during the endpoint wait

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

## The wake word stopped separating in the new room

Sep 8 2026. "It responds even though I haven't said hey nova" turned out to be
two independent defects.

### Nova said "Done." out loud when she correctly stayed silent

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

### No threshold separates true from false wakes any more

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

### The near misses are the neighbours, and level is what decides

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

### `scripts/label_wake.py`

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

## Two guards that could not do their jobs

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

## Reminders are fired by a poller, not by a thread

Changed Sep 6 2026. `set_reminder` used to write a row and then spawn a
`threading.Thread` that slept until the due time and fired from there.

**That made the thread the state and the row a record of it.** The row survived
a restart and the thread did not, and nothing scanned the table at boot, so
every pending reminder was silently dropped by any deploy or crash. With
`Restart=always` on the unit, both are routine. A reminder set for tomorrow
morning simply never happened, and the row sat at `completed = 0` forever with
nothing to distinguish it from one still waiting.

It had not bitten yet only because every reminder ever set was a one minute
test that fired before anything restarted.

`actions.poll_reminders` now reads `reminders` every `REMINDER_POLL_S` (20s) and
fires whatever is due. **There is no boot rearm, which is the point:** nothing is
held in memory, so a restart is just the next poll. `start_reminder_poller` is
called once from `voice_main.py`.

**Only the voice loop polls.** Both processes can create reminders and exactly
one may deliver them. A poller in the server would race for the same rows, and
the claim below would keep that correct while the alert landed in a queue
nothing drains.

**This fixed a second bug for free.** A reminder set through the app used to
spawn its thread inside the uvicorn process, so the alert queued into *that*
process's `alerts._pending`. `server.py` never calls `take_for_speech`, and
`brain.py` only folds, so it was delivered only if another chat message arrived
inside the 15 second fold window, and otherwise lost without even reaching
`alert_log`. Under the poller the row is created by whichever process and fired
by the voice loop.

**Completion happens before the alert is queued, deliberately.**
`complete_reminder` returns whether it changed a row, so the UPDATE doubles as a
claim and two passes cannot both win the same reminder. Firing first would be at
least once, whose failure mode is announcing every twenty seconds forever if
completion keeps failing. Claiming first is at most once, whose failure mode is
losing one reminder if the process dies in the microseconds between the commit
and the in memory append. The second is rarer and far less bad.

**Completion is by id.** The old code matched on `content AND due_at`, so two
reminders agreeing on both were closed by a single firing and only one was ever
spoken.

**A due time in the past is stored and announced, not dropped.** It is a bug
when it happens, almost always the clock guidance being ignored, and `alerts.py`
argues that silent non delivery is the worst available outcome. Past
`REMINDER_LATE_S` (1 hour) the wording says it came due while he was away,
because delivering a four hour old reminder as though it had just fired makes
the clock look broken.

**Timers are still in memory threads and do not survive a restart.** They are
not persisted at all, so there is no table and no record one ever existed.
Making them durable is a separate decision, not an oversight.

## What Is Next

- Native tool use migration, replacing bracket action tags. See
  `docs/SESSION_START.md` decision log for scope and the cache margin gate.
- SSH via Cloudflare Tunnel: ssh.lethanial.com + Cloudflare Access policy
- v0.8+: Morning briefings (Fall 2026), Mac control, HealthKit, EventKit,
  MusicKit, WeatherKit migration, barge in detection, parallel processing

## Coding Preferences

No hyphens in any written output, ever. Numbers spelled as words in Nova's
spoken responses. Nova calls me "Lethanial" by default. "Lee" is allowed but rare, and only
when something has genuinely gone well; it is a condition, not a frequency. Do not use
APScheduler yet. Python style: flat is better than nested, clarity over
cleverness. Comments explain WHY not WHAT.

## Critical Learning Constraint

I am a CpE student building this to learn, not just to ship. When making code changes:

1. Explain what we are about to do before writing code
2. Make small, logical changes one at a time
3. When introducing a new concept, explain what it does and why before using it
4. Do not refactor things outside the current task scope
5. Do not use hyphens in anything you write for me
6. If I push back, defend your position if you believe it is right. I value honesty above everything.

## Session Persistence

Claude Code sessions die on SSH disconnect or Pi reboot. Always run inside tmux:

```bash
tmux new -s miles
# Ctrl+B then D                # detach
tmux attach -t miles           # reattach
```

Before ending any session, work the end of session checklist in
`docs/SESSION_START.md`. That is what keeps this file true.

## Known Quirks

ALSA and JACK stderr suppressed via ctypes error handlers and a
`silence_stderr()` context manager in audio.py. Supercardioid mic means short
phrases at desk distance score lower on verification. OpenWeatherMap geocoder
works best with city names only. whisper.cpp compiled with NEON ARM optimizations.

whisper.cpp does not reliably emit `[BLANK_AUDIO]` for non speech input. Fed room
noise it hallucinates a short plausible token instead. The runaway conversation
loop on Aug 10 2026 was driven by an empty room transcribing as "over." three
times. `is_noise_transcript` in parsing.py matches a frozenset of known
hallucinations against the whole normalized transcript, never as a substring.

ElevenLabs specific:
- Emma voice catalog deprecates Dec 31 2026, save to "My Voices" before then
- Python's `stdin.write()` to the aplay subprocess buffers up to 64KB by default.
  ALWAYS call `flush()` after every write or you get ~1.5s phantom latency.
- aplay's ALSA buffer holds ~185ms of audio after writing stops (relevant for
  future barge in support)
- v3 stability above 0.7 makes it ignore audio tags
- v3 has no WebSocket support and no `use_speaker_boost`
- `optimize_streaming_latency` is deprecated in 2026, do not use

## Things to Never Commit

API keys (.env, ~/.bashrc), voiceprint (.npy), enrollment data (.npz), database
(.db), recording archive, Whisper weights (.bin), wake word model (.onnx),
compiled Whisper binaries, /build temp files, ~/.cloudflared/*.json. All
gitignored. Always run `git status` and confirm before `git add .`.
