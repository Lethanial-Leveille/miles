# M.I.L.E.S. — Modular Intelligent Learning & Execution System

> **Precedence rule.** This file describes the repo. The repo is the authority.
> If anything here conflicts with source, **the source wins**, and whoever finds
> the conflict fixes this file in the same session. Every config value below was
> read out of `src/config.py` on Sep 13 2026.
>
> Read `docs/SESSION_START.md` before proposing changes. It carries the preflight
> that catches drift and the decision log that explains why things are the way
> they are.
>
> **This file is the index, not the archive.** It holds what is true on every
> session regardless of task: status, layout, live values, and the rules of the
> road. The measurements, tuning history and root cause writeups live in the
> topic docs indexed below, and they are where a claim gets explained. Do not
> move that material back here.

## Current Status

v0.7.1 shipped. Unreleased work has landed on top of it since (latency
instrumentation, webrtcvad endpointing, recording archive, dismiss action,
clock injection). All three systemd services enabled and active. Tunnel live at
miles.lethanial.com, dashboard managed.

DO NOT modify existing services unless explicitly asked. They are running in
production.
DO NOT edit ~/.cloudflared/config.yml. Tunnel config is dashboard managed.
DO NOT touch the Swift companion app (separate repo at
github.com/Lethanial-Leveille/miles-app).

## What This Project Is

A personal AI voice assistant running on a Raspberry Pi 5 (8GB). The voice
personality is named Nova. M.I.L.E.S. is the system name on the resume and GitHub.

Built by Lethanial Leveille, CpE student at University of Florida, Class of 2029.
Primary portfolio project targeting embedded/firmware and full stack hardware to
cloud engineering roles.

The companion iOS app (Nova) is a separate repo at
github.com/Lethanial-Leveille/miles-app.

## Hardware

Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite 64 bit. Razer Seiren V3 Mini
USB microphone, found dynamically by name in PyAudio rather than hardcoded to a
card number, since ALSA card numbers shift on reboot. Amazon Basics USB speakers
via QianLink USB to 3.5mm adapter, also resolved by name at runtime in `tts.py`
using `SPEAKER_NAME_HINT` rather than a fixed device string. Development machine
is a MacBook connected via VS Code Remote SSH at theycallmelee@miles.local, local
network only. Remote SSH via Cloudflare Tunnel is planned and not yet configured.

Mic gain is tuned to `EXPECTED_MIC_GAIN` and checked at startup. The mixer card
and the mic itself are both resolved **by name**, never by ALSA card number,
because those shift on reboot. Why that matters, and the months of false drift
warnings it cost before Sep 6 2026, is in
[docs/AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md#microphone-and-gain).

## Documentation

| Doc | Read it when |
|---|---|
| [docs/SESSION_START.md](docs/SESSION_START.md) | **Every session, first.** Preflight commands, drift rules, and the decision log |
| [docs/BACKEND_TODO.md](docs/BACKEND_TODO.md) | Picking up deferred work, or before proposing anything hardware blocked |
| [docs/AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md) | Touching the mic, gain, wake word, VAD, Whisper, or speaker verification |
| [docs/BRAIN.md](docs/BRAIN.md) | Touching the prompt, the model, streaming, tools, channels, or local intent |
| [docs/VOICE_OUTPUT.md](docs/VOICE_OUTPUT.md) | Touching TTS, voice settings, the phrase bank, or pronunciation |
| [docs/LATENCY.md](docs/LATENCY.md) | Before quoting any latency number, or changing anything for speed |
| [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) | Touching systemd, the tunnel, the API, env vars, or reminders |
| [docs/INCIDENTS.md](docs/INCIDENTS.md) | Something broke, or you are about to change a guard that has never fired |

## Repo Structure

```
~/miles/
├── src/                          # Runtime. The three services import from here
│   ├── config.py              # Constants, paths, thresholds, env vars
│   ├── prompts.py             # System prompt assembly
│   ├── database.py            # SQLite operations + schema migrations
│   ├── parsing.py             # Memory tag extraction, noise transcripts
│   ├── actions.py             # Weather, timer, reminder handlers, reminder poller
│   ├── alerts.py              # Pending alert queue and fold into the next turn
│   ├── audio.py               # Mic, wake word, VAD, Whisper, Resemblyzer
│   ├── audio_segments.py      # Whisper window by clip length; cutting long recordings
│   ├── wake_listener.py       # hears the wake word during a recording
│   ├── speaker_encoder.py     # Encoder abstraction behind verify_voice
│   ├── embeddings.py          # Sentence embeddings for retrieval and intent
│   ├── tts.py                 # ElevenLabs synthesis, speaker resolution, aplay
│   ├── phrasebank.py          # Pre rendered Victoria, played from disk, no network
│   ├── netcheck.py            # Why a network call failed, answered locally
│   ├── local_intent.py        # Answer timers and dismiss without Claude
│   ├── stream_router.py       # Delta buffering, sentence flush
│   ├── tools.py               # Tool registry, @tool decorator, schema generation
│   ├── memory_tool.py         # remember tool, with supersede and expiry
│   ├── system_state.py        # get_system_state tool: uptime, temp, latency, commit
│   ├── tier_tool.py           # lower_access tool: demotion by voice, never escalation
│   ├── calendar_tools.py      # Google Calendar reads, freebusy, confirmed create/edit/delete
│   ├── oura_tools.py          # Oura readiness, sleep, heart rate, activity
│   ├── pending_action.py      # confirm_pending_action: outside writes wait a turn
│   ├── timing.py              # Per turn latency instrumentation
│   ├── brain.py               # ask_nova orchestrator
│   ├── auth.py                # JWT + bcrypt password hashing
│   ├── server.py              # FastAPI REST + WebSocket
│   ├── voice_main.py          # Audio loop entry point
│   ├── enroll.py              # Voice enrollment. Stays here: the suite imports it
│   └── tests/                 # pytest suite (751 passing, 6 skipped, Sep 16 2026)
├── docs/
│   ├── SESSION_START.md       # Preflight, drift rules, decision log
│   ├── BACKEND_TODO.md        # Deferred work, written to be picked up cold
│   ├── AUDIO_PIPELINE.md      # Mic to transcript: gain, wake, VAD, Whisper, verify
│   ├── BRAIN.md               # Prompt, model, streaming, channels, local intent
│   ├── VOICE_OUTPUT.md        # Voice settings, phrase bank, pronunciation
│   ├── LATENCY.md             # The turn budget and every timing measurement
│   ├── INFRASTRUCTURE.md      # Services, tunnel, API, env vars, reminders
│   └── INCIDENTS.md           # What broke, dated, with the evidence
├── scripts/                      # Operator tools, run by hand from the repo root,
│                              # or on a timer. Nothing in src/ imports any of these
│   ├── memory.py              # list/fix/chain/temporary on stored memories
│   ├── retrieval.py           # review and label what retrieval returned
│   ├── people.py              # people, tiers, birthdays
│   ├── voiceprint.py          # improve the voiceprint from real use
│   ├── encoder_bench.py       # compare speaker encoders on archived clips
│   ├── pronounce.py           # pronunciation aliases, live with no restart
│   ├── render_phrases.py      # render/trim the phrase bank, run while online
│   ├── label_transcripts.py   # hand labelled truth + word error rate per STT
│   ├── label_speakers.py      # label who is speaking in the eval clips, by ear
│   ├── label_wake.py          # label wake hits and misses by ear, worst first
│   ├── healthcheck.py         # what broke, on a timer, not by trying to use it
│   ├── wifi_watchdog.sh       # brings wlan0 back, on a timer, when it drops
│   ├── google_auth.py         # one time Google OAuth, writes data/token.json
│   ├── oura_auth.py           # one time Oura OAuth, writes data/oura_token.json
│   ├── list_calendars.py      # every calendar the Google token sees, and if selected
│   ├── analyze_timing.py      # latency analysis over timing_log
│   ├── analyze_verification.py # speaker verification analysis
│   ├── compare_whisper.py     # whisper model comparison harness
│   ├── check_gain.py          # mic gain verification
│   ├── profile_turn.py        # times the turn stages no stopwatch covers
│   ├── seed_memories.py       # seed corpus loader
│   └── setup_auth.py          # one time password and JWT secret setup
├── assets/                    # wake_chime.wav
├── systemd/                   # every unit, versioned. Installed by copying
│                              # to /etc/systemd/system/
├── models/                    # Wake word + voiceprint + enrollment (gitignored contents)
├── data/                      # SQLite database, seed corpus, recordings (gitignored)
├── build/                     # Temp WAV files (gitignored)
├── whisper.cpp/               # Compiled from source (gitignored)
├── .env                       # Password hash, JWT secret, API keys (gitignored)
├── README.md                  # Public facing project description
└── CLAUDE.md
```

## Production Commands

All three services are enabled and start on boot. `miles-voice` runs the room mic
pipeline, `miles-server` runs uvicorn on port 8000, `miles-tunnel` runs
cloudflared. Layout and routing are in
[docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md).

```bash
sudo systemctl status miles-voice miles-server miles-tunnel
sudo systemctl restart miles-voice
journalctl -u miles-voice -f
journalctl -u miles-voice -n 50
```

Claude Code sessions die on SSH disconnect or Pi reboot. Always run inside tmux:

```bash
tmux new -s miles
# Ctrl+B then D                # detach
tmux attach -t miles           # reattach
```

## Key Config Values

**Read out of `src/config.py` on Sep 13 2026.** This table is the single
declaration of what is live. When a value drifts, fix it here first. The topic
docs explain *why* each value is what it is and must never carry a second copy
of the value itself.

### Audio capture — why: [AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md)
- `CHUNK = 1280` (80ms frames, required by openWakeWord)
- `RATE = 16000`, `CHANNELS = 1`
- `WAKE_THRESHOLD = 0.4`
- `VERIFY_THRESHOLD = 0.5`
- `VAD_MODE = 2` (webrtcvad)
- `VAD_PREROLL_MS = 300`, `VAD_ONSET_FRAMES = 2`
- `SILENCE_LIMIT = 1.2`
- `MAX_RECORD = 30.0`
- `TTS_FLUSH_MARGIN_MS = 250`
- `EXPECTED_MIC_GAIN = 23`
- `MIC_NAME_HINT = "Seiren"`. `MIC_MIXER_CARD` is **derived** from it at import
  by `_resolve_mixer_card`, never written literally, and is `None` on a miss so
  callers report "cannot check" rather than measuring the wrong device.
- `SPEAKER_ENCODER = "resemblyzer"`
- `MIN_VOICED_SECONDS = 4.0`, `ENROLL_RECORD_SECONDS = 10`

### Whisper — why: [AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md#whisper)
- `WHISPER_MODEL = whisper.cpp/models/ggml-base.en-q8_0.bin` (quantized locally
  from base.en with `whisper-quantize`; derived, so regenerate after a
  whisper.cpp upgrade)
- `WHISPER_INITIAL_PROMPT = None` (a decision, not an omission)
- `WHISPER_AUDIO_CTX = 1000` (20 seconds. Do not lower without rerunning the
  validation; 750 and 900 both corrupted reference speech). Used up to 15
  seconds of audio, the length it was validated for (`audio_segments.py`)
- `WHISPER_AUDIO_CTX_LONG = 1500` (Whisper's full 30 seconds, for longer clips)
- `WHISPER_SEGMENT_SECONDS = 28.0` (longer recordings are cut at a pause and
  transcribed in pieces)

### Wake capture — why: [AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md#wake-capture)
- `CAPTURE_WAKE_MISSES = True`, `WAKE_MISS_DIR = ~/miles/data/wake_misses`
- `WAKE_MISS_FLOOR = 0.15`, `WAKE_LOG_FLOOR = 0.15`
- `WAKE_MISS_MAX_FILES = 400`, `WAKE_MISS_PREROLL_MS = 2500`
- `CAPTURE_WAKE_HITS = True`, `WAKE_HIT_DIR = ~/miles/data/wake_hits`
- `WAKE_HIT_MAX_FILES = 300`

### Speculative transcription — why:
[AUDIO_PIPELINE.md](docs/AUDIO_PIPELINE.md#speculative-transcription)
- `SPECULATIVE_TRANSCRIBE = True`, `SPECULATIVE_SILENCE_MS = 450`,
  `SPECULATIVE_THREADS = 3`

### Model and prompt — why: [BRAIN.md](docs/BRAIN.md)
- `MODEL_A = "claude-haiku-4-5"` — **this is the production model**
- `MODEL_B = "claude-sonnet-4-5-20250929"`
- `MODEL_AB_TEST = False`
- `PROMPT_CACHING = True`
- `HISTORY_ASSISTANT_WORDS = 30`

### Conversation loop — why: [BRAIN.md](docs/BRAIN.md#local-intent)
- `MAX_FOLLOWUP_TURNS = 6`, `FOLLOWUP_TIMEOUT = 3.0`
- `INTENT_SIMILARITY_THRESHOLD = 0.55`, `MAX_DISMISS_WORDS = 6`
- Exit is intent based, not a phrase list. Dismiss is a registered tool.

### TTS and phrase bank — why: [VOICE_OUTPUT.md](docs/VOICE_OUTPUT.md)
- `TTS_VOICE_ID` = Victoria (`qSeXEcewz7tA0Q0qk9fH`). The single place a voice
  is named; no call site references one.
- `DEFAULT_TTS_MODEL = "eleven_v3"` (chosen by ear, Sep 13 2026)
- `EXPRESSIVE_TTS_MODEL = "eleven_v3"` (HTTP only, no WebSocket)
- `TTS_OUTPUT_FORMAT = "pcm_22050"` (raw S16_LE mono, piped to aplay)
- `TTS_PHONEME_TAGS = True`
- `SPEAKER_NAME_HINT = "AB13X"` (resolved by name at runtime in tts.py)
- `PHRASE_DIR = ~/miles/data/phrases`, gitignored. `PHRASES` in `phrasebank.py`
  is the versioned source of truth; the WAVs are derived from it.
- `ACK_SPOKEN_CHANCE = 0.75`
- Voice settings: `stability=1.0`, `similarity_boost=0.75`, `style=0.00`,
  `speed=1.00` (v3 ignores it), `use_speaker_boost` unset. **Changing any of
  these means re rendering the whole phrase bank.**

### Tools and permissions — why: [BRAIN.md](docs/BRAIN.md#tools-and-the-permission-gate)
- `PERMISSION_TIERS` (in `tools.py`): READ `genin`, CONTROL `genin`, WRITE
  `chunin`, EXTERNAL_WRITE `hokage`. A tool's `min_tier` can only raise its floor.
- Raised by `min_tier`: `remember` to `jonin`; `lower_access`, every calendar
  tool, every Oura tool, `list_pending_memories` and `review_pending_memory` to
  `hokage`.
- `CONFIRM_WINDOW_S = 120` (in `pending_action.py`, not `config.py`)

### Storage and other
- `DB_PATH = ~/miles/data/miles.db`
- `ARCHIVE_RECORDINGS = True`, `ARCHIVE_DIR = ~/miles/data/recordings`,
  `ARCHIVE_MAX_FILES = 600` (~150MB, a few weeks of normal use, oldest pruned first)
- `DEFAULT_LOCATION = "Gainesville"`
- `REMINDER_POLL_S = 20`, `REMINDER_LATE_S = 3600` (both in `actions.py`, not
  `config.py`)

Recordings are of a real person in a real room. Treat them accordingly.

## Tech Stack

Wake word: openWakeWord (hey_nova.onnx).
VAD: webrtcvad, mode 2.
STT: whisper.cpp (base.en, greedy decoding, NEON ARM optimizations, capped
audio context).
LLM: Claude API, `claude-haiku-4-5`, streaming, prompt caching on the system prompt.
TTS: ElevenLabs (Victoria, eleven_v3), pcm_22050 piped to aplay.
Voice auth: Resemblyzer (256 dim cosine similarity).
Memory: SQLite WAL mode, schema migrations to version 21.
Weather: OpenWeatherMap.
Calendar: Google Calendar API v3, OAuth, dateparser for spoken times.
Health: Oura API v2, OAuth.
Backend: FastAPI + uvicorn + python-jose + passlib (bcrypt) + python-dotenv.
Tunnel: cloudflared (dashboard managed).
Process management: systemd.
Language: Python 3.13.5.

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
- Native tool use: the registry in `tools.py` and the tool loop in `brain.py`,
  replacing bracket action tags
- Permission gate enforced in the executor, one tier per turn
- Google Calendar and Oura tools; calendar create, edit and delete confirmed
  on the next turn
- A text turn is silent, answers in numerals, and can be streamed to the app
  over `/chat/stream`

## What Is Next

- SSH via Cloudflare Tunnel: ssh.lethanial.com + Cloudflare Access policy
- v0.8+: Morning briefings (Fall 2026), Mac control, HealthKit, EventKit,
  MusicKit, WeatherKit migration, barge in detection, parallel processing

## Coding Preferences

No hyphens in any written output, ever. Numbers spelled as words in Nova's
spoken responses. Nova calls me "Lethanial" by default. "Lee" is allowed but
rare, and only
when something has genuinely gone well; it is a condition, not a frequency. Do
not use
APScheduler yet. Python style: flat is better than nested, clarity over
cleverness. Comments explain WHY not WHAT.

## Critical Learning Constraint

I am a CpE student building this to learn, not just to ship. When making code
changes:

1. Explain what we are about to do before writing code
2. Make small, logical changes one at a time
3. When introducing a new concept, explain what it does and why before using it
4. Do not refactor things outside the current task scope
5. Do not use hyphens in anything you write for me
6. If I push back, defend your position if you believe it is right. I value
honesty above everything.

## Session Persistence

Sessions run in tmux; the commands are under Production Commands above.

Before ending any session, work the end of session checklist in
`docs/SESSION_START.md`. That is what keeps this file and the topic docs true.

## Things to Never Commit

API keys (.env, ~/.bashrc), voiceprint (.npy), enrollment data (.npz), database
(.db), recording archive, Whisper weights (.bin), wake word model (.onnx),
compiled Whisper binaries, /build temp files, ~/.cloudflared/*.json, OAuth
tokens (data/token.json, data/oura_token.json) and the Google client
(credentials.json). All gitignored. Always run `git status` and confirm before `git add .`.
