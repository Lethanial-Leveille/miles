# M.I.L.E.S.

**Modular Intelligent Learning and Execution System**

A voice activated personal AI assistant running on a Raspberry Pi 5, featuring custom wake word detection, on device speech recognition, speaker verification, persistent memory, Claude API streaming, ElevenLabs TTS, and a native Swift companion app accessible from anywhere in the world.

Built from the ground up as an exploration of full stack systems engineering, spanning embedded hardware, real time audio processing, machine learning inference, cloud APIs, and a native iOS mobile app.

---

## Demo

Watch the full system demo (voice + companion app): [linkedin.com/in/lethanial-lee-leveille](https://www.linkedin.com/in/lethanial-lee-leveille)

Companion app repo: [github.com/Lethanial-Leveille/miles-app](https://github.com/Lethanial-Leveille/miles-app)

---

## What It Does

Say "Hey Nova" and the Pi wakes up, plays a 320ms audio chime, listens to your command, verifies your voice matches the registered user, streams the request through Claude, and speaks a natural response back through the speaker via ElevenLabs Emma. Nova handles multi turn conversations without requiring the wake word between follow ups, remembers facts across sessions, and manages timers and reminders.

You can also text Nova from anywhere in the world through the native iOS companion app, which shares the same backend brain on the Pi.

Sample interactions:

* "Hey Nova, what's the weather in Miami?"
* "Set a timer for fifteen minutes."
* "Remind me to push my code at nine pm."
* "I just finished episode thirty of Bleach." *(stored as a memory for future conversations)*

---

## Architecture

```
+------------------------------------------------------------------+
|                      Raspberry Pi 5 (8GB)                        |
|                                                                  |
|  +----------+   +----------+   +----------+   +--------------+  |
|  |  USB Mic |-->| Wake Word|-->|  Whisper |-->|  Resemblyzer |  |
|  |  (Razer) |   | (ONNX)   |   |  .cpp    |   |  (Speaker    |  |
|  |          |   |          |   |          |   |   Verify)    |  |
|  +----------+   +----------+   +----------+   +------+-------+  |
|                                                       |          |
|  +----------+   +----------+   +----------+   +------v-------+  |
|  |  Speaker |<--|ElevenLabs|<--|  Claude  |<--|   SQLite     |  |
|  |  (aplay) |   |   TTS    |   |Streaming |   |  (Memory +   |  |
|  |          |   |  (Emma)  |   |   API    |   |   History)   |  |
|  +----------+   +----------+   +----------+   +--------------+  |
|                                                                  |
|  +------------------------------------------------------------+  |
|  | FastAPI + JWT -- Cloudflare Tunnel -- miles.lethanial.com  |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
                              |
                    +---------+----------+
                    |  External Services |
                    |  * Anthropic Claude|
                    |  * ElevenLabs Emma |
                    |  * OpenWeatherMap  |
                    +--------------------+
```

---

## Pipeline

1. **Wake Word Detection** — openWakeWord runs a custom ONNX model (hey_nova.onnx, 90.4% recall) continuously on CPU, listening for "Hey Nova."
2. **Wake Chime** — A 320ms two note ascending tone (C5 to G5) plays immediately on detection via aplay, before recording begins. Gives instant feedback.
3. **Voice Activity Detection** — Energy based VAD records until silence, with dynamic duration.
4. **Speech to Text** — Whisper.cpp compiled from source with NEON ARM optimizations, base.en model, greedy decoding. On device, no cloud dependency.
5. **Speaker Verification** — Resemblyzer generates a 256 dimensional voice embedding and compares against the registered voiceprint via cosine similarity (threshold: 0.65). Unauthorized voices are rejected.
6. **LLM Streaming** — Claude claude-sonnet-4-5 receives the command with injected memories and conversation history. Uses stop_sequences for action tag detection. StreamRouter buffers the first 50 characters to guard against false bracket matches before flushing to TTS.
7. **Action Execution** — Action tags trigger a second Claude call with real world data (weather, timer state, etc.) so responses are delivered naturally with context.
8. **TTS Streaming** — ElevenLabs Emma (Flash v2.5, pcm_22050) streams audio chunks directly to aplay via stdin. stdin.flush() after every chunk eliminates 1.5s phantom buffer latency. Time to first audio: ~520ms after VAD endpoint.
9. **Multi Turn** — Nova stays active post response, listening for follow ups without requiring the wake word.

---

## Tech Stack

**Hardware**
* Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite 64-bit
* Razer Seiren V3 Mini USB condenser microphone
* USB speakers via 3.5mm audio adapter

**Languages**
* Python 3.13 (pipeline, backend, orchestration)
* Swift / SwiftUI (companion app, separate repo)

**ML / Audio**
* openWakeWord (custom hey_nova.onnx, ONNX runtime)
* Whisper.cpp (on device STT, NEON ARM optimized)
* Resemblyzer (speaker verification via voice embeddings)
* PyAudio (mic input)
* aplay / ALSA (audio output)

**Backend**
* FastAPI (REST + WebSocket server)
* JWT authentication (HS256, 7 day tokens)
* SQLite with WAL mode (memory, history, reminders)
* Cloudflare Tunnel (secure remote access at miles.lethanial.com)
* systemd (three services on boot: voice loop, server, tunnel)

**APIs**
* Anthropic Claude claude-sonnet-4-5 (LLM, streaming)
* ElevenLabs Emma Flash v2.5 (TTS, streaming PCM)
* OpenWeatherMap (weather data)

---

## Key Engineering Decisions

**Sub 520ms time to first audio**
Achieved through greedy Whisper decoding over beam search, streaming TTS with direct aplay pipe instead of file writes, and a 50 character StreamRouter lookahead that begins TTS before the full Claude response completes.

**Hybrid stop_sequences + StreamRouter pattern**
Claude streaming uses stop_sequences=["[ACTION:"] to halt generation at action tags, while StreamRouter handles false positive brackets in prose via lookahead buffering. Both problems need different solutions and were solved independently.

**LLM driven intent classification**
Intent detection is delegated entirely to Claude via prompt engineering. The system understands "what's it like outside" as a weather request without hardcoded keyword matching. New intents require only a system prompt update.

**On device voice biometrics**
Speaker verification runs locally via Resemblyzer. A 256 dimensional embedding is compared via cosine similarity on every command, including follow up turns in multi turn conversations.

**Dashboard managed Cloudflare Tunnel**
Migrated from local YAML config to Cloudflare Zero Trust dashboard management. No local config files to maintain, no port forwarding, no home network exposure.

**ElevenLabs stdin.flush() discovery**
Omitting flush() after every audio chunk write caused a 1.5 second phantom buffer delay even with streaming enabled. Flushing after each chunk reduced TTFA by approximately 1.5 seconds with no other changes.

---

## API Endpoints

The FastAPI server is live at `https://miles.lethanial.com`.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/auth/login` | POST | No | Issue JWT |
| `/auth/refresh` | POST | Yes | Refresh token |
| `/chat` | POST | Yes | Send message, get Nova response |
| `/memories` | GET | Yes | Fetch persistent memory bank |
| `/memories/{id}` | DELETE | Yes | Remove a memory |
| `/history` | GET | Yes | Paginated conversation log |
| `/status` | GET | Yes | Backend health and version |
| `/ws` | WebSocket | Yes | Real time connection (auth via first message) |
| `/docs` | GET | No | Swagger UI |

---

## Performance

| Metric | Value |
|--------|-------|
| Wake word recall | 90.4% |
| False activations | ~2.7 per hour |
| Whisper transcription (base.en, greedy) | 0.8 to 1.5 seconds |
| Time to first audio after VAD endpoint | ~520ms |
| Voice verification cosine similarity (registered) | 0.65 to 0.85 |
| ElevenLabs Flash v2.5 model TTFB | ~50ms |

---

## Project Status

| Version | Feature | Status |
|---------|---------|--------|
| v0.1 | Audio pipeline (wake word + STT) | Complete |
| v0.2 | LLM + TTS integration | Complete |
| v0.3 | Speaker verification + VAD | Complete |
| v0.4 | Persistent memory + conversation history | Complete |
| v0.5 | Action system (weather, timers, reminders) | Complete |
| v0.6 | Multi turn conversations | Complete |
| v0.7 | FastAPI backend + Cloudflare Tunnel + JWT | Complete |
| v0.7.1 | ElevenLabs migration + StreamRouter + wake chime | Complete |
| v0.8 | Nova iOS companion app shipped | Complete |
| v0.9+ | ESP32 satellite mics, Apple ecosystem, GPU satellite compute | Planned |

---

## Repository Structure

```
miles/
├── src/
│   ├── voice_main.py        Main voice pipeline entry point
│   ├── config.py            Environment config and constants
│   ├── brain.py             Claude API integration and streaming
│   ├── stream_router.py     50-char lookahead TTS buffer router
│   ├── audio.py             Mic capture, playback, VAD
│   ├── server.py            FastAPI app, all endpoints
│   ├── auth.py              JWT issue and verification
│   ├── database.py          SQLite memory and history operations
│   ├── parsing.py           Action tag extraction and dispatch
│   ├── actions.py           Action handlers (weather, timers, etc.)
│   ├── prompts.py           System prompt definitions
│   └── enroll.py            Voice enrollment utility
├── assets/
│   └── wake_chime.wav       320ms two-note ascending chime
├── .gitignore
└── README.md
```

Sensitive files (`.env` with API keys, `voiceprint.npy`, `miles.db`, Whisper binaries) are gitignored.

---

## Companion App

The Nova iOS companion app is a separate repo:
[github.com/Lethanial-Leveille/miles-app](https://github.com/Lethanial-Leveille/miles-app)

Native SwiftUI, zero third party dependencies, FaceID auth, Keychain JWT storage, speech input via SFSpeechRecognizer. Both the room mic and the app talk to the same FastAPI backend.

---

## About

Built by **Lethanial Leveille**, Computer Engineering student at the University of Florida, Class of 2029. Targeting embedded and firmware engineering.

[LinkedIn](https://www.linkedin.com/in/lethanial-lee-leveille/) · [GitHub](https://github.com/Lethanial-Leveille)
