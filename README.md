# M.I.L.E.S.

**Modular Intelligent Learning and Execution System**

A voice activated personal AI assistant running on a Raspberry Pi 5, featuring custom wake word detection, on device speech recognition, speaker verification, persistent memory, and an extensible action system for external integrations.

Built from the ground up as an exploration of full stack systems engineering, spanning embedded hardware, real time audio processing, machine learning inference, cloud APIs, and (in progress) a native Swift companion app.

---

## Demo

*Video demo coming soon*

---

## What It Does

Say "Hey Nova" and the Pi wakes up, listens to your command, verifies your voice matches the registered user, processes the request through an LLM, and speaks a response back through the speaker. Nova handles multi turn conversations without requiring the wake word between follow ups, remembers facts across sessions, manages timers and reminders, and fetches real world data like weather.

Sample interactions:
- "Hey Nova, what's the weather in Miami?"
- "Set a timer for fifteen minutes."
- "Remind me to push my code at nine pm."
- "I just finished episode thirty of Bleach." *(silently stored as a memory for future conversations)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Raspberry Pi 5 (8GB)                        │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  USB Mic │──▶│ Wake Word│──▶│  Whisper │──▶│  Resemblyzer │  │
│  │          │   │ (ONNX)   │   │  (C++)   │   │  (Speaker    │  │
│  │          │   │          │   │          │   │   Verify)    │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬───────┘  │
│                                                       │          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────▼───────┐  │
│  │  Speaker │◀──│Fish Audio│◀──│  Claude  │◀──│   SQLite     │  │
│  │          │   │   TTS    │   │   API    │   │  (Memory +   │  │
│  │          │   │          │   │          │   │   History)   │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │  External Services │
                    │  • OpenWeatherMap  │
                    │  • Claude (LLM)    │
                    │  • Fish Audio (TTS)│
                    └────────────────────┘
```

---

## Pipeline

1. **Wake Word Detection** — openWakeWord runs a custom ONNX model continuously on the CPU, listening for "Hey Nova" (90.4% recall, 2.7 false activations per hour).
2. **Voice Activity Detection** — Energy based VAD records until 2.5 seconds of silence, with dynamic duration (0.5s minimum, 15s maximum) instead of a fixed window.
3. **Speech to Text** — Whisper.cpp compiled from source with NEON ARM optimizations, using the base.en model with greedy decoding. Sub one second latency for typical voice commands.
4. **Speaker Verification** — Resemblyzer generates a 256 dimensional voice embedding and compares it against the registered voiceprint using cosine similarity. Unauthorized voices are politely rejected.
5. **LLM Processing** — Claude Sonnet receives the transcribed command along with injected long term memories and recent conversation history. A structured tag system lets Claude request external actions (weather lookups, timers, reminders) inline.
6. **Action Execution** — Regex parser extracts action tags, dispatches to handlers, and for data fetching actions, makes a second LLM call with real world data so the response is delivered naturally.
7. **Text to Speech** — Fish Audio S2 Pro synthesizes audio with emotion tags for natural prosody. Streaming response with thread safe audio playback.
8. **Multi Turn Conversation** — After responding, Nova stays active for ten seconds, listening for follow ups without requiring the wake word again.

---

## Tech Stack

**Hardware**
- Raspberry Pi 5 (8GB)
- Razer Seiren V3 Mini USB condenser microphone
- USB speakers via 3.5mm audio adapter

**Languages**
- Python (main pipeline, orchestration)
- C++ (Whisper.cpp inference engine)
- Swift / SwiftUI (companion app, in progress)

**ML / Audio**
- openWakeWord (custom wake word, ONNX runtime)
- Whisper.cpp (on device speech recognition)
- Resemblyzer (speaker verification via voice embeddings)

**Backend / APIs**
- Anthropic Claude API (LLM)
- Fish Audio (text to speech)
- OpenWeatherMap (transitioning to Apple WeatherKit in v0.7)

**Data**
- SQLite with WAL mode (memory, conversation history, reminders)

**In Progress (v0.7)**
- FastAPI (REST + WebSocket server for cross device access)
- JWT authentication
- Cloudflare Tunnel (secure remote access)
- Swift companion app with FaceID

---

## Key Engineering Decisions

**Latency optimization: 9s → sub 1s transcription**
Switched Whisper from default beam search to greedy decoding and selected the base.en model as the right tradeoff between accuracy and speed for the Pi's CPU constraints.

**LLM driven intent classification**
Instead of regex keyword matching for memory storage and action triggering, all classification is delegated to Claude via prompt engineering. The system understands "what's it like outside" as a weather request without containing the word "weather."

**Extensible action system**
New integrations (timers, reminders, weather) require only three changes: a handler function, an elif branch in the router, and a line in the system prompt. Designed to scale to 25+ integrations over future versions.

**On device voice biometrics**
Speaker verification runs locally with no cloud dependency. A 256 dimensional voice embedding is compared via cosine similarity on every command, including follow up messages in multi turn conversations.

**Thread safety for concurrent audio**
Background timer and reminder threads use a shared lock to prevent audio collision when alerts fire during active conversations.

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
| v0.7 | Swift companion app + remote access | In Progress |
| v0.8+ | Scheduled briefings, Mac control, Apple ecosystem integrations | Planned |

---

## Performance

| Metric | Value |
|--------|-------|
| Wake word recall | 90.4% |
| Whisper transcription (base.en, greedy) | 0.8 to 1.9 seconds |
| Claude API response | 1.5 to 4.0 seconds |
| Voice verification (cosine similarity) | 0.60 to 0.85 for registered voice |
| End to end latency | 4 to 20 seconds |

---

## Repository Structure

```
miles/
├── src/
│   ├── nova.py              Main voice pipeline
│   ├── enroll.py            Voice enrollment script
│   ├── wake_test.py         Wake word testing utility
│   └── claude_test.py       Claude API connection test
├── models/
│   ├── hey_nova.onnx        Custom wake word model
│   └── voiceprint.npy       Registered voiceprint (gitignored)
├── data/
│   └── miles.db             SQLite database (gitignored)
├── .gitignore
└── README.md
```

Sensitive files (API keys, voiceprint, database, compiled Whisper binaries) are gitignored.

---

## Author

**Lethanial Leveille**
Computer Engineering, University of Florida, Class of 2029
[LinkedIn](https://www.linkedin.com/in/lethanial-lee-leveille/) · [GitHub](https://github.com/Lethanial-Leveille)
