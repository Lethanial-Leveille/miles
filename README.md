# M.I.L.E.S.

**Modular Intelligent Learning and Execution System**

A voice assistant running on a Raspberry Pi 5. Say "Hey Nova" and it listens, checks that the voice belongs to its owner, answers through Claude, and speaks back through ElevenLabs. It reads its owner's Google Calendar and Oura ring, keeps a long term memory, sets timers and reminders, and can create, move or delete calendar events after asking out loud first. A native Swift companion app talks to the same backend from anywhere.

It is an exploration of full stack systems engineering: real time audio on embedded Linux, on device speech and speaker recognition, an LLM agent loop with a permission model enforced in code, and an API reachable from the internet.

---

## Demo

Watch the full system demo (voice and companion app): [linkedin.com/in/lethanial-lee-leveille](https://www.linkedin.com/in/lethanial-lee-leveille)

Companion app repo: [github.com/Lethanial-Leveille/miles-app](https://github.com/Lethanial-Leveille/miles-app)

---

## What It Does

Say "Hey Nova." Nova acknowledges, records until you stop talking, transcribes on the Pi, checks your voice, and answers out loud. Follow ups do not need the wake word.

* **"How did I sleep?"** reads the Oura ring and says what the night means, not just the numbers.
* **"What does my week look like?"** reads Google Calendar from now on, and keeps club events apart from real commitments.
* **"Move LeetCode to 4pm."** Nova asks "Move LeetCode session tomorrow from 10 AM to 4 PM?" and changes nothing until you say yes.
* **"Set a timer for fifteen minutes."** is recognised and answered on the Pi without calling Claude at all.
* **"Remind me to push my code at nine pm."**

---

## Architecture

```
Raspberry Pi 5 (8GB)

  USB mic
    → openWakeWord (hey_nova.onnx)
    → webrtcvad endpointing, with speculative transcription during the silence wait
    → whisper.cpp (base.en, q8, greedy)
    → Resemblyzer speaker verification
    → local intent: timers, the time, cancel, dismiss, answered with no LLM call
    → Claude Haiku 4.5, streaming, prompt cached, native tool use
        → tool registry → permission gate → next turn confirmation for outside writes
    → sentence router → ElevenLabs (Victoria, eleven_v3) → aplay → speaker

  SQLite (WAL): memory, conversation history, reminders, tool calls, per turn timing
  FastAPI + JWT, published through a Cloudflare Tunnel at miles.lethanial.com

External: Anthropic Claude · ElevenLabs · Google Calendar · Oura · OpenWeatherMap
```

---

## Pipeline

1. **Wake word.** openWakeWord runs a custom `hey_nova.onnx` model continuously on the CPU.
2. **Acknowledge.** A short spoken acknowledgement from a pre rendered phrase bank, or a chime. The phrase bank also lets Nova tell you what failed when the network is down.
3. **Endpointing.** webrtcvad decides when you have stopped talking, with a short pre roll so a soft first consonant is not cut off.
4. **Speculative transcription.** Whisper starts during the silence wait, so the transcript is often ready the moment the endpoint fires.
5. **Speech to text.** whisper.cpp compiled from source with NEON optimizations, base.en quantized to q8, greedy decoding, capped audio context. Entirely on device.
6. **Speaker verification.** Resemblyzer scores a 256 dimensional voice embedding against the enrolled voiceprint by cosine similarity.
7. **Local intent.** Timers, the time, cancelling a reminder and ending the conversation are recognised on the Pi with sentence embeddings and answered without an LLM call.
8. **LLM.** Claude Haiku 4.5, streaming, with prompt caching on the system prompt and native tool use.
9. **Tools.** Nineteen registered tools. Every call passes a permission gate in code before it runs, and anything that writes to a system outside the Pi waits for a spoken yes on the next turn.
10. **Speech.** Each complete sentence is streamed to ElevenLabs as raw PCM and piped straight into aplay.
11. **Follow ups.** Nova keeps listening for a short window after answering.

---

## Tools and the Permission Model

The principle underneath all of it: **deterministic code does the math and enforces the rules; the model does judgment.**

* **The model proposes, code decides.** A tool call is a request. Between the request and the function, the executor checks the speaker's tier against the tool's permission. A refusal goes back to the model as a result, so Nova says no out loud instead of going silent. A rule in a prompt is a suggestion to a probabilistic system; a conditional in the executor is a guarantee.
* **Confirmation is enforced by turn order.** Creating, moving or deleting an event only stages it. The write runs only when confirmed on the very next turn, within two minutes, and the confirmation takes no event details, so it can only run what was read back.
* **Time is resolved in code.** Spoken times like "monday at 3pm" are resolved against the Pi's clock, and free time is computed from merged busy blocks, because that is arithmetic.
* **Results carry their units.** An early sleep tool returned contributor scores under names like `total_sleep`, and Nova reported a hundred out of a hundred as an hour and forty minutes of sleep. Every field now names what it is.

---

## Tech Stack

**Hardware**
* Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite, 64 bit
* Razer Seiren V3 Mini USB microphone, found by name rather than by ALSA card number
* USB speakers through a 3.5mm adapter

**Languages**
* Python 3.13 (pipeline, backend, orchestration)
* Swift and SwiftUI (companion app, separate repo)

**Audio and ML**
* openWakeWord (custom hey_nova.onnx)
* webrtcvad (endpointing)
* whisper.cpp (on device speech to text)
* Resemblyzer (speaker verification)
* Sentence embeddings (memory retrieval and local intent)
* PyAudio for input, aplay and ALSA for output

**Backend**
* FastAPI (REST and WebSocket)
* JWT authentication (HS256) with bcrypt password hashing
* SQLite in WAL mode
* Cloudflare Tunnel, dashboard managed
* systemd (voice loop, server, tunnel, and a health check timer)

**APIs**
* Anthropic Claude Haiku 4.5
* ElevenLabs, Victoria voice on eleven_v3
* Google Calendar API v3
* Oura API v2
* OpenWeatherMap

---

## Key Engineering Decisions

**Voice chosen by listening, not by reasoning.** When Nova sounded like she was reading rather than speaking, each suspect was rendered side by side on the same real reply and judged by ear: splitting replies into sentences, giving each sentence its neighbours as context, stability, speed, four other voices, and four synthesis models. Only the model made an audible difference.

**Endpointing moved from amplitude to webrtcvad.** The original energy threshold was never crossed by real speech at the measured capture level, so every recording ended on a timeout and long commands were cut off mid sentence. webrtcvad fixed the truncation and, with the silence window retuned, brought median perceived latency from about 8.1 seconds to 4.9.

**Response length cut by trimming history, not by instruction.** Nova's own past answers anchor her length far more strongly than any prompt line, so past replies are trimmed before being sent back as context.

**A failed turn never kills the process.** The voice loop has its own exception boundary, and a local network check picks what to say, so "I've lost wifi" is never spoken when the real problem is the API.

**The stdin.flush() discovery.** Writing audio chunks to aplay without flushing let Python buffer up to 64KB, adding about 1.5 seconds of phantom latency even with streaming on.

---

## API Endpoints

Live at `https://miles.lethanial.com`.

| Endpoint | Method | Auth |
|----------|--------|------|
| `/auth/login` | POST | No |
| `/auth/refresh` | POST | Yes |
| `/chat` | POST | Yes |
| `/memories` | GET | Yes |
| `/memories/pending` | GET | Yes |
| `/memories/{memory_id}/approve` | POST | Yes |
| `/memories/{memory_id}` | DELETE | Yes |
| `/history` | GET | Yes |
| `/status` | GET | Yes |
| `/ws` | WebSocket | Yes, token checked in the socket |
| `/docs` | GET | No, FastAPI's generated Swagger UI |

---

## Measured Performance

Measured on Sep 13 2026 from the per turn `timing_log`. Every stage and its history is in [docs/LATENCY.md](docs/LATENCY.md).

| Metric | Value |
|--------|-------|
| Median perceived latency, last 30 turns | 3.9 seconds, from the end of speech to first audio, including the silence wait that decides you have finished |
| ElevenLabs eleven_v3 time to first byte | 647ms median over 12 live turns |
| Registered tools | 19 |
| Test suite | 604 passing |

---

## Project Status

| Version | Feature | Status |
|---------|---------|--------|
| v0.1 | Wake word and speech to text | Complete |
| v0.2 | Claude and text to speech | Complete |
| v0.3 | Speaker verification | Complete |
| v0.4 | Persistent memory and conversation history | Complete |
| v0.5 | Weather, timers, reminders | Complete |
| v0.6 | Multi turn conversations | Complete |
| v0.7 | FastAPI backend, JWT, Cloudflare Tunnel, systemd, Nova iOS app | Complete |
| v0.7.1 | ElevenLabs streaming, sentence router, wake chime | Complete |
| Unreleased | Native tool use with a permission gate, Google Calendar and Oura, spoken confirmation, local intent, speculative transcription, eleven_v3 voice | Landed |
| Next | SSH through the tunnel, morning briefings, barge in, Apple ecosystem integrations | Planned |

---

## Repository Structure

```
miles/
├── src/            Runtime. The three services import from here
│   └── tests/      pytest suite
├── scripts/        Operator tools run by hand: pronunciation, phrase bank, labelling, analysis
├── docs/           Engineering docs. Start with docs/SESSION_START.md
├── systemd/        Health check units
└── assets/         Wake chime
```

Secrets and personal data are gitignored: API keys, the voiceprint and enrollment audio, the database and recording archive, OAuth tokens, and Whisper weights.

---

## Companion App

The Nova iOS companion app is a separate repo:
[github.com/Lethanial-Leveille/miles-app](https://github.com/Lethanial-Leveille/miles-app)

Native SwiftUI with no third party dependencies, Face ID, Keychain JWT storage, and speech input through SFSpeechRecognizer. The room microphone and the app talk to the same FastAPI backend.

---

## About

Built by **Lethanial Leveille**, Computer Engineering student at the University of Florida, Class of 2029. Targeting embedded, firmware, and full stack hardware to cloud engineering roles.

[LinkedIn](https://www.linkedin.com/in/lethanial-lee-leveille/) · [GitHub](https://github.com/Lethanial-Leveille)
