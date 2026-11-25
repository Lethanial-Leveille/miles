# M.I.L.E.S.

**Modular Intelligent Learning and Execution System**

A voice assistant running on a Raspberry Pi 5. Say "Hey Nova" and it listens, checks that the voice belongs to its owner, answers through Claude, and speaks back through ElevenLabs. It reads its owner's Google Calendar, Oura ring, Hevy workouts and job search tracker, looks up anything current on the web, keeps a long term memory, and sets timers and reminders. It changes the calendar when asked, with a thirty minute undo, and raises the weights in his workout routines when a session earns it. Talk over it and it stops to listen. A native Swift companion app talks to the same backend from anywhere.

It is an exploration of full stack systems engineering: real time audio on embedded Linux, on device speech and speaker recognition, an LLM agent loop with a permission model enforced in code, and an API reachable from the internet.

---

## Demo

Watch the full system demo (voice and companion app): [linkedin.com/in/lethanial-lee-leveille](https://www.linkedin.com/in/lethanial-lee-leveille)

Companion app repo: [github.com/Lethanial-Leveille/miles-app](https://github.com/Lethanial-Leveille/miles-app)

---

## What It Does

Say "Hey Nova." Nova acknowledges, records until you stop talking, transcribes it, checks your voice, and answers out loud. Follow ups do not need the wake word, and you can cut in while she is talking.

* **"How did I sleep?"** reads the Oura ring and says what the night means, not just the numbers.
* **"What does my week look like?"** reads Google Calendar from now on, and keeps club events apart from real commitments.
* **"Move LeetCode to 4pm."** Nova moves it and says so. "Undo that" puts it back for the next thirty minutes.
* **"What should I hit at the gym today?"** reads the routine and his last sessions in Hevy and says whether to add a rep or add weight on each lift.
* **"Wait, go back."** said over her stops her mid sentence, and what you said is answered next.
* **"Set a timer for fifteen minutes."** is recognised and answered on the Pi without calling Claude at all.
* **"Remind me to push my code at nine pm."**

---

## Architecture

```
Raspberry Pi 5 (8GB)

  ReSpeaker XVF3800 mic array (beamforming, echo cancellation on chip)
    → openWakeWord (hey_nova.onnx)
    → webrtcvad endpointing, with speculative transcription during the silence wait
    → Deepgram Nova 3, falling back to whisper.cpp on the Pi (base.en, q8, beam 5)
    → SpeechBrain ECAPA speaker verification
    → local intent: timers, the time, cancel, dismiss, answered with no LLM call
    → Claude Haiku 4.5, streaming, prompt cached, native tool use
        → tool registry → permission gate → next turn confirmation for outside writes
    → sentence router → ElevenLabs (Victoria, eleven_v3) → aplay → speaker
        ← barge in: the wake word, or his speech over her, stops playback

  SQLite (WAL): memory, conversation history, reminders, tool calls, per turn timing
  FastAPI + JWT, published through a Cloudflare Tunnel at miles.lethanial.com

External: Anthropic Claude (with web search) · Deepgram · ElevenLabs · Google Calendar · Oura · Hevy · Prowl · Apple WeatherKit and Maps
```

---

## Pipeline

1. **Wake word.** openWakeWord runs a custom `hey_nova.onnx` model continuously on the CPU.
2. **Acknowledge.** A short spoken acknowledgement from a pre rendered phrase bank, or a chime. The phrase bank also lets Nova tell you what failed when the network is down.
3. **Endpointing.** webrtcvad decides when you have stopped talking, with a short pre roll so a soft first consonant is not cut off.
4. **Speculative transcription.** Whisper starts during the silence wait, so the transcript is often ready the moment the endpoint fires.
5. **Speech to text.** Deepgram Nova 3 first, because it hears the room better. whisper.cpp stays on the Pi as the floor: compiled from source with NEON optimizations, base.en quantized to q8, beam search, capped audio context, and used whenever Deepgram fails or is slow.
6. **Speaker verification.** SpeechBrain ECAPA scores a 192 dimensional voice embedding against the enrolled voiceprint by cosine similarity. A voice that does not match is answered as a guest, with less access.
7. **Local intent.** Timers, the time, cancelling a reminder and ending the conversation are recognised on the Pi with sentence embeddings and answered without an LLM call.
8. **LLM.** Claude Haiku 4.5, streaming, with prompt caching on the system prompt and native tool use.
9. **Tools.** Thirty four registered tools. Every call passes a permission gate in code before it runs.
10. **Speech.** Each complete sentence is streamed to ElevenLabs as raw PCM and piped straight into aplay.
11. **Barge in.** While she speaks, a second listener watches the mic. "Hey Nova" or any speech over her stops playback mid sentence, and what was said is kept and answered as the next turn. A run of sound that rises and falls with her own voice is her echo and is ignored. If the stop was only a cough or a tap, she restarts the sentence she was cut off in, and a plain "go on" finishes it on request.
12. **Follow ups.** Nova keeps listening for a short window after answering.

---

## Tools and the Permission Model

The principle underneath all of it: **deterministic code does the math and enforces the rules; the model does judgment.**

* **The model proposes, code decides.** A tool call is a request. Between the request and the function, the executor checks the speaker's tier against the tool's permission. A refusal goes back to the model as a result, so Nova says no out loud instead of going silent. A rule in a prompt is a suggestion to a probabilistic system; a conditional in the executor is a guarantee.
* **Undo where possible, confirmation where not.** Calendar changes happen at once and can be undone for thirty minutes. The few that cannot be fully undone, like deleting one occurrence of a recurring event or accepting a job, only stage the change: it runs when confirmed on the very next turn, within two minutes, and the confirmation takes no details, so it can only run what was read back.
* **Time is resolved in code.** Spoken times like "monday at 3pm" are resolved against the Pi's clock, and free time is computed from merged busy blocks, because that is arithmetic.
* **Results carry their units.** An early sleep tool returned contributor scores under names like `total_sleep`, and Nova reported a hundred out of a hundred as an hour and forty minutes of sleep. Every field now names what it is.

---

## Evallo Session Logging

Evallo is the scheduling site the tutoring company uses, and it has no API. So Nova logs a session the way a person would: Playwright drives a headless Chromium through the website. "I finished with Taylor" becomes the tool `set_session_status`, which marks the 4:15 session Completed, No Show or Canceled.

**How a run goes.** Each step has a name, so a failure says where it broke.

1. `check_student`: the student is on his students list. If not, Prep Academy hasn't mapped them yet. He gets a push, the session goes on a local unlogged list, and nothing is logged.
2. `open_calendar` and `go_to_week`: the week view moves to the session's week.
3. `read_week`: every card in the week is read. Canceled sessions stay on the calendar, struck through with the same title, so they are dropped before matching. Exactly one live card must match the first name, the tutor and the start time to the minute. If none matches, he gets "No Evallo session for Taylor at 4:15 PM. Book it or check the time." If several match, the run stops rather than guess.
4. `open_event` and `verify_panel`: the panel's title, date, start time and student are checked against the request before anything is clicked. If it already reads Completed, the run returns "already done".
5. `open_status`, `pick_status` and `confirm_popup`: the status is chosen and Evallo asks "Are you sure?"
6. Dry run: `go_back` clicks "No, go back", then `verify_unchanged` reloads and proves nothing changed. Real run: `confirm`, then `verify_changed` waits for "Session updated successfully" and reads the new status back.

**Approval.** The tool is `EXTERNAL_WRITE`, so only the hokage tier can reach it. A real change is only staged through `pending_action`. Nova reads the question back word for word, and it runs only when `confirm_pending_action` comes on his very next turn. Dry runs go straight through.

**Where things are:**

| Path | What it is |
|---|---|
| `src/evallo_selectors.py` | Every selector, and the only file to edit when Evallo changes |
| `src/evallo_matching.py` | Which card a request means. Pure Python, tested without a browser |
| `src/evallo_runner.py` | The browser steps |
| `src/evallo_failures.py` | Failure records, deleted after 30 days |
| `src/evallo_unlogged.py` | Sessions taught that could not be logged yet |
| `src/evallo_session.py` | The saved login, its expiry, and push |
| `src/evallo_tools.py` | The Nova tool. It runs the script as a separate process, in `.venv-evallo` |
| `scripts/evallo_status.py` | By hand. Dry run unless `--real`; `--unlogged` lists the waiting sessions |
| `scripts/evallo_canary.py` | Nightly at 2 AM (`miles-evallo-canary.timer`); `--login-only` at 8 AM (`miles-evallo-logincheck.timer`) |
| `scripts/evallo_login.py` | One-time login, run on the Mac |

```bash
# Dry run, on the Pi. Changes nothing.
~/miles/.venv-evallo/bin/python scripts/evallo_status.py \
    --student Taylor --start "2026-11-23 16:15" --status Completed
```

**Logging in.** Evallo's login is a token in the site's localStorage that lasts 30 days. The Pi has no screen, so run `scripts/evallo_login.py` on the Mac. It opens real Chrome, because Google refuses its sign-in in an automated browser. Log in by hand. It saves only Evallo's data to `data/evallo_state.json`; then copy that file to the Pi's `~/miles/data/`. The canary warns three days before the token runs out. If it does run out, the next run stops with "Evallo login expired, log in again". No password is stored anywhere.

**The nightly canary.** At 2 AM it walks the script on his next session up to the open status menu. It closes the menu without choosing anything and proves the status is unchanged. It does not click an option. Evallo only offers Completed once a session has started, so the only option it could click on a future session is Canceled, every night, on a real student's lesson. A pass is one quiet line in the journal. A failure pushes "Evallo script broke at step X".

**Fixing a broken selector.** A failure saves `data/evallo_failures/<time>_<step>/`, which holds `failure.json`, `screenshot.png` and `page.html`. The snapshots hold students' contact details: the folder is git-ignored and pruned after 30 days.

1. Read `failure.json`. `step` says where it broke; `selector_key` and `selector` say what it was looking for.
2. Open `screenshot.png` to see what was on the page instead.
3. Find the element in `page.html`, or open Evallo and look. Prefer the words a person sees, and the element's role (button, textbox, gridcell). Never use Evallo's styling classes such as `text-[10px]`: they change on every redesign.
4. Change that one entry in `src/evallo_selectors.py`, then run a dry run.

---

## Approvals from the Phone

Some actions leave the Pi entirely: texting a student's family a reminder with the link to his Zoom room. Those run on a Mac, and none of them happen until he taps **Approve** on that one request in Nova.

1. The Mac proposes the action with the exact text and the chat it is for. The Pi stores it and pushes it to the phone with Approve and Skip buttons.
2. He taps Approve or Skip, on the lock screen or in the app's Approvals screen.
3. The Mac claims approved requests, does them, and reports back. Every step is logged with a timestamp, the exact text and the chat.

**Rules enforced in code, in `src/approvals.py`:**

* One request, one tap. There is no batch approve, and nothing approves itself.
* The Mac logs in with its own 90 day token, scope `agent`. It can propose, claim and report, and the approve routes refuse it, so a bug or a leaked token on the Mac cannot send a text he did not tap.
* A request expires two hours after it is proposed. An approval nobody claimed in time expires instead of running late.
* A claimed request is handed out once and never retried, so a text that may already have gone out cannot go out twice.
* The Pi knows students' chats by display name only. The Mac keeps the actual iMessage chat ids.
* A text may carry no link except his one Zoom link from `data/zoom.json` (a `pwd` and/or `omn` number in it is fine). A message that is exactly that link is allowed, so Messages shows the Zoom preview card.
* A reminder is two iMessages from one approval: the words, then the Zoom link alone. The payload's `parts` must join (with a blank line) to exactly the text on the card, and each part must pass the same check, or the Pi refuses it. The Mac sends each part once, in order. If the second fails after the first went, it says which part failed and never resends the first.
* The Mac drafts nothing within 30 minutes of a session's start, and checks again at the moment of sending. An approved text that hits the cutoff is not sent; he gets "Missed reminder for <Name> <time>, text them yourself if needed."

`/agent/notify` takes `title`, `body` (the lock screen's 500 characters), an optional `detail` (up to 1800), `approval_id` and `error`. They ride in the push, so a tap opens Nova's error screen with the whole message, or Approvals on that request.

### The Mac Watchdog and Expired Logins

The Mac checks in with `/agent/heartbeat` on every launchd run and every 30 second poll, and each scheduled job reports with `/agent/jobs/report` for the day it belongs to. `scripts/agent_watchdog.py` runs every 5 minutes (`miles-agent-watchdog.timer`) and pushes "Mac missed a job" when there is no check-in by 7:15 PM, or when the 7 PM drafts or the 9 PM summaries have not reported success 30 minutes after their time. Each alert is pushed once, with three buttons:

* **Retry when awake.** The missed job comes back in the answer to the Mac's next heartbeat, and the Mac runs it. A draft retried after midnight says "today", and no session within 30 minutes of its start is ever drafted.
* **Remind me at 7 AM.** The same alert is pushed again at 7 AM.
* **Skip tonight.** It is closed.

A job that later succeeds on its own closes its alert.

UF Zoom, Google and Evallo logins die when he changes his UF password. A job that lands on a sign in page, gets a 401 or `invalid_grant`, or finds a login screen in place of the page stops at once. Nothing types a password or retries a login. The Mac (`/agent/login-expired`) or the Pi's Evallo code pushes "<Service> login expired (password change?). On the Mac run: <exact command>", with the failed job, once per service per day. The Mac queues the failed drafts or summaries and runs them once that login has been redone. At 8 AM each saved login is tested read only: on the Mac by its own launchd job, and on the Pi by `evallo_canary.py --login-only`. Only a broken one pushes.

### Lesson Planning Lives in Claude, Not Here

Since Nov 25 2026, lessons are planned in his Claude "Tutoring" project (one chat per student) when he says "prep". The 7:37 AM lesson plan task is paused. MILES and the Mac do not plan lessons and must not start to.

That project reads Google Drive as its memory, so these are a contract:

* `Tutoring/<Name>/questions`: the used, baseline and misses docs. He maintains them; nothing here writes them.
* `Zoom summaries/<Name>`: written by the Mac's 9 PM summary grabber, one Doc per session, titled `<Name> summary <Mon DD>` (for example "Alex summary Nov 23").

Do not rename or move those folders, and do not change the grabber's folder name or title format. Prep depends on both.

Tutoring automation order: Phase 4 (did the session happen?), Phase 5 (Stage E replies) and Phase 6 (Evallo commands from Nova) go ahead as planned. Phase 4 starts only after a Thursday 7 PM reminder run works end to end. Phase 7 (multiple Nova chats) is lower priority, since tutoring chats live in the Claude project.

---

## Tech Stack

**Hardware**
* Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite, 64 bit
* ReSpeaker XVF3800 USB 4 mic array (on chip beamforming, noise suppression, AGC, echo cancellation), found by name rather than by ALSA card number
* Speakers on the array's own 3.5mm jack, so its echo canceller hears her on the same clock
* LD2410C mmWave presence radar on the GPIO UART

**Languages**
* Python 3.13 (pipeline, backend, orchestration)
* Swift and SwiftUI (companion app, separate repo)

**Audio and ML**
* openWakeWord (custom hey_nova.onnx)
* webrtcvad (endpointing)
* Deepgram Nova 3, with whisper.cpp as the on device fallback
* SpeechBrain ECAPA (speaker verification)
* Sentence embeddings (memory retrieval and local intent)
* PyAudio for input, aplay and ALSA for output

**Backend**
* FastAPI (REST and WebSocket)
* JWT authentication (HS256) with bcrypt password hashing
* SQLite in WAL mode
* Cloudflare Tunnel, dashboard managed
* systemd (voice loop, server, tunnel, and timers for health checks, wifi recovery and workout progression)

**APIs**
* Anthropic Claude Haiku 4.5, with Anthropic's web search tool
* Deepgram Nova 3
* ElevenLabs, Victoria voice on eleven_v3
* Google Calendar API v3
* Oura API v2
* Hevy API v1
* Prowl (his own job search tracker)
* Apple WeatherKit and Maps Server API (geocoding), signed with the push key

---

## Key Engineering Decisions

**Voice chosen by listening, not by reasoning.** When Nova sounded like she was reading rather than speaking, each suspect was rendered side by side on the same real reply and judged by ear: splitting replies into sentences, giving each sentence its neighbours as context, stability, speed, four other voices, and four synthesis models. Only the model made an audible difference.

**Endpointing moved from amplitude to webrtcvad.** The original energy threshold was never crossed by real speech at the measured capture level, so every recording ended on a timeout and long commands were cut off mid sentence. webrtcvad fixed the truncation and, with the silence window retuned, brought median perceived latency from about 8.1 seconds to 4.9.

**Response length cut by trimming history, not by instruction.** Nova's own past answers anchor her length far more strongly than any prompt line, so past replies are trimmed before being sent back as context.

**A failed turn never kills the process.** The voice loop has its own exception boundary, and a local network check picks what to say, so "I've lost wifi" is never spoken when the real problem is the API.

**Barge in that does not stop for her own voice.** Interrupting on any speech worked until it stopped her on her own first syllables, which leak through the echo canceller after hours of quiet, at a level no threshold could separate from a real voice. Instead, playback records what she sent to the speaker, and a run of sound whose loudness rises and falls with hers at the speaker's delay is ignored. Her echo matched at 0.94 and 0.97; his real interruptions over her scored 0.69 at most, and every one still stops her.

**The stdin.flush() discovery.** Writing audio chunks to aplay without flushing let Python buffer up to 64KB, adding about 1.5 seconds of phantom latency even with streaming on.

---

## API Endpoints

Live at `https://miles.lethanial.com`.

| Endpoint | Method | Auth |
|----------|--------|------|
| `/auth/login` | POST | No |
| `/auth/refresh` | POST | Yes |
| `/chat` | POST | Yes |
| `/chat/stream` | POST, server sent events | Yes |
| `/speak` | POST, returns mp3 | Yes |
| `/memories` | GET, POST | Yes |
| `/memories/{memory_id}` | PATCH, DELETE | Yes |
| `/memories/{memory_id}/history` | GET | Yes |
| `/memories/pending`, `/memories/{memory_id}/approve` | GET, POST | Yes |
| `/memories/similar`, `/memories/similar/dismiss`, `/memories/merge/draft`, `/memories/merge` | GET, POST | Yes |
| `/reminders`, `/reminders/{reminder_id}` | GET, DELETE | Yes |
| `/calendar/events`, `/calendar/events/{event_id}` | GET, PATCH, DELETE | Yes |
| `/lists`, `/lists/{list_id}/items` | GET | Yes |
| `/lists/items/{item_id}/check`, `/lists/items/{item_id}/uncheck` | POST | Yes |
| `/tasks/today`, `/tasks/{task_id}/complete` | GET, POST | Yes |
| `/sessions`, `/sessions/{session_id}` | GET, POST | Yes |
| `/sessions/{session_id}/turns` | POST, server sent events | Yes |
| `/sessions/{session_id}/close` | POST | Yes |
| `/capture` | POST, a capture key instead of a login: adds one task or list item, reads nothing | Key |
| `/approvals`, `/approvals/{approval_id}` | GET | Yes |
| `/approvals/{approval_id}/approve`, `/approvals/{approval_id}/skip` | POST | Yes, full login only |
| `/auth/agent` | POST | Yes, issues the Mac's agent token |
| `/agent/approvals`, `/agent/approvals/claim`, `/agent/approvals/{approval_id}/result`, `/agent/notify` | POST | Agent token |
| `/agent/contacts` | PUT | Agent token |
| `/agent/heartbeat`, `/agent/jobs/report`, `/agent/login-expired` | POST | Agent token |
| `/watchdog` | GET | Yes |
| `/watchdog/{alert_id}/retry`, `/watchdog/{alert_id}/remind`, `/watchdog/{alert_id}/skip` | POST | Yes, full login only |
| `/history` | GET | Yes |
| `/status`, `/status/details` | GET | Yes |
| `/ws` | WebSocket | Yes, token checked in the socket |
| `/docs` | GET | No, FastAPI's generated Swagger UI |

---

## Measured Performance

From the per turn `timing_log`. Every stage and its history is in [docs/LATENCY.md](docs/LATENCY.md).

| Metric | Value |
|--------|-------|
| Median perceived latency, last 30 Claude turns (Oct 2 to 12 2026) | 3.0 seconds, from the end of speech to first audio, including the silence wait that decides you have finished |
| ElevenLabs eleven_v3 time to first byte | 647ms median over 12 live turns (Sep 2026) |
| Registered tools | 43 |
| Test suite | 1899 passing |

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
| Unreleased | Native tool use with a permission gate, Google Calendar and Oura, calendar undo, local intent, speculative transcription, eleven_v3 voice | Landed |
| Unreleased | ReSpeaker XVF3800 array, Deepgram with Whisper fallback, ECAPA verification, web search, Hevy workouts and automatic progression, Prowl job search, barge in, picking up a reply that was cut off | Landed |
| Unreleased | Push notifications for reminders and timers through Apple's push service, and Apple Health data copied from the phone so Nova can answer about weight and runs | Landed, untested on a real phone |
| Unreleased | Home or away from a geofence on the phone, the first half of telling an empty apartment from a quiet one | Logging, not yet acted on |
| Unreleased | Approvals: the Mac proposes texts to students' families and Evallo edits, and nothing runs until he taps Approve in Nova | Landed, first test on a real phone in progress |
| Unreleased | Sessions: a nightly review pushed at 9:30 PM that closes out the day's tasks, plans two or three must dos for tomorrow and blocks them out on the calendar | On a branch, not yet tested on a real night |
| Next | Evening check in, SSH through the tunnel, morning briefings, Apple ecosystem integrations | Planned |

---

## Repository Structure

```
miles/
├── src/            Runtime. The three services import from here
│   └── tests/      pytest suite
├── scripts/        Operator tools run by hand: pronunciation, phrase bank, labelling, analysis
├── docs/           Engineering docs. Start with docs/SESSION_START.md
├── systemd/        Every unit and timer, versioned
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
