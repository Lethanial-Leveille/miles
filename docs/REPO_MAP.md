# Repo map: find a feature in two minutes

One page across all three repos. For the backend's own areas in more detail see [REPOSITORY_MAP.md](REPOSITORY_MAP.md). Rules start at [AGENTS.md](../AGENTS.md), the order of work is [ROADMAP.md](ROADMAP.md), open findings are [KNOWN_LIMITS.md](KNOWN_LIMITS.md).

## The three repos

| Repo | Where | What it is | Tests |
|---|---|---|---|
| Backend | `~/PersonalProjects/MILES-dev/backend` (the Pi runs `~/miles`) | Python 3.13, FastAPI, SQLite, the voice loop, the brain, approvals and the watchdog | `./scripts/dev test` (needs `./scripts/dev bootstrap` once per checkout) |
| Mac agent | `~/PersonalProjects/miles-evallo-mac` | Does what only the Mac can: Evallo and Zoom with the saved logins, iMessage, launchd jobs | `.venv-evallo/bin/python -m pytest -q src/tests --ignore=src/tests/test_evallo_tools.py` (that file imports a Pi module, see KNOWN_LIMITS) |
| Nova | `~/PersonalProjects/Nova` | SwiftUI iPhone app: chat, tasks, sessions, approvals, push | `Tests/` swiftc checks and `xcodebuild` for the simulator |

Branches and work live in git worktrees under `~/PersonalProjects/MILES-dev/worktrees/`. `main` of the backend is owned by whoever is deploying: check `git worktree list` and the branch before editing. Nothing deploys by itself.

## Where each feature lives

Paths are relative to the repo named in the first column. Contracts and handoffs are in the backend's `docs/`.

| Feature | Code | Tests | Contract | Handoff or work notes |
|---|---|---|---|---|
| Brain and tools (voice and text turns) | backend `src/brain.py`, `tools.py`, `task_intent.py` | `test_brain_loop`, `test_task_intent*`, `test_voice_action_safety` | [BRAIN.md](BRAIN.md), `work/tasks-intent-contract.md` | `work/tasks-voice-safety-amendment.md` |
| Tasks and lists | backend `src/tasks_view.py`, `lists_tasks.py`, `task_scheduling.py`; Nova `Views/TasksView.swift`, `ListsView.swift` | `test_tasks_api`, `test_lists_tasks*`, `test_task_*` | [TASKS_CONTRACT.md](TASKS_CONTRACT.md) | [handoffs/](handoffs/), `work/tasks-*.md` |
| Sessions and nightly review | backend `src/sessions.py`, `nightly_review.py`, `session_*.py`; Nova `Views/SessionsListView.swift` | `test_sessions*`, `test_review_v2`, `test_session_*` | [SESSIONS_CONTRACT.md](SESSIONS_CONTRACT.md), [SESSIONS.md](SESSIONS.md) | none |
| Calendar and the Evallo filter | backend `src/calendar_tools.py`, `calendar_roles.py`, `evallo_filter.py`; Mac `src/evallo_sessions.py` | `test_calendar_*`, `test_evallo_filter`; Mac `test_evallo_sessions.py` | [EVALLO_FILTER_CONTRACT.md](EVALLO_FILTER_CONTRACT.md), the permanent rules in [ROADMAP.md](ROADMAP.md#evallo-rules-permanent) | none |
| Approvals (tap in Nova, run on the Mac) | backend `src/approvals.py`, server routes under `/approvals` and `/agent`; Mac `src/stage_d.py`, `agent_runs.py`; Nova `Views/ApprovalsView.swift`, `Services/PushService.swift` | `test_approvals`; Mac `test_stage_d.py`, `test_agent_runs.py` | the docstring of `approvals.py` | none |
| Tutoring reminders (7 PM drafts, texts) | Mac `src/stage_d.py`, `reminders.py`, `imessage_send.py`, `contacts.py` | Mac `test_stage_d.py` | [README](../README.md) tutoring section | none |
| Tutoring session follow ups | backend `approvals.py`, `agent_watchdog.py`; Mac `src/session_followups.py`, `evallo_runner.py`; Nova `PushRoute.swift`, `ApprovalsView.swift` | `test_followups_*`; Mac `test_session_followups.py` | `TUTORING_PHASE4_CONTRACT.md` on `feature/tutoring-phase4` | `TUTORING_PHASE4_DEPLOY_PLAN.md` on the same branch |
| Evallo status script and canary | Mac `src/evallo_runner.py`, `evallo_selectors.py`, `scripts/evallo_status.py`; backend `src/evallo_tools.py`, `scripts/evallo_canary.py` | Mac `test_evallo_*`, backend `test_evallo_*` | [ROADMAP.md](ROADMAP.md#login-failsafe-and-the-evallo-token) | none |
| Mac watchdog and login alerts | backend `src/agent_watchdog.py`, `systemd/miles-agent-watchdog.*`; Mac `src/jobs.py`, `logins.py` | `test_agent_watchdog` | docstring of `agent_watchdog.py` | none |
| Push notifications and the phone | backend `src/push.py`, `apple_keys.py`; Nova `Services/PushService.swift` | `test_push` | [INFRASTRUCTURE.md](INFRASTRUCTURE.md#push-notifications-oct-30-2026) | none |
| Memory | backend `src/memory_*.py`, `embeddings.py`; Nova `Views/MemoriesView.swift` | `test_memory_*`, `test_remember_tool` | [BRAIN.md](BRAIN.md) | [APP_HANDOFF_DUPLICATES.md](APP_HANDOFF_DUPLICATES.md) |
| Quick capture | backend `src/capture.py`, `scripts/capture_key.py` | `test_capture` | [ROADMAP.md](ROADMAP.md#quick-capture-built-nov-25-2026) | none |
| Health, Oura, Hevy | backend `src/health_tools.py`, `oura_tools.py`, `hevy_*.py` | `test_health`, `test_oura_tools`, `test_hevy_*` | [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | none |
| Voice, wake and audio | backend `src/voice_main.py`, `audio.py`, `wake_*.py`, `tts.py` | `test_voice_pipeline`, `test_wake_*` | [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md), [VOICE_OUTPUT.md](VOICE_OUTPUT.md) | [LATENCY.md](LATENCY.md), [INCIDENTS.md](INCIDENTS.md) |
| Study (learn and quiz) | backend [STUDY.md](STUDY.md) | none yet | [STUDY.md](STUDY.md) | none |
| Dev environment and the review loop | backend `scripts/dev`, `scripts/loop`, `scripts/dev_runtime.py` | `test_dev_environment` | [DEV_ENVIRONMENT_CONTRACT.md](DEV_ENVIRONMENT_CONTRACT.md), [DEVELOPMENT.md](DEVELOPMENT.md) | `work/dev-environment.md`, `work/engineering-workflow.md` |
| Deploy and the Pi | backend `scripts/deploy.sh`, `systemd/` | `scripts/tests` | [INFRASTRUCTURE.md](INFRASTRUCTURE.md) | none |

## Reading order for a fresh agent

1. [AGENTS.md](../AGENTS.md): tiers, the loop, what counts as a Blocker, production limits, secrets.
2. This page, then the feature's row above. Open its contract, then its code, then its tests.
3. [ROADMAP.md](ROADMAP.md) for where the work sits in the order, [KNOWN_LIMITS.md](KNOWN_LIMITS.md) for what is knowingly open.
4. [SESSION_START.md](SESSION_START.md) when picking up someone else's branch.

Older decisions and measurements: [ENGINEERING_HISTORY.md](ENGINEERING_HISTORY.md). Finished backlog items: [archive/](archive/).
