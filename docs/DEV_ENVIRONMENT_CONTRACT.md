# MILES headless development environment contract

**Status** — Reconciled against canonical main and ready for independent contract review. Application implementation is paused pending that review and the normal build gate.

**Canonical base** — `infra/dev-environment` is based on local `main` at `51676b912eac708d03024d02942ba05364ba89e6`, matching the checkout's local `origin/main`. The canonical migration list reaches 45, Evallo verification. Extracted initialization and all 45 migrations ran on blank in-memory SQLite on macOS. Repeat initialization left a single version row at 45 and the integrity check passed. This probe exercised SQLite only. See the [Friday handoff](work/dev-environment.md) for provenance, discrepancies, and follow-up checks.

## Objective and scope

A fresh checkout on arm64 macOS or Linux must support headless backend development with reproducible dependencies, synthetic data, generated MILES login credentials, offline tests, and a portable task handoff. Claude Code, Codex CLI, and cloud coding sessions use the same repository-owned commands. One agent's worktree, database, and credentials must be independent of another's.

This phase delivers the dependency bootstrap, isolated application startup and configuration, SQLite initialization and synthetic seed, local authentication, offline verification, and the portable orientation and handoff workflow. Existing production startup remains supported. Nova development, Pi diagnostic tooling and remote access, cloud environment configuration and orchestration, GitHub Actions, and deployment changes are later milestones.

Use the canonical [roadmap amendments](ROADMAP.md#amendments-agreed-nov-24-and-25-2026) and the [Tasks](TASKS_CONTRACT.md), [Sessions](SESSIONS_CONTRACT.md), and [Evallo filter](EVALLO_FILTER_CONTRACT.md) contracts for existing behavior. The roadmap's nightly backup, queued Tasks, planner and Evallo fixes, and private data follow-ups remain in their own phases. This infrastructure phase adds no roadmap feature or API contract. Supporting references are [infrastructure](INFRASTRUCTURE.md), [testing](TESTING.md), [session start and drift control](SESSION_START.md), [audio pipeline](AUDIO_PIPELINE.md), and [repository guidance](../CLAUDE.md).

## C1. Reproducible setup

- Provide one documented repository-owned bootstrap command and the same documented start and test commands on macOS and Linux. Use a per-worktree virtual environment, one exact Python 3.13 patch release validated on both platforms, pinned headless API and test dependencies, and a bootstrap check for SQLite FTS5. The documented Pi baseline is Python 3.13.5. The inspected Mac has Apple Git and Python 3.13.1 with FTS5, so basic inspection and cloning need no additional backend software on that Mac. Select Python 3.13 explicitly.
- Use the existing [runtime manifest](../requirements.txt), [test manifest](../requirements-dev.txt), and API import graph to determine the headless set. The current API import graph includes FastAPI, Pydantic, bcrypt, Passlib, python-jose, python-dotenv, HTTPX, Requests, NumPy, dateparser, Anthropic, ElevenLabs, Google auth and API client libraries, and requests-oauthlib. Pin Uvicorn, pytest, and the SDKs and transports still required by the headless entry point. Include a pinned WebSocket transport if `/ws` is exercised over the wire. Keep the existing manifests compatible with production and full-suite users.
- Keep PortAudio, ALSA tools, PyAudio, wake and speaker models, Torch, whisper.cpp, radar, systemd, cloudflared, and Docker out of the ordinary headless bootstrap. Mock or defer imports that would require them. Treat native voice and hardware testing as separate.
- Bootstrap may use authorized package sources. The application and tests must run after bootstrap with external application egress blocked. Record the existing suite command, `cd src && python3 -m pytest tests/ -q`, and document the supported headless command and any hardware-only exclusions with their reasons.

## C2. Startup and production compatibility

- Development and test entry points must choose and validate their mode, state paths, and provider configuration before importing any module that initializes the database, reads credentials, creates a live client, or starts work. Missing or invalid mode and state configuration on these entry points must fail closed. Direct pytest invocation must install the test guard before collection or fail before importing MILES.
- In dev/test, validate resolved paths for the database and all runtime state within the designated worktree or test state directory. Reject outside paths, traversal, and symlink escape. Omitted overrides must resolve to dev/test state or fail. Guard filesystem access as needed so production paths are rejected before opening them, even when the caller supplies an alias. Give concurrent worktrees separate state and ports, and make repeat startup safe.
- Preserve the existing production startup invocation, including the systemd Uvicorn `server:app` entry point, existing environment-file convention, configuration defaults, and established production startup, auth, migration, and route behavior. `src/server.py` currently initializes SQLite during import and clears the Evallo verification flag on application startup. Preserve that production reset, and ensure any dev or test startup touches only its isolated state. An existing production caller must not need a new mode variable or service edit. Establish the boundary in the dedicated dev/test entry points and verify compatibility with synthetic configuration in an isolated test. This phase makes no changes to running services or deployment configuration.
- Start a single-process API on `127.0.0.1` on a documented development port. Keep voice, scheduled jobs, health and deployment operators, and both systemd and launchd workers outside the headless launcher. Make system status deterministic or return the existing unavailable values for Pi-specific readings.

## C3. Application isolation and credentials

- Skip ambient dotenv files, production defaults, OAuth and browser state, Keychain, credentials inherited from the invoking shell, and ambient proxy settings in dev/test. Use independently generated local password and JWT signing material with the existing login and token contract. Store any persisted dev credentials in ignored owner-readable files, with mode 0600, and exclude their values from output, handoffs, and commits. Pass secrets through protected files, stdin, or a pipe. Keep account and fixture identifiers demonstrably synthetic.
- Select in-process fakes or explicit disabled providers before creating live clients. Reuse the test suite's Google, Claude, and embeddings substitutes where practical. For supported routes, provide deterministic chat, session, and calendar behavior. API behavior for disabled providers must be documented and testable.
- Cover access paths to Google Calendar and OAuth, Evallo, production MILES endpoints and worker callbacks, Anthropic and model web search, ElevenLabs, Deepgram, APNs, Apple Maps and WeatherKit, Oura, Hevy, and Prowl. Leave external browser automation, including Canvas when introduced, and Mac jobs for Zoom, Drive, Messages, and Reminders disabled. Include read paths, notifications, downloads, and subprocesses as well as writes.
- Enforce egress isolation independently of fake-provider selection. Apply the restriction to the application and test process trees, before imports and collection, including DNS, direct connections, alternate clients, redirects, and child processes. Permit in-process traffic and only explicitly required loopback endpoints. Record attempted violations without payloads or values and fail the test run even if code catches the original error. A host without the required runtime boundary must fail the supported launcher rather than silently run with live egress. Use local or reserved test targets to verify denials.
- The coding agent runs outside the application's network boundary. It retains its authorized model, repository, and tool access under its own permissions. Run application commands inside the isolated runtime and dependency installation in its authorized setup context. Agent access supplies neither live MILES credentials nor access to production services.

## C4. SQLite and synthetic fixtures

- Create a blank file-backed SQLite database using the existing ordered runner in [database.py](../src/database.py) and `schema_version` tracking. The canonical head is migration 45, whose DDL and version update share a transaction. Verify WAL, FTS5, schema head, repeat initialization, and upgrade from a synthetic prior schema with representative rows retained. Preserve migration behavior. Handle startup and reset without inspecting or importing a production database.
- Keep schema initialization and synthetic seeding separate. The seed command is rerunnable and creates only fake records required to exercise the supported API. Cover lists, tasks, reminders, sessions, and capture. The canonical API includes `/tasks/overview`, `/sessions`, `/capture`, and `/agent/evallo/sessions`. For calendar paths, use fake identifiers and events, including conflicts and tagged Evallo records. Keep the write protections and the existing rules for fresh, stale, and unknown verification and the documented out-of-window fallback. Exercise those paths with synthetic calendar data and snapshots.
- Retain per-test SQLite databases and extend the import-time guards in [conftest.py](../src/tests/conftest.py). The current fixture redirects the database and calendar roles paths, blocks one known production DB path, and supplies provider fakes and a quiet Evallo state for tests outside its own suite. Reuse [fake Google](../src/tests/fake_google.py) and the existing [Tasks](../src/tests/test_tasks_api.py), [Sessions](../src/tests/test_sessions_api.py), [capture](../src/tests/test_capture.py), and [Evallo verification](../src/tests/test_evallo_filter.py) tests. Test worktree isolation and confirm collection has no hardware, production data, or model download dependency.

## C5. Portable handoff and independent review

Keep a short Markdown handoff such as `docs/work/<feature>.md` on the feature branch. Update it at milestones and handoffs. Use relative links and reproducible commands.

| Field | Required content |
|---|---|
| Objective | Outcome, scope, and measurable acceptance criteria |
| Contracts and baseline | Relevant contracts and roadmap sections, base SHA, and paired client SHA when applicable |
| Decisions | Accepted decisions, reasons, and outstanding owner decisions |
| Progress | Completed and pending work, relevant files, and implementation commits |
| Test results | Exact command, platform, tested SHA, result, and skipped or unverified cases |
| Unresolved findings | Severity, file and line, concrete case, and disposition |
| Restrictions | Permitted data and execution scope, production restrictions, and permissions still needed |
| Exact next action | One runnable command or specific edit or review target and its expected result |

Repository context includes contracts, decisions, source, reproducible setup, synthetic fixtures, and sanitized results. Keep local agent memory, chats, real datasets, credentials, browser state, and private evidence outside the handoff. Include enough context for an independent session to use Git without reading the prior conversation.

For a local contract review, Claude Code uses the configured Codex CLI workflow and supplies the contract and relevant roadmap section only. For the end-of-phase review, specify the exact implementation commit under review and the base or diff range. Supply the diff, its touched files, and the contract. Record findings against the reviewed commit, with file and line and a concrete failure case. Review results may land in a later documentation commit. A document does not need to identify its own containing commit. Follow the owner's limits on repeat reviews and builder worktrees.

After acceptance, merge the infrastructure changes into the agreed integration base. Start subsequent feature branches from that base and deliberately rebase existing work, checking migration numbering and API changes. An infrastructure merge is independent of any deployment.

## C6. Shared orientation and session workflow

Use root [AGENTS.md](../AGENTS.md) as the shared orientation entry point for Claude Code, Codex CLI, and independent cloud sessions. Its current introduction directs agents to `CLAUDE.md` first. When implementing this section, update that orientation to point to the navigation map, safe development commands, the handoff convention, and the workflow below. `CLAUDE.md` references the shared workflow and retains Claude-specific planning, delegation, and coordination instructions. Reuse existing documentation such as `docs/SESSION_START.md` where practical. Distinguish ordinary development commands from production operator procedures.

Keep the navigation map concise and verify every path against the canonical checkout. For each major subsystem, list the governing contract or design document, actual source entry points, test files or directories, and dependencies that affect how it can be developed or tested. Cover API and authentication, SQLite and migrations, tasks and lists, sessions and capture, model and memory, calendar and Evallo, other integration and worker boundaries, and the voice and hardware stack. Link the existing docs for details. Read the shared entry point and only the map entries, contracts, source, and tests needed for the assignment. For independent reviews, the owner's narrower review scope still applies.

| Short request | Standard behavior |
|---|---|
| Start a feature | Identify the feature and applicable contract or roadmap section. Confirm the approved base and Git state, create or use the authorized feature worktree, and open its handoff with scope, acceptance criteria, and next action. Follow the existing planning and review gates before building. |
| Resume a feature | Match the feature to its local or cloud branch and read its handoff. Verify the checkout, branch, HEAD, and working-tree state against the handoff. Check the linked contracts, source, and tests for the documented next action, then continue it within authorization. If the branch is ambiguous or Git state conflicts with the handoff, resolve that discrepancy before changing work. |
| Wrap up | Stop at a coherent checkpoint. Update the handoff with relevant files, decisions, progress, exact test outcomes, blockers, restrictions, and the exact next action. Inspect and stage only intended changes, preserve unrelated work, and prepare a safe local commit using normal hooks and guards. A wrap-up grants no push or deployment authority. |
| Request an independent review | Identify the review type and use the scoped handoff in C5. For implementation review, provide the exact reviewed SHA, base or diff range, and required files. Record findings against that SHA, including when the handoff update lands in a later documentation commit. |

Keep Git hooks and safety guards in effect for worktree creation, transfer, staging, and commits. If a guard blocks an action, report the blocker and follow the authorized workflow.

## Acceptance criteria

1. **Clean setup and API startup.** Fresh macOS arm64 and Linux worktrees use the same supported bootstrap, start, and test commands. A real loopback HTTP smoke test proves readiness, successful synthetic login and refresh, and refusal of invalid auth. Test status without Pi dependencies and streaming or WebSocket behavior where the supported API uses it.
2. **Isolation under failure.** Unsafe paths and symlinks, missing dev/test setup, inherited production credentials, live integration configuration, and forbidden DNS, direct egress, and child-process traffic are refused or excluded before the relevant side effect. Existing dev/test callers cannot reach a production fallback. Test guards catch swallowed attempts. Two parallel worktrees cannot read or modify each other's state.
3. **Database lifecycle.** Fresh file-backed schema, WAL and FTS5 checks, repeat initialization, synthetic prior-schema upgrade with row preservation, and repeatable synthetic seeding all pass on both platforms.
4. **Offline coverage.** Run supported regression tests for auth and token scopes, lists and tasks, reminders, fake calendar reads and writes, tool permissions, disabled providers, and protected Evallo writes. Cover the canonical Tasks overview, capture, and sessions routes. Record hardware-only exclusions and exercise provider failure paths without live services.
5. **Production compatibility.** Every existing production startup invocation that the phase changes indirectly still works under a synthetic production-shaped configuration and isolated runtime. The checked-in systemd command, environment conventions, and production API behavior remain compatible. Only synthetic verification is required for this phase.
6. **Portable orientation and handoff.** A fresh session can start from `AGENTS.md`, find a subsystem's contract, current source, tests, and dependencies, and resume from a feature handoff without reading unrelated material or relying on a prior conversation. Verify the map against the canonical checkout and demonstrate the start, resume, wrap-up, and independent review workflow with a synthetic feature handoff.

After independent contract review and the normal build gate, implementation order is to establish the pre-import runtime and test safety boundary, pin dependencies and verify the DB and auth lifecycle, complete offline regression coverage, and hand off the implementation commit for independent review. No Pi evidence is needed to meet these criteria.

## Later milestones

### Nova development

Pair backend and Nova SHAs and check their endpoint contracts. Use a separate Nova Dev scheme with a distinct bundle ID, Keychain namespace, preferences, and explicit local backend URL for both ordinary requests and auth flows. Reject production URL fallback. Begin with the simulator and disable APNs registration, HealthKit synchronization, and geofencing for the dev build. Consider a private TLS endpoint for a physical phone only as later work. Release configuration and stored production credentials remain isolated.

### Diagnostics and deployment separation

| Work category | Scope and authority |
|---|---|
| Ordinary software development | Mac or cloud, synthetic data and isolated runtime under normal development authorization |
| Offline diagnostics | Mac only, using an owner-authorized private dataset, kept outside the repository and processed locally |
| Live Pi hardware diagnostics | Separately owner-authorized, time-bounded observation of the Pi with named actions and resources. Any live action beyond the approved diagnostic scope requires separate authorization. Deployment requires its own approval. |

Define the symptom and permitted evidence first. Use existing measurements where possible. For live observations, prefer existing logs and counters and avoid changing services, thresholds, gain, device configuration, or acquiring an occupied microphone or serial device. Any collection involving such actions needs separately scoped authorization.

Use owner-run offline local scripts to read private inputs and return only sanitized measurements. Running a coding agent locally does not guarantee the data stays on the machine. Keep raw inputs and outputs outside its context and access, and give it only owner-reviewed sanitized results. Keep the raw data in approved local storage outside the worktree and outside automatic sync or export. Raw recordings, private datasets, real database contents, credentials, and identifying mappings must never enter Git or be automatically transferred to cloud environments. Exclude credentials from diagnostic bundles.

Future analyses should align wake-word false triggers and misses with labeled clips, measure audio clipping, levels, channel behavior, and echo, compare genuine and impostor speaker scores at the observed threshold, compare radar distance or energy and presence transitions against observations, and correlate logs with restarts, device dropouts, or timing. Record sample counts, device/model revisions, test conditions, and uncertainty in sanitized results. Prefer synthetic regression inputs such as generated audio, score arrays, radar traces, and fabricated logs. Hardware validation of a candidate patch and deployment remain separate actions with separately granted authorization. Any deployment must use the established approval, fresh backup, and rollback workflow.

### Cloud environments and reviews

Reuse the same versioned bootstrap and test commands in independent tasks. For review, select the same immutable implementation SHA and provide the same scoped handoff to each reviewer. Generate synthetic state per task. Use existing tool and repository access. This milestone adds no authentication bridge, copied auth file, new credential, API spending, or access to live integrations.

### GitHub Actions

After portable local validation, add CI using the same bootstrap and test commands. Make startup, production isolation, database migration, regression, permission, and private-data scanning checks required once validated. Run synthetic fixtures and generate ephemeral test auth. Scan the proposed changes and tracked paths for private data and credentials, reporting locations and categories without matched values. Use read-only repository permissions for checks and restrict application runtime egress. CI has no deployment path to the Pi and uses no production secrets.

## Gates before implementation

The authorized documentation transfer and canonical reconciliation are complete. Submit this contract and the relevant roadmap sections for independent review. Begin implementation only after the review and the owner's normal build gate. Python and dependency pins, file-backed WAL and upgrades, runtime egress on both runners, and clean HTTP startup remain implementation checks.

If a later compatibility or hardware milestone genuinely depends on current Pi state, separately authorize only the minimal observation needed. Examples are a deployed revision, schema revision, sanitized unit difference, or hardware measurement that source and docs cannot establish.
