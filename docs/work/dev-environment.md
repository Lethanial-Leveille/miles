# MILES development environment handoff

## Objective and baseline

Deliver the [approved headless development contract](../DEV_ENVIRONMENT_CONTRACT.md#acceptance-criteria) with reproducible dependencies, synthetic backend state, generated credentials, an offline suite, and a portable repository workflow. The work is on `infra/dev-environment`, based on canonical `main` at `51676b912eac708d03024d02942ba05364ba89e6`. The implementation resumed from `eddb502` after the owner installed the dependencies. The owner held acceptance after independent review of implementation `59f7627` and handoff `13cb8ac`. There is no paired Nova revision for this phase.

Use the [Tasks contract](../TASKS_CONTRACT.md), [Sessions contract](../SESSIONS_CONTRACT.md), [Evallo filter contract](../EVALLO_FILTER_CONTRACT.md), and [permanent Evallo rules](../ROADMAP.md#evallo-rules-permanent) for existing API behavior. The canonical migration head is 45. Schema 40 in the original scratch repository and schema 44 in an earlier source mirror are historical.

The contract and handoff were transferred from the owner authorized scratch commit `0d184d5` and reconciled at `8accb63`. An earlier scratch checkout disabled checkout hooks. The historical configuration at that point remains unknown. Canonical worktree creation, staging, and commits for this phase used normal Git hooks.

## Decisions and progress

The owner approved a clean environment, private state, fakes and explicit provider disablement, and Python network and subprocess guards for ordinary work. The approved scope excludes native process containment and a change to production startup. The launcher derives an API port from the worktree path and uses a separate port range for test runs.

| Revision | Change |
|---|---|
| `eddb502` | Simplified isolation and prepared the headless manifest. |
| `ec1f588` | Checked in the owner generated lock. Added preimport activation, private state and generated auth, fake Calendar, model and embeddings providers, disabled live integrations, network controls, the isolated suite, and repository navigation. The WebSocket handler runs the synchronous chat turn on a worker thread so it works inside the ASGI event loop. |
| `9c6246a` | Fixed guarded directory opens during development init, seed, and reset. |
| `77d49ff`, `59f7627` | Fixed temporary staging cleanup and moved the Evallo worker test lock into its private fixture. An earlier test run created the zero byte synthetic lock in the worktree's ignored data directory. It was removed. |
| `f986239` | Restricted WebSocket worker dispatch and full login authorization to the headless profile. Added scoped token rejection and production dispatch regressions. Disabled Zoom file opens, guarded supported opens against existing aliases, and rejected symlinked virtual environment directories before sync or launch. Verified default production selection with guards active, watched for unsupported pytest sentinel access, exercised the Calendar selector in the API smoke test, and corrected the loopback fixtures to check their actual bind and reuse the assigned address. |

The [production WebSocket repair](../BACKEND_TODO.md#production-websocket-chat-repair) is queued separately. The compatibility test uses a synthetic chat callback to assert the existing dispatch on the ASGI event loop.

Start from [AGENTS.md](../../AGENTS.md), [the development commands](../DEVELOPMENT.md), and [the subsystem map](../REPOSITORY_MAP.md). The runtime boundary, isolated entry points, and acceptance checks are in [dev_guards.py](../../scripts/dev_guards.py), [dev_runtime.py](../../scripts/dev_runtime.py), and [test_dev_environment.py](../../src/tests/test_dev_environment.py).

## Verification of corrections

macOS arm64 with CPython 3.13.1 and SQLite 3.45.3. Tests used synthetic inputs and guarded state.

| Command or check | Result |
|---|---|
| `./scripts/dev test -- tests/test_dev_environment.py tests/test_label_spoken.py tests/test_approvals.py tests/test_device_token.py tests/test_agent_watchdog.py tests/test_calendar_app.py tests/test_production_db_guard.py --tb=short -rs` | Exit 0. 124 passed, 7 loopback skips, one Starlette test client deprecation warning. |
| `./scripts/dev test -- --tb=short -rs` | Exit 0. 2,268 passed, 13 skipped, one Starlette test client deprecation warning in 41.22 seconds. Seven skips were denied loopback binds and six required live voice prerequisites. Enrollment and mic channel tests remain excluded by the headless launcher. |
| `./scripts/dev verify-direct` | Exit 0. The early plugin import was refused before any sentinel access. In a disposable source copy, both a file read and a SQLite read only URI connection before the app import caused the verifier to fail. |
| `./scripts/dev verify-production` | Exit 0. The unconfigured selector used the production path with a repository virtual environment prefix and synthetic dotenv settings. Startup, migration from 44 to 45, row preservation, restart, Evallo reset, and in process WebSocket dispatch passed. A disposable source copy with the production worker dispatch reintroduced failed the WebSocket assertion. The runner denied the real bind. |
| `sh scripts/tests/test_dev_venv.sh && ./scripts/dev check` | Exit 0. Stub sync and interpreter markers confirmed rejection of the directory symlink before either could run. A real directory containing a symlinked interpreter passed the launcher check. FTS5 was available in the installed environment. |
| `git diff --check` | Passed. |

## Owner verification on macOS

The owner ran the approved development environment and label server tests from the normal macOS Terminal. The run reported **34 passed, zero failed, zero skipped, and three dependency deprecation warnings in 4.74 seconds**. The real loopback API lifecycle passed. This closes the macOS listener evidence gap in the checks for implementation `f986239`. The owner reported the test targets and results. The supported command to reproduce those targets is

```sh
./scripts/dev test -- tests/test_dev_environment.py tests/test_label_spoken.py
```

The [dependency deprecation warnings](../BACKEND_TODO.md#headless-development-dependency-warnings) are deferred maintenance.

## Verification before independent review

macOS arm64 with CPython 3.13.1 and SQLite 3.45.3. The implementation revision under test was `59f7627fefcbbeb89d46655b983856f9cfd8d3d2`. The production unit and dependency manifests remain in their existing locations. The validation used only synthetic configuration and databases.

| Command or check | Result |
|---|---|
| `uv lock --check --offline --no-config --no-python-downloads --default-index https://pypi.org/simple` | Passed. Resolved 67 locked packages. All 164 wheel entries specify SHA-256 hashes and the 66 registry packages use the approved index. |
| `uv sync --locked --check --offline --no-build --no-config --no-python-downloads --default-index https://pypi.org/simple` | Passed. Checked 66 installed packages with no changes. |
| `./scripts/dev check` | Passed the FTS5 probe. |
| `./scripts/dev verify-direct` | Passed. Unsupported pytest with an explicit early app plugin failed before accessing the synthetic sentinel. |
| `./scripts/dev verify-production` | Passed the staged `server:app` startup and shutdown lifecycle with no mode variable, generated dotenv auth, migration 44 to 45, row preservation, WAL, integrity, and Evallo active flag reset on restart. The runner denied the actual loopback bind. |
| `MILES_MODE=invalid MILES_DB_PATH=/tmp/synthetic-override.db PYTEST_ADDOPTS=--unsupported-ambient-option ./scripts/dev test -- --tb=short -rs` | Passed with exit 0. 2,260 passed, 13 skipped, and one Starlette test client deprecation warning in 36.28 seconds. Seven skips were denied loopback binds and six required live voice prerequisites. Enrollment and mic channel tests were excluded because the headless dependencies omit PyAudio and SciPy. |
| `./scripts/dev start` at `9c6246a` | Uvicorn completed app startup and shutdown. The runner denied the approved IPv4 loopback bind with EPERM. A plain socket bind to the same address also received EPERM. |
| Two source trees exported from `9c6246a` | Both installed independently from the offline cache with `uv sync --locked --offline --no-build --no-config --no-python-downloads --python "$(command -v python3.13)" --default-index https://pypi.org/simple`. Both passed `./scripts/dev bootstrap`, `./scripts/dev init`, and `./scripts/dev seed`. The credentials differed and had mode 0600. Reset refused a hard linked peer WAL and an active operation lock. Reset after release left the peer database unchanged. |
| Repository navigation | Verified the 67 local map links and the linked orientation and development paths against the canonical checkout. The map demonstrates start, resume, wrap up and independent review with a synthetic reminder task. |

## Acceptance status

- **macOS arm64 development acceptance complete.** The Terminal result supplies the real loopback evidence for synthetic login and refresh, invalid credential refusal, status, streaming, and listener shutdown. The earlier macOS results cover setup, isolation, database lifecycle, offline regressions, staged production compatibility, and repository navigation.
- **SQLite residual risk accepted.** The owner accepted the possibility that an independent process running as the same user can replace a database file or sidecar between validation and the native SQLite open. The [contract](../DEV_ENVIRONMENT_CONTRACT.md#c2-startup-and-production-compatibility) records the limit under the accidental failure threat model.
- **Cross platform acceptance open.** Clean setup, startup and shutdown on a real loopback listener, and the supported suite still need verification on Linux.

## Cloud readiness preflight

**Disposition: hold publication and normal cloud checkout.** The Linux command path is identified. Privacy clearance, live repository automation settings and Linux execution remain open.

This was a Medium, read only preflight from the clean macOS worktree on `docs/engineering-workflow` at `a39167f`, whose parent is the accepted macOS infrastructure checkpoint `6e72b9e`. The local `infra/dev-environment` ref remains at `6e72b9e`. The launcher, lock and listener tests match the accepted checkpoint. The local `main` and cached `origin/main` both resolved to `51676b9`. This section's commit records the preflight. It changes only this handoff.

### Publication and automation

- The inspected Git trees at `a39167f` and `51676b9` contain zero tracked GitHub Actions workflows and no definitions for the other common CI systems checked. The effective local hooks path has no executable `pre-push` hook.
- If installed and enabled, [the deployment timer](../../systemd/miles-deploy.timer) invokes [the deployment service](../../systemd/miles-deploy.service). [Auto deploy](../../scripts/auto_deploy.sh) fetches and selects `origin/main`, checks the local `main` state, and tests the candidate before calling [the manual deploy script](../../scripts/deploy.sh). That script requires local `main`, pulls `origin main` and restarts services. A push to a separate topic ref that leaves `main` unchanged does not select that topic commit through these checked in scripts. A later update to `main` could deploy if the live timer is active and its gates pass.
- Current GitHub refs, Actions settings, webhooks, installed apps, branch settings and external deployment or CI integrations are unverified. The local refs are cached, and the available GitHub connector does not expose the repository administration surfaces needed to clear these triggers. Current Pi timer state is also unverified under the production access restriction. The dated installation note in [Infrastructure](../INFRASTRUCTURE.md#systemd-services) is historical evidence.

### Privacy

- The repository is documented as public. Seven representative paths for dotenv files, generated development state, calendar roles, a browser session file and a database matched ignore rules. Ignore rules do not inspect committed history.
- **History scan unverified.** No approved scanner was available in the repository, on PATH or at standard local binary locations. The checks found none of Gitleaks, TruffleHog, detect-secrets or git-secrets. No history content scan was run and there is no finding count to report. No private datasets, credentials or production state were opened.
- The [privacy roadmap](../ROADMAP.md#private-data-out-of-the-repo-nov-26-2026) records a completed calendar identifier scrub in current source and one known personal name in earlier public history. The owner has yet to decide on the history rewrite, which requires separate force push approval. A pre push scanner for private names, emails, calendar and tutor identifiers and credentials is queued. The roadmap also queues [removal of raw tailnet addresses from current paths](../ROADMAP.md#tailscale-hostname-cleanup-queued-nov-26-2026) and [retirement of the temporary Pi Evallo login copy](../ROADMAP.md#login-failsafe-and-the-evallo-token-planned-nov-25-2026). The separate [Phase S audit](../ROADMAP.md#phase-s-security-audit) remains required before unattended errands. None of these items authorizes accessing private data for this preflight.

### Minimum owner actions and Linux verification

1. Clear the publication gate. Check the actual remote visibility, intended refs, Actions, webhooks, installed apps and deployment integrations from the owner account. Confirm the proposed topic ref and any PR events cannot reach production or cause unapproved external actions. Keep `main` untouched. Approve a read only history scan on the Mac and the disposition of known historical exposure before allowing a normal clone into the cloud. Review the exact outgoing commits, added lines, messages, paths and branch name. Keep scanner match text, paths containing private data, and author or credential values out of agent output and shared reports. Record only sanitized counts and dispositions. Publish or transfer the selected revision only with separate owner authorization and normal guards.
2. Select a fresh Linux x86_64 or aarch64 session with Python 3.13, uv, an FTS5 capable SQLite build and permission to bind IPv4 loopback. Keep port forwarding and live integrations off. Allow the locked public registry and wheel host during bootstrap if the cache is empty. Use the owner approved checkout and record its exact revision. Use only checked in source and generated synthetic state. Keep local environment files, database copies, browser profiles and credentials out of the session.
3. From the repository root, confirm a clean checkout and run the following. Stop on an unexpected result.

```sh
git status --short --branch
git rev-parse HEAD
uname -m
command -v python3.13
command -v uv
./scripts/dev bootstrap
./scripts/dev init
./scripts/dev seed
./scripts/dev start --port 42000
```

Bootstrap runs `uv sync --locked --no-build --no-config --no-python-downloads` against `uv.lock` with the existing Python 3.13 interpreter and the public PyPI index, then probes FTS5. The lock lists 67 packages and 164 SHA256 wheel entries. While the API is running, use a second shell in the same session for a credential free loopback response check. Use an unused approved loopback port and change both commands together if the example port is occupied.

```sh
python3.13 -c 'import http.client; c = http.client.HTTPConnection("127.0.0.1", 42000, timeout=5); c.request("GET", "/openapi.json"); status = c.getresponse().status; print(status); raise SystemExit(status != 200)'
```

Expect HTTP 200. Stop the listener with Ctrl+C and record clean shutdown. Then run the supported verification commands.

```sh
./scripts/dev test -- tests/test_dev_environment.py::test_real_loopback_api_lifecycle --tb=short -rs
./scripts/dev verify-direct
./scripts/dev verify-production
sh scripts/tests/test_dev_venv.sh
./scripts/dev test -- --tb=short -rs
git status --short --branch
```

Require the focused lifecycle test to pass without a skip. It exercises login, refresh, invalid credential refusal, authenticated status, streaming and shutdown over a real socket. Require `Synthetic production loopback HTTP passed` from the guarded synthetic compatibility verifier. Its in process fallback does not close the listener gate. Record the architecture, Python and SQLite versions, tested SHA, command outcomes, skip counts and reasons. The historical macOS results are comparison evidence. Linux test counts have yet to be measured.

## Restrictions and exact next action

Use synthetic state and checked in source. Keep canonical main, scratch work and the Pi untouched. Retain normal Git guards. No push, cloud session connection, production access or deployment was performed. Nova, CI, cloud orchestration, hardware diagnostics and roadmap features remain separate scopes.

Next action: the owner verifies remote trigger settings and authorizes a redacted local history scan. After those gates are resolved, run the Linux sequence above at the approved revision. Cross platform acceptance remains open. The independent review of `59f7627` and `13cb8ac` is complete.
