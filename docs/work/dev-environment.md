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

## Open acceptance items

- **Owner acceptance pending for the SQLite same user race.** The owner private root, descriptor rooted validation, and lifetime lock protect the supported workflow. A separate process running as the same user can replace a file or sidecar between validation and the native SQLite open. Acceptance of this documented C2 limitation remains with the owner.
- **Live loopback verification pending.** The runner denied the assigned loopback bind. Real HTTP and listener shutdown still need a runner that permits the bind. In process HTTP, event streams, WebSockets, and the staged production lifecycle passed.
- **Linux verification pending.** Clean setup, the suite, and startup still need a supported Linux runner.

## Restrictions and exact next action

Use synthetic state and checked in source. Keep canonical main, the scratch work, and the Pi untouched. Retain normal Git guards. Do not push or deploy. Nova, CI, cloud orchestration, hardware diagnostics, and roadmap features have separate scopes.

Run the following from the infrastructure worktree on a supported host that permits IPv4 loopback binds.

```sh
./scripts/dev bootstrap && ./scripts/dev verify-production && ./scripts/dev test -- --tb=short -rs
```

Record real loopback HTTP and listener shutdown, and record Linux results separately. Acceptance remains on hold pending that evidence and the owner's decision on the SQLite limitation. The independent review of `59f7627` and `13cb8ac` is complete.
