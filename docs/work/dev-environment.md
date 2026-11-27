# MILES development environment handoff

## Objective and baseline

Deliver the [approved headless development contract](../DEV_ENVIRONMENT_CONTRACT.md#acceptance-criteria) with reproducible dependencies, synthetic backend state, generated credentials, an offline suite, and a portable repository workflow. The work is on `infra/dev-environment`, based on canonical `main` at `51676b912eac708d03024d02942ba05364ba89e6`. The implementation resumed from `eddb502` after the owner installed the dependencies. There is no paired Nova revision for this phase.

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

Start from [AGENTS.md](../../AGENTS.md), [the development commands](../DEVELOPMENT.md), and [the subsystem map](../REPOSITORY_MAP.md). The runtime boundary, isolated entry points, and acceptance checks are in [dev_guards.py](../../scripts/dev_guards.py), [dev_runtime.py](../../scripts/dev_runtime.py), and [test_dev_environment.py](../../src/tests/test_dev_environment.py).

## Verification

macOS arm64 with CPython 3.13.1 and SQLite 3.45.3. The implementation revision under test is `59f7627fefcbbeb89d46655b983856f9cfd8d3d2`. The production unit and dependency manifests remain in their existing locations. The validation used only synthetic configuration and databases.

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

- **High, filesystem boundary.** [SQLite target validation and native open](../../scripts/dev_guards.py#L139) uses an owner private directory, descriptor rooted validation, and the worktree lifetime lock. If a separate process running as the same user replaces a file or sidecar between validation and SQLite's native open, the current Python guard cannot guarantee race prevention. Supported launchers coordinate through the lock. Static links, foreign paths, and cooperating concurrent resets are covered. The unqualified race prevention requirement in C2 remains open.
- **Medium, live loopback evidence.** [The acceptance listener](../../scripts/dev_runtime.py#L210) uses the designated address and port. This runner rejects local bind. Real HTTP and listener shutdown still need a runner that allows it. In process HTTP, event streams, WebSockets and the staged production lifecycle have separate coverage.
- **Medium, Linux evidence.** The lock describes Linux x86_64 and aarch64. There was no authorized Linux runner in this session. Clean setup, suite, and startup remain unverified there.

## Restrictions and exact next action

Use synthetic state and checked in source. Keep canonical main, the scratch work, and the Pi untouched. Retain normal Git guards. Do not push or deploy. Nova, CI, cloud orchestration, hardware diagnostics, and roadmap features have separate scopes.

Run the following from the infrastructure worktree on a supported host that permits IPv4 loopback binds.

```sh
./scripts/dev bootstrap && ./scripts/dev verify-production && ./scripts/dev test -- --tb=short -rs
```

The production compatibility check should report successful real loopback HTTP and the headless listener test should pass. Record Linux separately. If requesting independent implementation review, use revision `59f7627fefcbbeb89d46655b983856f9cfd8d3d2`, implementation range `eddb502..59f7627`, the files touched by that range, and the approved contract. No independent implementation review ran in this session.
