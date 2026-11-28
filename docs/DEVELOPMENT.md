# Headless development

Use Python 3.13 and uv on macOS arm64 or Linux x86_64 or aarch64. SQLite must include FTS5. Each worktree has its own virtual environment, database, and generated credentials. The launcher requires a real `.venv` directory and accepts interpreter symlinks inside it.

```sh
./scripts/dev bootstrap
./scripts/dev init
./scripts/dev seed
./scripts/dev start
```

Bootstrap installs from [uv.lock](../uv.lock), checks the installed Python and SQLite, and requires access to the package source if artifacts are absent from the uv cache. An existing install can be checked with `./scripts/dev check`. Regenerate the lock when the headless manifest changes.

```sh
uv lock --no-config --no-python-downloads --python "$(command -v python3.13)" --default-index https://pypi.org/simple
```

Use the loopback URL in the startup message. The default port is derived from the worktree path. If it is already in use, choose an unused port with `./scripts/dev start --port 42000`. One process holds the worktree lock for the entire API lifetime. `./scripts/dev reset` refuses while the API or another development command holds the lock. Reset recreates the blank schema and preserves the generated login and capture credentials. Run seed again to restore the sample records.

The launcher creates the ignored `.miles-dev/credentials.json` with mode 0600. A local client can read the `password` field for `POST /auth/login` and the `capture_key` field for `POST /capture`. Keep values in the file or in the client's memory. Use the returned bearer token for `POST /auth/refresh` and ordinary authenticated routes.

The launcher clears inherited environment settings before loading application modules, establishes a private state directory and lock, and installs SQLite, network, and subprocess guards. Chat and sessions use deterministic model responses. Headless WebSocket chat requires a full login and dispatches the synthetic turn on a worker thread. Calendar has synthetic appointments, a conflict, and a tagged Evallo event. The same calendar write protection and verification code applies. The seeded list, task, reminder, session, and capture data use the canonical API. Voice synthesis returns HTTP 502, and Pi status readings return null. Live Calendar OAuth, Apple services, Oura, Hevy, Prowl, Deepgram, browser automation, and external Mac jobs are unavailable. Push delivers to zero devices. In development, worker claim and retry callbacks return no jobs. Voice, schedulers, health operators, and deployment tools have separate entry points.

The Python transport policy allows TCP on the selected IPv4 loopback address and port, plus anonymous local socket pairs for Python event loops. It rejects other destinations and DNS lookups and records the category of each refusal. The test run fails on unexpected refusals, including attempts caught by application code. The private directory and worktree lock coordinate filesystem operations made through this workflow. SQLite opens its database and sidecars through its native path APIs after Python validates them. A cooperating worktree follows the lock. Concurrent file replacement by a separate process running as the same user remains outside that protection. Native code and independently launched processes have their own execution boundaries.

## Offline verification

```sh
./scripts/dev test
./scripts/dev test -- tests/test_evallo_filter.py tests/test_dev_environment.py
./scripts/dev verify-direct
./scripts/dev verify-production
sh scripts/tests/test_dev_venv.sh
```

The test launcher creates a temporary private root, installs guards before pytest and its plugins, and verifies that an explicit early plugin imports the app safely. For a focused run, pass pytest arguments after `--` with paths relative to `src/`. Test databases are separate from development state. The isolated launcher replaces the historical direct command `cd src && python3 -m pytest tests/ -q`. The headless suite omits enrollment audio and mic channel modules because they require PyAudio and SciPy. Live voice tests also honor their hardware prerequisites. A runner that forbids loopback bind skips socket tests and still runs the in process HTTP and WebSocket coverage.

The production verification stages the checked in Uvicorn application target and a generated dotenv file inside disposable synthetic state. It exercises the default production selector with a repository virtual environment prefix and guards active. It checks migration from 44 to 45, row preservation, startup and shutdown, clearing the Evallo active flag on restart, and existing WebSocket dispatch using a synthetic chat callback. Production operator procedures are in [Infrastructure](INFRASTRUCTURE.md).

See the [navigation map](REPO_MAP.md) for the relevant sources and tests. See the [development environment contract](DEV_ENVIRONMENT_CONTRACT.md) for the acceptance criteria.
