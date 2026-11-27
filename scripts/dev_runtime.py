"""Repository launcher. Install isolation before loading application or test code."""

import argparse
import asyncio
from contextlib import closing, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT / "scripts"))

from dev_guards import BoundaryError, State, directory, install_transports


def fts_check():
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
    print(f"Python {sys.version.split()[0]}, SQLite {sqlite3.sqlite_version}, FTS5 available")


def port_number(value):
    port = int(value)
    if not 1024 <= port <= 65535:
        raise argparse.ArgumentTypeError("Choose a local port between 1024 and 65535")
    return port


def worktree_port():
    return 20000 + int.from_bytes(hashlib.sha256(os.fsencode(ROOT)).digest()[:4], "big") % 30000


def site_packages():
    if sys.version_info[:2] != (3, 13):
        raise BoundaryError("The launcher requires Python 3.13")
    venv = ROOT / ".venv"
    if Path(sys.executable) != venv / "bin" / "python":
        raise BoundaryError("The launcher requires the worktree virtual environment")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(venv / "lib" / "python3.13" / "site-packages"))


def credentials(state):
    import bcrypt

    try:
        with state.file("credentials.json", os.O_RDONLY) as fd:
            with os.fdopen(os.dup(fd)) as stream:
                result = json.load(stream)
    except FileNotFoundError:
        password = secrets.token_urlsafe(24)
        result = {
            "password": password,
            "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            "jwt_secret": secrets.token_urlsafe(48),
            "capture_key": "mcap_" + secrets.token_urlsafe(32),
        }
        with state.credentials_writer() as fd:
            with os.fdopen(os.dup(fd), "w") as stream:
                json.dump(result, stream)
                stream.flush()
                os.fsync(stream.fileno())
    if (not isinstance(result, dict)
            or set(result) != {"password", "password_hash", "jwt_secret", "capture_key"}
            or not all(isinstance(value, str) and value for value in result.values())):
        raise BoundaryError("Invalid local credentials file")
    return result


def activate(state, source=SRC):
    os.environ.clear()
    os.environ.update(
        HOME=str(state.root),
        TMPDIR=str(state.root),
        TZ="America/New_York",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    os.umask(0o077)
    time.tzset()
    sys.path.insert(0, str(source))
    import runtime_profile

    runtime_profile.configure(state)
    state.protect_files(ROOT)
    state.install_sqlite()
    install_transports(state)
    os.chdir(source)
    if state.mode == "compat":
        return
    os.environ["MILES_MODE"] = state.mode
    os.environ["MILES_DB_PATH"] = str(state.db_path)
    os.environ["MILES_CALENDAR_ROLES_FILE"] = str(state.root / "missing-calendar-roles.json")
    state.auth = credentials(state)
    return state.auth


def seed(auth):
    from capture import hash_key
    import database

    now = datetime.now()
    groceries = database.get_or_create_list("groceries", now=now)
    with closing(sqlite3.connect(database.DB_PATH)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        stamp = now.isoformat()
        conn.execute(
            "INSERT INTO list_items (list_id, text, created_at) SELECT ?, ?, ? "
            "WHERE NOT EXISTS (SELECT 1 FROM list_items WHERE list_id = ? AND text = ?)",
            (groceries, "Sample oranges", stamp, groceries, "Sample oranges"),
        )
        conn.execute(
            "INSERT INTO reminders (content, created_at) SELECT ?, ? "
            "WHERE NOT EXISTS (SELECT 1 FROM reminders WHERE content = ?)",
            ("Sample reminder", stamp, "Sample reminder"),
        )
        key = conn.execute("SELECT id FROM capture_keys WHERE name = ?", ("Synthetic capture",)).fetchone()
        if key is None:
            key_id = conn.execute(
                "INSERT INTO capture_keys (name, key_hash, created_at) VALUES (?, ?, ?)",
                ("Synthetic capture", hash_key(auth["capture_key"]), stamp),
            ).lastrowid
        else:
            key_id = key[0]
        captured = conn.execute(
            "SELECT 1 FROM capture_log WHERE key_id = ? AND source = ? AND text = ?",
            (key_id, "synthetic", "Sample task"),
        ).fetchone()
        if captured is None:
            task_id = conn.execute(
                "INSERT INTO tasks (text, created_at) VALUES (?, ?)", ("Sample task", stamp),
            ).lastrowid
            conn.execute(
                "INSERT INTO capture_log (created_at, key_id, source, text, kind, ref_id, message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (stamp, key_id, "synthetic", "Sample task", "task", task_id, "Added to backlog: Sample task"),
            )
        existing = conn.execute("SELECT 1 FROM sessions WHERE type = ?", ("nightly_review",)).fetchone()
    if not existing:
        database.create_session("nightly_review", now.date().isoformat(), "synthetic",
                                opening="Sample review.", now=now)
    print("Synthetic fixtures ready")


def test(state, args):
    import pytest

    class Isolation:
        def pytest_sessionfinish(self, session):
            if state.violations:
                session.exitstatus = pytest.ExitCode.TESTS_FAILED
                print(f"MILES isolation recorded {len(state.violations)} refused operation(s)")

    command = ["-q", "-p", "no:cacheprovider", "-p", "dev_early_plugin",
               "--ignore=tests/test_enrollment_audio.py", "--ignore=tests/test_mic_channel.py", *args,
               "--basetemp", str(state.root / "pytest")]
    return int(pytest.main(command, plugins=[Isolation()]))


def verify_direct(state):
    """Invoke unsupported pytest with an early plugin and a synthetic sentinel."""
    sentinel = state.root / "miles.db"
    with closing(sqlite3.connect(sentinel)) as conn, conn:
        conn.execute("CREATE TABLE sentinel (value TEXT)")
        conn.execute("INSERT INTO sentinel VALUES ('synthetic')")
    fingerprint = hashlib.sha256(sentinel.read_bytes()).digest()

    install_transports(state)
    os.environ.update(HOME=str(state.root), MILES_DB_PATH=str(sentinel),
                      PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    sys.path.insert(0, str(SRC))
    os.chdir(SRC)
    import pytest

    output = io.StringIO()
    with redirect_stdout(output), redirect_stderr(output):
        try:
            result = pytest.main(["-p", "dev_early_plugin", "--collect-only", "-q"])
        except (ImportError, RuntimeError) as exc:
            if "Use the isolated MILES launcher" not in str(exc):
                raise BoundaryError("Unexpected early pytest import failure") from None
        else:
            if result == 0 or "Use the isolated MILES launcher" not in output.getvalue():
                raise BoundaryError("Direct pytest did not fail at the runtime boundary")
    if hashlib.sha256(sentinel.read_bytes()).digest() != fingerprint:
        raise BoundaryError("Direct pytest altered the synthetic sentinel")
    if state.violations:
        raise BoundaryError("Direct pytest attempted a forbidden operation")
    print("Direct pytest refused its early app import, and the sentinel is intact")


class LoopbackUnavailable(RuntimeError):
    pass


@contextmanager
def serve_loopback(config):
    """Exercise a designated Uvicorn listener without touching any existing service."""
    import uvicorn

    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", config.port))
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EACCES}:
                raise LoopbackUnavailable("Runner denies designated loopback binds") from None
            raise
        api = uvicorn.Server(config)
        background = threading.Thread(target=api.run, kwargs={"sockets": [listener]}, daemon=True)
        background.start()
        try:
            deadline = time.monotonic() + 5
            while not api.started and background.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            if not api.started:
                raise BoundaryError("Synthetic Uvicorn listener did not start")
            yield f"http://127.0.0.1:{config.port}"
        finally:
            api.should_exit = True
            background.join(5)
            if background.is_alive():
                raise BoundaryError("Synthetic Uvicorn listener did not stop")


async def lifespan(config):
    from uvicorn.lifespan.on import LifespanOn

    server = LifespanOn(config)
    await server.startup()
    if server.should_exit:
        raise BoundaryError("Synthetic production startup failed")
    await server.shutdown()
    if server.error_occurred or server.shutdown_failed:
        raise BoundaryError("Synthetic production shutdown failed")


def verify_production(state):
    source = state.root / "src"
    source.mkdir(mode=0o700)
    for module in SRC.glob("*.py"):
        shutil.copyfile(module, source / module.name)
    state.db_path = state.root / "miles" / "data" / "miles.db"
    state.db_path.parent.mkdir(parents=True, mode=0o700)
    activate(state, source)
    local_auth = credentials(state)
    with (state.root / ".env").open("x") as env:
        env.write(f'MILES_PASSWORD_HASH={local_auth["password_hash"]}\n')
        env.write(f'MILES_JWT_SECRET={local_auth["jwt_secret"]}\n')
        env.write("ANTHROPIC_API_KEY=synthetic-placeholder\n")
    import auth
    import database
    import uvicorn

    if "MILES_MODE" in os.environ or database.DB_PATH != str(state.db_path):
        raise BoundaryError("Synthetic production startup selected an unexpected configuration")
    if not auth.verify_password(local_auth["password"]):
        raise BoundaryError("Synthetic production dotenv authentication failed")
    migrations = database.MIGRATIONS
    database.MIGRATIONS = migrations[:-1]
    database.init_db()
    database.MIGRATIONS = migrations
    database.add_task("Synthetic prior schema task")
    config = uvicorn.Config("server:app", host="localhost", port=state.port, lifespan="on",
                            access_log=False)
    try:
        with serve_loopback(config) as url:
            import httpx

            with httpx.Client(base_url=url, trust_env=False) as client:
                login = client.post("/auth/login", json={"password": local_auth["password"]})
                if login.status_code != 200:
                    raise BoundaryError("Synthetic production HTTP login failed")
                headers = {"Authorization": "Bearer " + login.json()["access_token"]}
                if client.get("/status", headers=headers).status_code != 200:
                    raise BoundaryError("Synthetic production authenticated route failed")
        print("Synthetic production loopback HTTP passed")
    except LoopbackUnavailable:
        print("Synthetic production loopback unavailable. Testing the Uvicorn lifecycle in process.")
        asyncio.run(lifespan(config))
    with closing(sqlite3.connect(state.db_path)) as conn, conn:
        if (conn.execute("SELECT version FROM schema_version").fetchall() != [(migrations[-1][0],)]
                or conn.execute("SELECT text FROM tasks").fetchall() != [("Synthetic prior schema task",)]
                or conn.execute("PRAGMA journal_mode").fetchone() != ("wal",)
                or conn.execute("PRAGMA integrity_check").fetchone() != ("ok",)):
            raise BoundaryError("Synthetic production migration check failed")
        conn.execute(
            "INSERT INTO evallo_verification VALUES (1, ?, ?, ?, ?, 1, 1)",
            ("2026-01-01T12:00:00Z", "2026-01-01T12:00:00Z", "2026-01-01", "2026-01-02"),
        )
        conn.execute("INSERT INTO evallo_verified_sessions VALUES (?)", ("a" * 24,))
    asyncio.run(lifespan(config))
    with closing(sqlite3.connect(state.db_path)) as conn:
        if (conn.execute("SELECT active FROM evallo_verification").fetchall() != [(0,)]
                or conn.execute("SELECT count(*) FROM evallo_verified_sessions").fetchone() != (1,)):
            raise BoundaryError("Synthetic production startup did not clear the Evallo active flag")
    if state.violations:
        raise BoundaryError("Synthetic production startup attempted a forbidden operation")
    print("Synthetic production target passed dotenv auth, migration, restart, and Evallo reset")



def run(args):
    site_packages()
    if args.command == "check":
        fts_check()
        return 0
    os.umask(0o077)
    if args.command in {"test", "verify-direct", "verify-production"}:
        # Ignore inherited TMPDIR, including aliases such as macOS /tmp.
        parent = Path("/private/tmp" if sys.platform == "darwin" else "/tmp")
        with directory(parent), tempfile.TemporaryDirectory(prefix="miles-isolated-runtime-", dir=parent) as tmp:
            root = Path(tmp)
            with directory(root, private=True) as fd:
                mode = "compat" if args.command == "verify-production" else "test"
                state = State(root, fd, mode, args.port)
                with state.exclusive():
                    if args.command == "verify-production":
                        verify_production(state)
                    elif args.command == "verify-direct":
                        verify_direct(state)
                    else:
                        activate(state)
                        return test(state, args.pytest_args)
        return 0

    root = ROOT / ".miles-dev"
    with directory(root, create=True, private=True) as fd:
        state = State(root, fd, "dev", args.port)
        with state.exclusive():
            auth = activate(state)
            if args.command == "reset":
                state.reset_database()
            if args.command == "start":
                import uvicorn

                uvicorn.run("server:app", host="127.0.0.1", port=args.port, workers=1,
                            access_log=False)
            elif args.command == "seed":
                with state.db_target(state.db_path) as (db_dir, db_name):
                    try:
                        os.stat(db_name, dir_fd=db_dir, follow_symlinks=False)
                    except FileNotFoundError:
                        raise BoundaryError("Run ./scripts/dev init before seeding") from None
                seed(auth)
            else:
                import database

                database.init_db()
                print("Development schema ready")
        return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "init", "seed", "reset", "start", "test", "verify-direct", "verify-production"):
        command = sub.add_parser(name)
        if name == "start":
            command.add_argument("--port", type=port_number, default=worktree_port())
        elif name == "test":
            command.add_argument("--port", type=port_number, default=50000 + secrets.randbelow(12000))
        else:
            command.set_defaults(port=8000 if name == "verify-production" else None)
    args, tail = parser.parse_known_args()
    if args.command != "test" and tail:
        parser.error("Unexpected arguments")
    args.pytest_args = tail[1:] if tail[:1] == ["--"] else tail
    try:
        return run(args)
    except (BoundaryError, OSError, json.JSONDecodeError) as exc:
        # An OS or parser error can include a path or submitted content.
        print(str(exc) if isinstance(exc, BoundaryError)
              else f"MILES launcher refused the operation ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
