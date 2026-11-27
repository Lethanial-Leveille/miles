"""Filesystem and transport boundary for the supported Python runners."""

import fcntl
import os
from pathlib import Path
import socket
import sqlite3
import stat
import subprocess
import sys
from contextlib import contextmanager


DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_NOFOLLOW | os.O_NONBLOCK
SQLITE_SUFFIXES = ("", "-journal", "-wal", "-shm")


class BoundaryError(RuntimeError):
    pass


@contextmanager
def directory(path, *, create=False, private=False):
    """Walk with directory descriptors so a swapped ancestor cannot redirect opens."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise BoundaryError("Unsafe state directory")
    fd = os.open("/", DIR_FLAGS)
    try:
        for index, part in enumerate(path.parts[1:], 2):
            leaf = index == len(path.parts)
            if leaf and create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
            info = os.fstat(fd)
            if info.st_uid not in {0, os.getuid()}:
                raise BoundaryError("Untrusted state directory owner")
            if info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX):
                raise BoundaryError("Writable state directory ancestor")
        if private:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise BoundaryError("State directory must be owned by this user with mode 0700")
        yield fd
    finally:
        os.close(fd)


def check_file(info, *, secret=False):
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
            or info.st_nlink != 1 or info.st_mode & 0o022
            or secret and stat.S_IMODE(info.st_mode) != 0o600):
        raise BoundaryError("Unsafe state file")


def inspect_file(fd, name):
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    check_file(info)


class State:
    def __init__(self, root, fd, mode, port):
        self.root = root
        self.fd = fd
        self.mode = mode
        self.port = port
        self.db_path = root / "miles.db"
        self.violations = []
        self.locked = False

    def deny(self, kind):
        self.violations.append(kind)
        print(f"MILES isolation refused {kind}", file=sys.stderr, flush=True)
        raise BoundaryError(f"MILES isolation refused {kind}")

    def check_anchor(self, relpath, fd):
        expected = os.fstat(fd)
        with directory(self.root / relpath, private=True) as current:
            actual = os.fstat(current)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise BoundaryError("State directory changed")

    @contextmanager
    def file(self, name, flags):
        if name not in {"runtime.lock", "credentials.json", "credentials.pending"}:
            raise BoundaryError("Unknown protected state file")
        fd = os.open(name, flags | FILE_FLAGS, mode=0o600, dir_fd=self.fd)
        try:
            check_file(os.fstat(fd), secret=True)
            yield fd
        finally:
            os.close(fd)

    @contextmanager
    def credentials_writer(self):
        # An interrupted writer leaves an unpublished file. Validate it before
        # unlinking and install the replacement only after the caller syncs it.
        pending = "credentials.pending"
        inspect_file(self.fd, pending)
        try:
            os.unlink(pending, dir_fd=self.fd)
        except FileNotFoundError:
            pass
        with self.file(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL) as fd:
            try:
                yield fd
                inspect_file(self.fd, "credentials.json")
                os.replace(pending, "credentials.json", src_dir_fd=self.fd, dst_dir_fd=self.fd)
            finally:
                try:
                    os.unlink(pending, dir_fd=self.fd)
                except FileNotFoundError:
                    pass

    @contextmanager
    def exclusive(self):
        with self.file("runtime.lock", os.O_RDWR | os.O_CREAT) as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BoundaryError("State is in use by another MILES operation") from None
            try:
                self.check_anchor(Path("."), self.fd)
                self.locked = True
                yield
            finally:
                self.locked = False

    @contextmanager
    def db_target(self, path):
        raw = os.fsdecode(os.fspath(path))
        path = Path(raw)
        if (not self.locked or raw != str(path) or not path.is_absolute()
                or ".." in path.parts or not path.is_relative_to(self.root)
                or path.suffix not in {".db", ".sqlite", ".sqlite3"}):
            raise BoundaryError("SQLite target is outside the assigned state or lock")
        relative = path.relative_to(self.root)
        if self.mode == "test" and relative.parts[0] == "miles":
            raise BoundaryError("SQLite cannot use the production-shaped path in test mode")
        if relative.parent != Path(".") and self.mode not in {"test", "compat"}:
            raise BoundaryError("Development SQLite must stay at the state root")
        if self.mode == "dev" and path != self.db_path:
            raise BoundaryError("Unknown development SQLite target")
        self.check_anchor(Path("."), self.fd)
        with directory(path.parent) as fd:
            parent = os.fstat(fd)
            if parent.st_uid != os.getuid() or parent.st_mode & 0o022:
                raise BoundaryError("Unsafe SQLite parent")
            for suffix in SQLITE_SUFFIXES:
                inspect_file(fd, path.name + suffix)
            yield fd, path.name

    def reset_database(self):
        with self.db_target(self.db_path) as (fd, name):
            for suffix in SQLITE_SUFFIXES:
                try:
                    os.unlink(name + suffix, dir_fd=fd)
                except FileNotFoundError:
                    pass

    def protect_files(self, checkout):
        """Keep supported Python operations out of another runtime namespace."""
        private_checkout = checkout / "data"
        temporary_parent = Path("/private/tmp" if sys.platform == "darwin" else "/tmp")

        def check(value, directory_fd=None):
            if not isinstance(value, (str, bytes)):
                return
            path = Path(os.fsdecode(value))
            if not path.is_absolute() and directory_fd not in {None, -1}:
                # State's relative descriptor operations already validate the
                # anchor and each file without following links.
                return
            path = Path(os.path.abspath(path))
            for candidate in (path, *path.parents):
                if (candidate.name == ".miles-dev"
                        or candidate.parent == temporary_parent
                        and candidate.name.startswith("miles-isolated-runtime-")):
                    if candidate != self.root:
                        self.deny("foreign runtime state")
                    break
            if path.is_relative_to(private_checkout) or (
                    path.parent == checkout and path.name.startswith(".env")):
                self.deny("checkout private data")

        def audit(event, args):
            if event in {"open", "os.listdir", "os.scandir", "os.chdir"}:
                check(args[0])
            elif event in {"os.remove", "os.rmdir"}:
                check(args[0], args[1])
            elif event in {"os.mkdir", "os.chmod"}:
                check(args[0], args[2])
            elif event in {"os.rename", "os.link"}:
                check(args[0], args[2])
                check(args[1], args[3])
            elif event == "os.symlink":
                check(args[1], args[2])

        sys.addaudithook(audit)

    def install_sqlite(self):
        connect = sqlite3.connect

        def authorize(action, _first, _second, _db, _source):
            if action == sqlite3.SQLITE_ATTACH:
                self.deny("SQLite attachment")
            return sqlite3.SQLITE_OK

        def audit(event, args):
            if event == "sqlite3.connect" and args[0] != ":memory:":
                try:
                    with self.db_target(args[0]):
                        pass
                except (BoundaryError, OSError, TypeError):
                    self.deny("SQLite target")

        def guarded(database, *args, **kwargs):
            if kwargs.get("uri"):
                self.deny("SQLite URI")
            if database == ":memory:":
                conn = connect(database, *args, **kwargs)
            else:
                # Keep the validated parent descriptor open across the native
                # call. The lifetime lock covers every supported writer.
                try:
                    with self.db_target(database):
                        conn = connect(database, *args, **kwargs)
                except (BoundaryError, OSError, TypeError):
                    self.deny("SQLite target")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.set_authorizer(authorize)
            return conn

        sys.addaudithook(audit)
        sqlite3.connect = sqlite3.dbapi2.connect = guarded


def install_transports(state):
    """Guard Python sockets and subprocess APIs before any third-party imports."""
    original = socket.socket
    getaddrinfo = socket.getaddrinfo

    def endpoint(address):
        return address == ("127.0.0.1", state.port) and state.port is not None

    class GuardedSocket(original):
        def connect(self, address):
            if not endpoint(address) or self.family != socket.AF_INET or self.type != socket.SOCK_STREAM:
                state.deny("socket connection")
            return super().connect(address)

        def connect_ex(self, address):
            if not endpoint(address) or self.family != socket.AF_INET or self.type != socket.SOCK_STREAM:
                state.deny("socket connection")
            return super().connect_ex(address)

        def bind(self, address):
            if not endpoint(address) or self.family != socket.AF_INET or self.type != socket.SOCK_STREAM:
                state.deny("socket bind")
            return super().bind(address)

        def sendto(self, *args):
            state.deny("datagram send")

        if hasattr(original, "sendmsg"):
            def sendmsg(self, *args, **kwargs):
                state.deny("socket message send")

    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        if state.mode == "compat" and host == "localhost":
            host = "127.0.0.1"
        if not endpoint((host, port)):
            state.deny("DNS lookup")
        return getaddrinfo(host, port, family, type, proto,
                           flags | socket.AI_NUMERICHOST | socket.AI_NUMERICSERV)

    def deny_dns(*args, **kwargs):
        state.deny("DNS lookup")

    def audit(event, args):
        if event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                     "os.fork", "os.forkpty", "os.startfile"}:
            state.deny("process launch")
        # Catch calls using a reference captured before socket.socket was wrapped.
        if event in {"socket.connect", "socket.bind"}:
            sock, address = args
            if not endpoint(address) or sock.family != socket.AF_INET or sock.type != socket.SOCK_STREAM:
                state.deny("socket operation")
        if event == "socket.getaddrinfo" and not endpoint(args[:2]):
            state.deny("DNS lookup")
        if event in {"socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo",
                     "socket.sendto", "socket.sendmsg"}:
            state.deny("socket lookup or send")

    # The profile has only IPv4. Libraries use this flag to avoid IPv6 bind probes.
    socket.has_ipv6 = False
    socket.socket = GuardedSocket
    socket.getaddrinfo = resolve
    socket.gethostname = lambda: "miles-local"
    def fqdn(name=""):
        if name in {"", "127.0.0.1"}:
            return "127.0.0.1"
        state.deny("DNS lookup")

    socket.getfqdn = fqdn
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        setattr(socket, name, deny_dns)
    sys.addaudithook(audit)
