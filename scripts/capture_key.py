#!/usr/bin/env python3
"""Make, list and revoke quick capture keys (see src/capture.py).

    python3 scripts/capture_key.py new mac_hotkey   writes the key to a private file
    python3 scripts/capture_key.py new mac_hotkey --path /some/file
    python3 scripts/capture_key.py list
    python3 scripts/capture_key.py revoke 3

A key can only add tasks and list items through POST /capture. It is stored in
the database only as a hash, so a lost key is revoked and replaced, never
recovered. A new key is written once to a file only he can read (mode 600,
default ~/.miles-keys/capture_<name>.key) and is never printed, so it never
lands in a terminal log, a chat or shell history; the script prints only the
key's id and the path.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import capture  # noqa: E402
from database import add_capture_key, capture_keys, init_db, revoke_capture_key  # noqa: E402


def write_key_file(path, key):
    """Create the file with mode 600 from the start (os.open with 0o600), so
    there is no moment anyone else can read it. Refuses to overwrite one."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")


init_db()
command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "new" and len(sys.argv) in (3, 5):
    name = sys.argv[2]
    path = os.path.expanduser(sys.argv[4] if len(sys.argv) == 5 and sys.argv[3] == "--path"
                              else f"~/.miles-keys/capture_{name}.key")
    key, key_hash = capture.new_key()
    # The file first: if it cannot be written, no key row is left behind that
    # nobody holds.
    write_key_file(path, key)
    key_id = add_capture_key(name, key_hash)
    print(f"Key {key_id} ({name}) written to {path} (mode 600). Copy it from there.")
elif command == "list":
    for key_id, name, created, used, revoked in capture_keys():
        state = f"revoked {revoked[:16]}" if revoked else f"last used {(used or 'never')[:16]}"
        print(f"{key_id}  {name:<16} created {created[:16]}  {state}")
elif command == "revoke" and len(sys.argv) == 3:
    print("Revoked." if revoke_capture_key(int(sys.argv[2])) else "No active key with that id.")
else:
    print(__doc__)
    sys.exit(1)
