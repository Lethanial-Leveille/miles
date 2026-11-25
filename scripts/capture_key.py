#!/usr/bin/env python3
"""Make, list and revoke quick capture keys (see src/capture.py).

    python3 scripts/capture_key.py new mac_hotkey   prints the key once
    python3 scripts/capture_key.py list
    python3 scripts/capture_key.py revoke 3

A key can only add tasks and list items through POST /capture. It is printed
once and stored only as a hash, so a lost key is revoked and replaced, never
recovered.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import capture  # noqa: E402
from database import add_capture_key, capture_keys, init_db, revoke_capture_key  # noqa: E402

init_db()
command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "new" and len(sys.argv) == 3:
    key, key_hash = capture.new_key()
    key_id = add_capture_key(sys.argv[2], key_hash)
    print(f"Key {key_id} ({sys.argv[2]}). Shown once; paste it into the Shortcut now:\n{key}")
elif command == "list":
    for key_id, name, created, used, revoked in capture_keys():
        state = f"revoked {revoked[:16]}" if revoked else f"last used {(used or 'never')[:16]}"
        print(f"{key_id}  {name:<16} created {created[:16]}  {state}")
elif command == "revoke" and len(sys.argv) == 3:
    print("Revoked." if revoke_capture_key(int(sys.argv[2])) else "No active key with that id.")
else:
    print(__doc__)
    sys.exit(1)
