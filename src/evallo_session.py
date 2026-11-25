"""The saved Evallo login, and how the Evallo script tells Lethanial things.

The login is data/evallo_state.json, written by scripts/evallo_login.py. It
holds Evallo's token ("evalloToken" in the site's localStorage), which Evallo
issues for 30 days. token_expiry() reads the date out of the token itself, so
the canary can warn before it runs out rather than after.

notify() sends a push to Nova through push.py, the same path reminders use.
Where push is not set up (the Mac, or the Pi without its Apple key) it prints
instead, so a message is never silently dropped.
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "evallo_state.json"
WARN_DAYS = 3


class LoginExpired(Exception):
    """The saved login is missing or Evallo sent the script to sign in."""


def load_state(path=STATE):
    if not path.exists():
        raise LoginExpired(f"No saved Evallo login at {path}.")
    return json.loads(path.read_text())


def token_expiry(state):
    """When the saved token runs out, as an aware UTC datetime, or None when
    there is no readable token. Reads the token's own "exp" claim; nothing is
    verified, since only Evallo can do that."""
    for origin in state.get("origins", []):
        for item in origin.get("localStorage", []):
            if item.get("name") == "evalloToken":
                try:
                    payload = item["value"].split(".")[1]
                    payload += "=" * (-len(payload) % 4)
                    exp = json.loads(base64.urlsafe_b64decode(payload))["exp"]
                    return datetime.fromtimestamp(exp, tz=timezone.utc)
                except Exception:
                    return None
    return None


def days_left(state, now=None):
    exp = token_expiry(state)
    if exp is None:
        return None
    return (exp - (now or datetime.now(timezone.utc))).total_seconds() / 86400


def login_expired(job):
    """The saved login stopped working: one push a day, through the watchdog,
    with the login command and the job that failed. Returns whether it pushed."""
    try:
        import agent_watchdog
        pushed = agent_watchdog.login_expired("evallo", job)
    except Exception as exc:
        print(f"(login alert unavailable: {type(exc).__name__}: {exc})", flush=True)
        pushed = False
    print(f"LOGIN EXPIRED: Evallo, during {job}{'' if pushed else ' (already pushed today)'}",
          flush=True)
    return pushed


def notify(body, title="Evallo"):
    """A push to Nova, or a printed line where push is unavailable. Returns
    whether a phone accepted it."""
    try:
        import push
        accepted = push.notify(title, body, kind="evallo")
    except Exception as exc:   # not on the Pi, or push not configured
        accepted = 0
        print(f"(push unavailable: {type(exc).__name__})", flush=True)
    print(f"NOTIFY{'' if accepted else ' (not delivered)'}: {body}", flush=True)
    return bool(accepted)
