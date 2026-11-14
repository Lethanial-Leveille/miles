"""Push notifications to his phone, through Apple's push service (APNs).

Until now everything Nova had to say was said in his room. A reminder that came
due while he was at class was queued for a speaker nobody was near, and the poller
could not tell: alerts.py delivers to the room and nowhere else. This is the
other way out of the building.

How a push travels:

  1. The app asks iOS for a device token and registers it at POST /push/register.
     It is a string that names this phone, for this app, on one APNs host.
  2. Something here calls notify(). It signs a short lived JWT with the .p8 key
     Apple issued, and POSTs the notification to that host over HTTP/2, which
     is the only transport APNs speaks.
  3. APNs hands it to the phone, or holds it until the phone is reachable.

Two details that cost people an afternoon, so they are written down:

  The host depends on how the build was signed. A build run from Xcode talks to
  the sandbox host and an App Store or TestFlight build talks to production,
  and a token is refused by the wrong one. The app reports which it is, and the
  token is sent to that host.

  The JWT is a provider token, and Apple rejects one older than an hour
  (ExpiredProviderToken) and also one minted more often than every twenty
  minutes (TooManyProviderTokenUpdates). So it is cached and reused for 45.

Nothing here raises. A reminder is already delivered to the room by the time a
push is attempted, so a push that cannot go is a log line and not a failed
reminder. That is also why the sender runs on its own thread: the poller ticks
every five seconds and a slow APNs must not make a timer late.
"""

import json
import os
import threading
import time

import httpx
from jose import jwt

import config
from database import active_push_devices, deactivate_push_device

_HOSTS = {
    "production": "https://api.push.apple.com",
    "sandbox": "https://api.sandbox.push.apple.com",
}

# A token this old or younger is reused. Apple's window is 20 to 60 minutes.
_TOKEN_MAX_AGE_S = 45 * 60

# Reasons that mean this token will never work again, so asking again is noise.
# BadDeviceToken is also what a token gets from the wrong host; the phone
# registers again on its next launch and that reactivates the row.
_DEAD_TOKEN_REASONS = {"BadDeviceToken", "Unregistered"}

# The provider token itself is at fault. Mint a new one on the next send.
_BAD_PROVIDER_REASONS = {"ExpiredProviderToken", "InvalidProviderToken"}

_token_lock = threading.Lock()
_provider_token_cache = None        # (signed jwt, minted_at)


def missing():
    """Names of the settings still needed before a push can leave, in words."""
    gaps = []
    if not config.APNS_TEAM_ID:
        gaps.append("APNS_TEAM_ID")
    if not config.APNS_KEY_ID:
        gaps.append("APNS_KEY_ID")
    if not config.APNS_KEY_PATH:
        gaps.append("APNS_KEY_PATH")
    elif not os.path.isfile(config.APNS_KEY_PATH):
        gaps.append(f"the key file at APNS_KEY_PATH ({config.APNS_KEY_PATH})")
    return gaps


def configured():
    return bool(config.PUSH_NOTIFICATIONS) and not missing()


def _provider_token(now=None):
    """The signed JWT APNs wants in the authorization header, reused while fresh."""
    global _provider_token_cache
    now = time.time() if now is None else now
    with _token_lock:
        if _provider_token_cache and now - _provider_token_cache[1] < _TOKEN_MAX_AGE_S:
            return _provider_token_cache[0]
        with open(config.APNS_KEY_PATH) as key_file:
            key = key_file.read()
        # ES256 is what Apple's .p8 keys are. The key id travels in the header so
        # Apple knows which of the team's keys to check the signature against.
        signed = jwt.encode({"iss": config.APNS_TEAM_ID, "iat": int(now)}, key,
                            algorithm="ES256", headers={"kid": config.APNS_KEY_ID})
        _provider_token_cache = (signed, now)
        return signed


def _forget_provider_token():
    global _provider_token_cache
    with _token_lock:
        _provider_token_cache = None


def build_payload(title, body, kind, data=None):
    """The JSON APNs delivers. Anything in `data` rides beside "aps", where the
    app reads it when the notification is tapped."""
    payload = {"aps": {"alert": {"title": title, "body": body},
                       "sound": "default",
                       # Groups a kind's notifications together on the lock screen.
                       "thread-id": kind}}
    for key, value in (data or {}).items():
        if key != "aps":
            payload[key] = value
    return payload


def _reason(response):
    try:
        return response.json().get("reason", "")
    except ValueError:
        return ""


def _make_client():
    # http2=True needs the h2 package; without it httpx raises here, and notify
    # logs that rather than sending nothing in silence.
    return httpx.Client(http2=True, timeout=config.APNS_TIMEOUT_S)


def _send_one(client, token, environment, payload, ttl_s, collapse_id):
    """(accepted, why not) for one phone."""
    host = _HOSTS.get(environment)
    if host is None:
        return False, f"unknown environment {environment!r}"

    headers = {
        "authorization": f"bearer {_provider_token()}",
        "apns-topic": config.APNS_BUNDLE_ID,
        "apns-push-type": "alert",
        "apns-priority": "10",
        # Absolute time APNs stops holding it for an unreachable phone.
        "apns-expiration": str(int(time.time()) + ttl_s),
    }
    if collapse_id:
        # A second push with the same id replaces the first instead of stacking.
        headers["apns-collapse-id"] = collapse_id[:64]

    try:
        response = client.post(f"{host}/3/device/{token}", headers=headers,
                               content=json.dumps(payload).encode())
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if response.status_code == 200:
        return True, None

    reason = _reason(response)
    if response.status_code == 410 or reason in _DEAD_TOKEN_REASONS:
        deactivate_push_device(token, reason or f"HTTP {response.status_code}")
    elif reason in _BAD_PROVIDER_REASONS:
        _forget_provider_token()
    return False, f"HTTP {response.status_code} {reason}".strip()


def notify(title, body, kind="info", data=None, ttl_s=None, collapse_id=None):
    """Send to every active phone. Returns how many APNs accepted it.

    Accepted means APNs took it, not that the phone showed it. APNs does not
    say whether a notification was seen."""
    if not configured():
        return 0
    devices = active_push_devices()
    if not devices:
        return 0

    payload = build_payload(title, body, kind, data)
    if ttl_s is None:
        ttl_s = config.PUSH_TTL_S.get(kind, config.PUSH_DEFAULT_TTL_S)

    accepted = 0
    try:
        with _make_client() as client:
            for token, environment in devices:
                ok, why = _send_one(client, token, environment, payload, ttl_s, collapse_id)
                if ok:
                    accepted += 1
                else:
                    # The last six characters tell phones apart without logging a token.
                    print(f"Push to device ...{token[-6:]} ({environment}) failed: {why}",
                          flush=True)
    except Exception as exc:
        print(f"Push failed ({type(exc).__name__}): {exc}", flush=True)
    return accepted


def notify_async(title, body, kind="info", data=None, ttl_s=None, collapse_id=None):
    """notify() on its own thread, for callers that must not wait on Apple."""
    threading.Thread(
        target=notify, args=(title, body, kind, data, ttl_s, collapse_id),
        daemon=True).start()
