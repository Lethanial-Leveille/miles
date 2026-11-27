"""Signed tokens for Apple's web services, all from the one key "MILES Pi".

Nov 21 2026: one key (APNS_KEY_ID, the AuthKey .p8 at APNS_KEY_PATH) was made
with APNs, WeatherKit and Maps enabled, so weather no longer needs an
OpenWeatherMap account. push.py signs its own APNs token; this covers the two
REST services, which want different claims from the same key:

  WeatherKit  a JWT carrying the App ID ("sub") whose WeatherKit service is on,
              and "id" in the header as TEAM.APP_ID. Sent as is.
  Maps        a JWT with no subject, swapped at /v1/token for an access token
              that lasts thirty minutes. The access token is what geocode takes.

Both are cached until a few minutes before they expire, so a weather turn costs
no signing and, most of the time, no token round trip.
"""
import threading
import time

import requests
from jose import jwt

import config
from runtime_profile import refuse_live

# Not the bundle that receives pushes by accident: WeatherKit was enabled on
# this App ID under Identifiers, and a token naming any other is refused.
WEATHERKIT_APP_ID = "com.Lethanial.Nova"
_LIFETIME_S = 3600
_REFRESH_EARLY_S = 300
_MAPS_TOKEN_URL = "https://maps-api.apple.com/v1/token"

_lock = threading.Lock()
_cache = {}


def configured():
    return bool(config.APNS_TEAM_ID and config.APNS_KEY_ID and config.APNS_KEY_PATH)


def _key():
    refuse_live("Apple services")
    with open(config.APNS_KEY_PATH) as key_file:
        return key_file.read()


def _cached(name, make, now=None):
    now = time.time() if now is None else now
    with _lock:
        hit = _cache.get(name)
        if hit and hit[1] - _REFRESH_EARLY_S > now:
            return hit[0]
    token, expires_at = make(now)
    with _lock:
        _cache[name] = (token, expires_at)
    return token


def forget():
    """Drop every cached token, so the next call signs or fetches afresh."""
    with _lock:
        _cache.clear()


def weatherkit_token(now=None):
    def make(now):
        team = config.APNS_TEAM_ID
        signed = jwt.encode(
            {"iss": team, "iat": int(now), "exp": int(now) + _LIFETIME_S,
             "sub": WEATHERKIT_APP_ID},
            _key(), algorithm="ES256",
            headers={"kid": config.APNS_KEY_ID, "id": f"{team}.{WEATHERKIT_APP_ID}"})
        return signed, now + _LIFETIME_S
    return _cached("weatherkit", make, now)


def maps_access_token(now=None):
    def make(now):
        auth = jwt.encode(
            {"iss": config.APNS_TEAM_ID, "iat": int(now), "exp": int(now) + _LIFETIME_S},
            _key(), algorithm="ES256", headers={"kid": config.APNS_KEY_ID, "typ": "JWT"})
        r = requests.get(_MAPS_TOKEN_URL, headers={"Authorization": f"Bearer {auth}"},
                         timeout=10)
        r.raise_for_status()
        body = r.json()
        return body["accessToken"], now + int(body.get("expiresInSeconds", 1800))
    return _cached("maps", make, now)
