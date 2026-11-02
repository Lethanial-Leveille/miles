"""The phone's device token, and the geofence crossings it reports.

These use real signed tokens, not the dependency override the other app tests
use, because what is being tested is the check itself: that a device token
opens only the background writes, and that nothing else accepts it.
"""

import os
import tempfile
from datetime import datetime, timezone

import database

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402
from jose import jwt                             # noqa: E402

import auth                                      # noqa: E402
import server                                    # noqa: E402


@pytest.fixture
def client(db, monkeypatch):
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    server.app.dependency_overrides.clear()
    return TestClient(server.app)


def _bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def login(client):
    return auth.create_token("Lethanial")


@pytest.fixture
def device(client, login):
    return client.post("/auth/device", headers=_bearer(login)).json()["access_token"]


# ── the token ──

def test_a_device_token_is_scoped_and_lasts_months(device):
    claims = jwt.decode(device, "test secret", algorithms=["HS256"])
    assert claims["scope"] == "device"
    days = (datetime.fromtimestamp(claims["exp"], timezone.utc) - datetime.now(timezone.utc)).days
    assert 178 <= days <= 180


def test_a_device_token_cannot_read_anything(client, device):
    for path in ("/reminders", "/memories", "/history", "/presence/phone"):
        assert client.get(path, headers=_bearer(device)).status_code == 401, path


def test_a_device_token_cannot_mint_another(client, device):
    assert client.post("/auth/device", headers=_bearer(device)).status_code == 401


def test_a_device_token_writes_presence_and_health(client, device):
    assert client.post("/presence/phone", headers=_bearer(device),
                       json={"state": "away", "at": "2026-11-02T08:00:00"}).status_code == 200
    assert client.post("/health/sync", headers=_bearer(device),
                       json={"days": [{"day": "2026-11-02", "steps": 100}]}).status_code == 200


def test_a_login_still_works_on_the_device_routes(client, login):
    assert client.post("/presence/phone", headers=_bearer(login),
                       json={"state": "home", "at": "2026-11-02T08:00:00"}).status_code == 200


def test_a_token_with_any_other_scope_opens_nothing(client):
    other = auth.create_token("Lethanial", scope="something else")
    assert client.post("/presence/phone", headers=_bearer(other),
                       json={"state": "home", "at": "2026-11-02T08:00:00"}).status_code == 401
    assert client.get("/reminders", headers=_bearer(other)).status_code == 401


def test_an_expired_device_token_is_refused(client):
    stale = auth.create_token("Lethanial", expires_minutes=-1, scope="device")
    assert client.post("/presence/phone", headers=_bearer(stale),
                       json={"state": "home", "at": "2026-11-02T08:00:00"}).status_code == 401


# ── the crossings ──

def test_the_latest_crossing_is_by_the_phones_clock_not_arrival(client, login, device):
    # The newer crossing arrives first; iOS held the older one back
    client.post("/presence/phone", headers=_bearer(device),
                json={"state": "home", "at": "2026-11-02T18:30:00"})
    client.post("/presence/phone", headers=_bearer(device),
                json={"state": "away", "at": "2026-11-02T08:00:00"})
    assert client.get("/presence/phone", headers=_bearer(login)).json() == {
        "state": "home", "at": "2026-11-02T18:30:00"}


def test_nothing_reported_yet_says_so(client, login):
    assert client.get("/presence/phone", headers=_bearer(login)).json() == {"state": None, "at": None}


def test_an_unknown_state_is_refused(client, device):
    assert client.post("/presence/phone", headers=_bearer(device),
                       json={"state": "nearby", "at": "2026-11-02T08:00:00"}).status_code == 422
    assert database.latest_phone_presence() is None
