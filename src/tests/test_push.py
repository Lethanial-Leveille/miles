"""Pushes to his phone: what is signed, where it goes, and what a refusal does.

APNs cannot be called from a test, so httpx's MockTransport stands in for Apple
and these pin the parts that are ours: the provider token's claims and reuse,
the host chosen per environment, the headers APNs requires, and that a token
Apple calls dead is stopped while a failure on our side is not.
"""

import json
import os
import sqlite3
import tempfile

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from jose import jwt

import actions
import config
import database
import push

# server.py calls init_db() at import. Point it somewhere throwaway first.
database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

from fastapi.testclient import TestClient        # noqa: E402

import server                                    # noqa: E402

TOKEN_A = "a" * 64
TOKEN_B = "b" * 64


@pytest.fixture
def apns_key(tmp_path, monkeypatch):
    """A real P-256 key in the PKCS8 PEM form Apple's .p8 files use, with the
    settings pointing at it. Returns the public half, for checking signatures."""
    private = ec.generate_private_key(ec.SECP256R1())
    path = tmp_path / "AuthKey_TEST.p8"
    path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    monkeypatch.setattr(config, "APNS_TEAM_ID", "TEAM123456")
    monkeypatch.setattr(config, "APNS_KEY_ID", "KEY1234567")
    monkeypatch.setattr(config, "APNS_KEY_PATH", str(path))
    monkeypatch.setattr(config, "APNS_BUNDLE_ID", "com.example.Nova")
    monkeypatch.setattr(push, "_provider_token_cache", None)
    return private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


@pytest.fixture
def apns(db, apns_key, monkeypatch):
    """Replaces Apple. `answer` is what it replies, `seen` every request made."""
    class Apple:
        status = 200
        body = {}
        seen = []

    def handler(request):
        Apple.seen.append(request)
        return httpx.Response(Apple.status, json=Apple.body)

    Apple.seen = []
    monkeypatch.setattr(push, "_make_client",
                        lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    return Apple


def _active():
    conn = sqlite3.connect(database.DB_PATH)
    rows = conn.execute("SELECT token FROM push_devices WHERE active = 1").fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── the provider token ──

def test_the_provider_token_names_the_team_and_the_key(apns_key):
    signed = push._provider_token(now=1_000_000)
    assert jwt.get_unverified_header(signed) == {"alg": "ES256", "kid": "KEY1234567", "typ": "JWT"}
    claims = jwt.decode(signed, apns_key, algorithms=["ES256"], options={"verify_exp": False})
    assert claims == {"iss": "TEAM123456", "iat": 1_000_000}


def test_the_provider_token_is_reused_inside_apples_window_and_replaced_after(apns_key):
    first = push._provider_token(now=1_000_000)
    assert push._provider_token(now=1_000_000 + 44 * 60) == first
    assert push._provider_token(now=1_000_000 + 46 * 60) != first


# ── what is configured ──

def test_missing_names_each_gap(monkeypatch):
    monkeypatch.setattr(config, "APNS_TEAM_ID", None)
    monkeypatch.setattr(config, "APNS_KEY_ID", "KEY1234567")
    monkeypatch.setattr(config, "APNS_KEY_PATH", "/nowhere/key.p8")
    gaps = push.missing()
    assert gaps[0] == "APNS_TEAM_ID"
    assert "the key file" in gaps[1]
    assert not push.configured()


def test_nothing_is_sent_when_push_is_not_set_up(db, monkeypatch):
    monkeypatch.setattr(config, "APNS_KEY_PATH", None)
    database.save_push_device(TOKEN_A, "sandbox")
    monkeypatch.setattr(push, "_make_client", lambda: pytest.fail("no client should open"))
    assert push.notify("Reminder", "call mom") == 0


def test_nothing_is_sent_with_no_phone_registered(apns):
    assert push.notify("Reminder", "call mom") == 0
    assert apns.seen == []


# ── a send ──

def test_a_push_goes_to_the_host_its_phone_registered_against(apns):
    database.save_push_device(TOKEN_A, "sandbox")
    database.save_push_device(TOKEN_B, "production")
    assert push.notify("Reminder", "call mom", kind="reminder") == 2

    urls = sorted(str(r.url) for r in apns.seen)
    assert urls == [f"https://api.push.apple.com/3/device/{TOKEN_B}",
                    f"https://api.sandbox.push.apple.com/3/device/{TOKEN_A}"]


def test_the_request_carries_what_apns_requires(apns, apns_key):
    database.save_push_device(TOKEN_A, "sandbox")
    push.notify("Reminder", "call mom", kind="reminder",
                data={"reminder_id": 7}, collapse_id="reminder-7")

    request = apns.seen[0]
    assert request.headers["apns-topic"] == "com.example.Nova"
    assert request.headers["apns-push-type"] == "alert"
    assert request.headers["authorization"].startswith("bearer ")
    assert request.headers["apns-collapse-id"] == "reminder-7"
    # A reminder is held six hours for an unreachable phone.
    held = int(request.headers["apns-expiration"]) - int(__import__("time").time())
    assert 6 * 3600 - 5 <= held <= 6 * 3600

    body = json.loads(request.content)
    assert body["aps"]["alert"] == {"title": "Reminder", "body": "call mom"}
    assert body["aps"]["thread-id"] == "reminder"
    assert body["reminder_id"] == 7


def test_custom_data_cannot_overwrite_the_aps_dictionary():
    payload = push.build_payload("t", "b", "reminder", {"aps": "x", "reminder_id": 1})
    assert payload["aps"]["alert"]["title"] == "t"
    assert payload["reminder_id"] == 1


# ── a refusal ──

@pytest.mark.parametrize("status,reason", [(410, "Unregistered"), (400, "BadDeviceToken")])
def test_a_token_apple_calls_dead_is_not_asked_again(apns, status, reason):
    database.save_push_device(TOKEN_A, "sandbox")
    apns.status, apns.body = status, {"reason": reason}
    assert push.notify("Reminder", "call mom") == 0
    assert _active() == []

    # The app registers on every launch, which brings the same row back.
    database.save_push_device(TOKEN_A, "production")
    assert _active() == [TOKEN_A]


def test_a_bad_provider_token_is_reminted_and_the_phone_is_kept(apns):
    database.save_push_device(TOKEN_A, "sandbox")
    push._provider_token()
    apns.status, apns.body = 403, {"reason": "ExpiredProviderToken"}
    assert push.notify("Reminder", "call mom") == 0
    assert _active() == [TOKEN_A]
    assert push._provider_token_cache is None


def test_a_server_error_keeps_the_phone(apns):
    database.save_push_device(TOKEN_A, "sandbox")
    apns.status, apns.body = 503, {"reason": "ServiceUnavailable"}
    assert push.notify("Reminder", "call mom") == 0
    assert _active() == [TOKEN_A]


def test_a_network_failure_is_logged_not_raised(db, apns_key, monkeypatch, capsys):
    database.save_push_device(TOKEN_A, "sandbox")

    def refuse(request):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(push, "_make_client",
                        lambda: httpx.Client(transport=httpx.MockTransport(refuse)))
    assert push.notify("Reminder", "call mom") == 0
    assert "ConnectError" in capsys.readouterr().out
    assert _active() == [TOKEN_A]


# ── the poller ──

@pytest.fixture
def pushed(db, monkeypatch):
    sent = []
    monkeypatch.setattr(actions, "DB_PATH", database.DB_PATH)
    monkeypatch.setattr(push, "notify_async", lambda *a, **k: sent.append((a, k)))
    return sent


def test_a_due_reminder_is_also_pushed(pushed):
    actions.set_reminder("take the trash out", "2000-01-01T09:00:00")
    assert actions.poll_reminders() == 1

    (args, kwargs), = pushed
    assert args == ("Reminder", "take the trash out")
    assert kwargs["kind"] == "reminder"
    assert kwargs["collapse_id"] == f"reminder-{kwargs['data']['reminder_id']}"


def test_a_finished_timer_is_pushed_in_plain_words(pushed):
    conn = sqlite3.connect(database.DB_PATH)
    conn.execute("INSERT INTO reminders (content, due_at, created_at, kind) "
                 "VALUES ('10 minute timer', '2000-01-01T09:00:00', '2000-01-01T08:50:00', 'timer')")
    conn.commit()
    conn.close()

    assert actions.poll_reminders() == 1
    (args, kwargs), = pushed
    assert args == ("Timer", "Your 10 minute timer is up")
    assert kwargs["kind"] == "timer"


def test_a_kind_left_out_of_push_kinds_stays_in_the_room(pushed, monkeypatch):
    monkeypatch.setattr(actions, "PUSH_KINDS", ())
    actions.set_reminder("take the trash out", "2000-01-01T09:00:00")
    assert actions.poll_reminders() == 1
    assert pushed == []


# ── the endpoints ──

@pytest.fixture
def client(db):
    server.app.dependency_overrides[server.get_current_user] = lambda: "Lethanial"
    yield TestClient(server.app)
    server.app.dependency_overrides.clear()


def test_a_phone_registers_and_the_call_can_repeat(client):
    body = {"token": TOKEN_A.upper(), "environment": "sandbox", "device_name": "Lee's iPhone"}
    assert client.post("/push/register", json=body).json() == {"registered": True}
    assert client.post("/push/register", json=body).status_code == 200
    # Stored lowercase, so the same token in either case is one row.
    assert _active() == [TOKEN_A]


@pytest.mark.parametrize("body", [
    {"token": "../etc/passwd", "environment": "sandbox"},
    {"token": TOKEN_A, "environment": "staging"},
    {"token": "abc", "environment": "sandbox"},
])
def test_a_malformed_registration_is_refused(client, body):
    assert client.post("/push/register", json=body).status_code == 422
    assert _active() == []


def test_signing_out_unregisters_the_phone(client):
    database.save_push_device(TOKEN_A, "sandbox")
    assert client.post("/push/unregister", json={"token": TOKEN_A}).json() == {"unregistered": True}
    assert _active() == []
    # Safe to retry.
    assert client.post("/push/unregister", json={"token": TOKEN_A}).status_code == 200


def test_the_test_push_says_what_is_missing(client, monkeypatch):
    monkeypatch.setattr(config, "APNS_TEAM_ID", None)
    monkeypatch.setattr(config, "APNS_KEY_ID", None)
    monkeypatch.setattr(config, "APNS_KEY_PATH", None)
    response = client.post("/push/test")
    assert response.status_code == 503
    assert "APNS_TEAM_ID, APNS_KEY_ID, APNS_KEY_PATH" in response.json()["detail"]


def test_the_test_push_reports_how_many_phones_took_it(client, apns):
    database.save_push_device(TOKEN_A, "sandbox")
    assert client.post("/push/test").json() == {"sent": 1}

    apns.status, apns.body = 403, {"reason": "InvalidProviderToken"}
    assert client.post("/push/test").status_code == 502
