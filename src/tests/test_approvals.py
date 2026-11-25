"""Approvals: one tap per request, two hour expiry, the Mac can never approve."""

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import database

database.DB_PATH = os.path.join(tempfile.mkdtemp(), "test_miles.db")

import pytest                                    # noqa: E402
from fastapi.testclient import TestClient        # noqa: E402

import approvals                                 # noqa: E402
import auth                                      # noqa: E402
import push                                      # noqa: E402
import server                                    # noqa: E402
from tools import Permission, permits, registry  # noqa: E402

T0 = datetime(2026, 11, 23, 19, 0, tzinfo=timezone.utc)
ZOOM = "https://ufl.zoom.us/j/1234567890?pwd=abc123"
TEXT = ("Hey Taylor, just a reminder that your session is tomorrow at 4:15 PM. Here's the "
        f"Zoom link: {ZOOM}. Let me know if anything comes up!")


@pytest.fixture(autouse=True)
def zoom_file(tmp_path, monkeypatch):
    path = tmp_path / "zoom.json"
    path.write_text(json.dumps({"link": ZOOM}))
    monkeypatch.setattr(approvals, "ZOOM_PATH", str(path))
    return path


@pytest.fixture
def pushes(monkeypatch):
    sent = []
    monkeypatch.setattr(push, "notify", lambda *a, **k: sent.append((a, k)) or 1)
    return sent


def _propose(**kw):
    args = dict(kind="imessage", summary="Remind Taylor", text=TEXT, chat_name="Taylor & Mom",
                payload={"chat_id": "iMessage;+;chat1"}, dedupe_key="k1", now=T0)
    args.update(kw)
    return approvals.propose(**args)


def test_propose_then_approve_then_claim_once(db):
    row, created = _propose()
    assert created and row["status"] == "pending"
    approvals.decide(row["id"], True, now=T0 + timedelta(minutes=5))
    claimed = approvals.claim_approved(now=T0 + timedelta(minutes=6))
    assert [c["text"] for c in claimed] == [TEXT]
    assert approvals.claim_approved(now=T0 + timedelta(minutes=7)) == []


def test_a_repeat_dedupe_key_never_asks_again(db):
    row, _ = _propose()
    approvals.decide(row["id"], False, now=T0)
    again, created = _propose()
    assert not created and again["status"] == "skipped"


def test_expired_request_cannot_be_approved_and_does_nothing(db):
    row, _ = _propose()
    with pytest.raises(approvals.ApprovalError) as err:
        approvals.decide(row["id"], True, now=T0 + timedelta(hours=2, seconds=1))
    assert err.value.code == 410
    assert approvals.get(row["id"], now=T0 + timedelta(hours=3))["status"] == "expired"
    assert approvals.claim_approved(now=T0 + timedelta(hours=3)) == []


def test_approved_but_unclaimed_past_two_hours_expires(db):
    row, _ = _propose()
    approvals.decide(row["id"], True, now=T0 + timedelta(minutes=110))
    assert approvals.claim_approved(now=T0 + timedelta(minutes=121)) == []
    assert approvals.get(row["id"], now=T0 + timedelta(minutes=122))["status"] == "expired"


def test_cannot_decide_twice(db):
    row, _ = _propose()
    approvals.decide(row["id"], True, now=T0)
    with pytest.raises(approvals.ApprovalError) as err:
        approvals.decide(row["id"], False, now=T0)
    assert err.value.code == 409


def test_every_step_is_logged_with_text_and_chat(db):
    row, _ = _propose()
    approvals.decide(row["id"], True, now=T0)
    approvals.claim_approved(now=T0)
    approvals.report(row["id"], True, "sent, seen in chat.db", now=T0)
    log = list(reversed(approvals.events()))
    assert [e["event"] for e in log] == ["proposed", "approved", "claimed", "done"]
    assert all(e["text"] == TEXT and e["chat_name"] == "Taylor & Mom" and e["at"] for e in log)


@pytest.mark.parametrize("bad", [
    "", "x" * 401, "Visit https://evil.example.com/x now", "see bit.ly/abc",
    "Join https://meet.google.com/abc-defg-hij",
    "Join https://ufl.zoom.us/j/9999999999?pwd=other",          # a different Zoom meeting
    f"Join {ZOOM}&x=1",                                         # the link, altered
    "Join ufl.zoom.us/j/1234567890",                            # no scheme, still a link
    f"Join https://zoom.us.evil.com/j/1234567890 or {ZOOM}",
])
def test_texts_are_short_and_carry_only_the_zoom_link(db, bad):
    with pytest.raises(ValueError):
        _propose(text=bad, dedupe_key=None)


def test_the_zoom_link_is_allowed_with_trailing_punctuation(db):
    for text in (TEXT, f"Here: {ZOOM}!", "No link at all, see you tomorrow."):
        assert approvals.check_text(text) == text


@pytest.mark.parametrize("content", [None, "", "{bad", '{"link": ""}',
                                     '{"link": "https://meet.google.com/abc-defg-hij"}'])
def test_without_a_valid_zoom_file_no_link_is_allowed(db, zoom_file, content):
    if content is None:
        zoom_file.unlink()
    else:
        zoom_file.write_text(content)
    with pytest.raises(ValueError, match="No links are allowed"):
        approvals.check_text(TEXT)
    assert approvals.check_text("See you tomorrow!") == "See you tomorrow!"


@pytest.mark.parametrize("link,ok", [
    ("https://ufl.zoom.us/j/8607898540?omn=94543491016", True),
    ("https://ufl.zoom.us/j/8607898540?pwd=abc123&omn=94543491016", True),
    ("https://ufl.zoom.us/j/8607898540?omn=abc", False),
    ("https://ufl.zoom.us/j/8607898540?omn=1&extra=2", False),
])
def test_zoom_links_with_an_omn_number(db, zoom_file, link, ok):
    zoom_file.write_text(json.dumps({"link": link}))
    assert approvals.zoom_link() == (link if ok else None)
    if ok:
        text = f"Here's the Zoom link: {link}. Let me know if anything comes up!"
        assert approvals.check_text(text) == text
        with pytest.raises(ValueError):
            approvals.check_text(f"Here: {link}1.")


def test_meeting_link_edits_are_no_longer_a_kind(db):
    with pytest.raises(ValueError):
        _propose(kind="evallo_meet_link", text=None, dedupe_key=None)


# ── HTTP ──

@pytest.fixture
def client(db, monkeypatch, pushes):
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    server.app.dependency_overrides.clear()
    return TestClient(server.app)


def _h(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def login(client):
    return auth.create_token("Lethanial")


@pytest.fixture
def agent(client, login):
    return client.post("/auth/agent", headers=_h(login)).json()["access_token"]


def _agent_propose(client, agent, **kw):
    body = dict(kind="imessage", summary="Remind Taylor", text=TEXT, chat_name="Taylor & Mom",
                payload={"chat_id": "iMessage;+;chat1"}, dedupe_key="http1")
    body.update(kw)
    return client.post("/agent/approvals", json=body, headers=_h(agent))


def test_agent_proposes_and_phone_gets_a_push_with_buttons(client, agent, pushes):
    r = _agent_propose(client, agent)
    assert r.status_code == 201 and r.json()["created"]
    (args, kwargs), = pushes
    assert kwargs["category"] == approvals.APPROVAL_CATEGORY
    assert "Taylor & Mom" in args[1] and TEXT in args[1]


def test_agent_token_can_never_approve(client, agent):
    aid = _agent_propose(client, agent).json()["approval"]["id"]
    assert client.post(f"/approvals/{aid}/approve", headers=_h(agent)).status_code == 401
    assert client.post("/agent/approvals/claim", headers=_h(agent)).json()["approvals"] == []


def test_device_token_cannot_approve_or_propose(client, login, agent):
    device = client.post("/auth/device", headers=_h(login)).json()["access_token"]
    aid = _agent_propose(client, agent).json()["approval"]["id"]
    assert client.post(f"/approvals/{aid}/approve", headers=_h(device)).status_code == 401
    assert _agent_propose(client, device, dedupe_key="d").status_code == 401


def test_full_flow_over_http(client, login, agent):
    aid = _agent_propose(client, agent).json()["approval"]["id"]
    assert [a["id"] for a in client.get("/approvals", headers=_h(login)).json()["approvals"]] == [aid]
    assert client.post(f"/approvals/{aid}/approve", headers=_h(login)).json()["status"] == "approved"
    claimed = client.post("/agent/approvals/claim?kinds=imessage", headers=_h(agent)).json()["approvals"]
    assert [c["id"] for c in claimed] == [aid]
    r = client.post(f"/agent/approvals/{aid}/result", json={"ok": True, "result": "sent"},
                    headers=_h(agent))
    assert r.json() == {"status": "done"}


# ── the tool ──

def test_send_imessage_is_external_write_hokage_only():
    spec = registry.get("send_imessage")
    assert spec.permission is Permission.EXTERNAL_WRITE
    assert permits(spec, "hokage") and not permits(spec, "chunin")


def test_send_imessage_never_guesses_a_chat(db, pushes):
    import imessage_tools
    from tools import ToolNotice
    with pytest.raises(ToolNotice):
        imessage_tools.send_imessage("Zed", "Hi! Can we move to 5 PM?")
    assert pushes == [] and approvals.list_pending() == []


def test_send_imessage_only_queues(db, pushes):
    import imessage_tools
    approvals.set_contacts({"Taylor": "Taylor & Mom"})
    said = imessage_tools.send_imessage("taylor", "Hi! Can we move to 5 PM tomorrow?")
    assert "Nothing has been sent" in said
    (row,) = approvals.list_pending()
    assert row["chat_name"] == "Taylor & Mom" and row["status"] == "pending"
