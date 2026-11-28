"""The chat WebSocket applies the same token rule as every REST route, in
production as well as in the test runtime.

The test runtime is always headless, so a check that only the headless branch
made would pass every test and protect nothing in production. This forces the
non headless answer and requires the same refusal."""

from unittest.mock import Mock

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

import server
from auth import AGENT_SCOPE, DEVICE_SCOPE, create_token


@pytest.mark.parametrize("headless", [True, False])
@pytest.mark.parametrize("scope", [DEVICE_SCOPE, AGENT_SCOPE])
def test_scoped_token_cannot_open_the_chat_socket(db, monkeypatch, scope, headless):
    dispatch = Mock()
    monkeypatch.setattr(server, "ask_nova", dispatch)
    monkeypatch.setattr(server, "headless", lambda: headless)

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"token": create_token("Alex", scope=scope)})
        assert ws.receive_json() == {"error": "Unauthorized"}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008
    dispatch.assert_not_called()


@pytest.mark.parametrize("headless", [True, False])
def test_full_login_still_opens_the_chat_socket(db, monkeypatch, headless):
    monkeypatch.setattr(server, "headless", lambda: headless)
    monkeypatch.setattr(server, "ask_nova", Mock(return_value=Mock(text="Hello")))

    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"token": create_token("Alex")})
        assert ws.receive_json() == {"status": "authenticated"}
        ws.send_json({"message": "Synthetic request", "channel": "text"})
        assert ws.receive_json() == {"response": "Hello"}


def test_garbage_token_is_refused(db):
    with TestClient(server.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"token": "not a token"})
        assert ws.receive_json() == {"error": "Unauthorized"}


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"])
def test_api_docs_and_schema_are_not_served(db, path):
    with TestClient(server.app) as client:
        assert client.get(path).status_code == 404


def test_status_still_refuses_a_request_with_no_token(db):
    # scripts/healthcheck.py relies on exactly this answer to prove the path is up
    with TestClient(server.app) as client:
        reply = client.get("/status")
    assert reply.status_code in (401, 403)
    assert reply.json()["detail"] in {"Not authenticated", "Invalid or expired token"}
