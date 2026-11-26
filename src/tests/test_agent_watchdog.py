"""The Mac watchdog and the once-a-day login-expired push."""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import agent_watchdog as W
import auth
import evallo_session
import push
import server

ET = W.TZ
DAY = "2026-11-24"


def t(h, m=0, day=24):
    return datetime(2026, 11, day, h, m, tzinfo=ET)


@pytest.fixture
def pushes(monkeypatch):
    sent = []
    monkeypatch.setattr(push, "notify", lambda title, body, **k: sent.append((title, body, k)) or 1)
    return sent


# ── when an alert goes out ──

def test_nothing_before_the_deadlines(db, pushes):
    assert W.check(now=t(19, 10)) == [] and pushes == []


def test_no_heartbeat_since_7pm_alerts_at_7_15_once(db, pushes):
    W.heartbeat("poll", now=t(18, 50))
    (body,) = W.check(now=t(19, 15))
    assert "has not checked in since 7 PM" in body
    title, _, k = pushes[0]
    assert k["category"] == W.CATEGORY and "watchdog_id" in k["data"]
    assert W.check(now=t(19, 20)) == [] and W.check(now=t(19, 45)) == []
    assert len(pushes) == 1


def test_heartbeat_but_no_draft_report_alerts_at_7_30(db, pushes):
    W.heartbeat("poll", now=t(19, 5))
    assert W.check(now=t(19, 15)) == []
    (body,) = W.check(now=t(19, 30))
    assert "did not report in within 30 minutes" in body


def test_a_successful_report_means_no_alert(db, pushes):
    W.heartbeat("draft", now=t(19, 0))
    W.job_report("draft", DAY, True, "2 proposed", now=t(19, 1))
    assert W.check(now=t(20, 0)) == []


def test_a_failed_report_alerts_with_its_reason(db, pushes):
    W.heartbeat("draft", now=t(19, 0))
    W.job_report("draft", DAY, False, "Google login expired", now=t(19, 1))
    (body,) = W.check(now=t(19, 30))
    assert "Google login expired" in body


def test_summaries_alert_at_9_30(db, pushes):
    W.heartbeat("poll", now=t(21, 20))
    W.job_report("draft", DAY, True, now=t(19, 1))
    assert W.check(now=t(21, 29)) == []
    (body,) = W.check(now=t(21, 30))
    assert "9 PM Zoom summary" in body


def test_a_pi_down_overnight_still_alerts_for_yesterday_before_7am(db, pushes):
    bodies = W.check(now=t(2, 0, day=25))
    assert any("Tue Nov 24" in b for b in bodies)


# ── what the buttons do ──

def _alert(db, pushes):
    # Only the drafts missed; the summary grab ran fine.
    W.job_report("summaries", DAY, True, now=t(21, 5))
    W.check(now=t(19, 15))
    (a,) = W.open_alerts()
    return a["id"]


def test_retry_is_handed_to_the_mac_once(db, pushes):
    aid = _alert(db, pushes)
    W.decide(aid, "retry", now=t(19, 20))
    assert W.heartbeat("poll", now=t(23, 0)) == [{"job": "draft", "day": DAY}]
    assert W.heartbeat("poll", now=t(23, 1)) == []


def test_success_after_retry_closes_the_alert(db, pushes):
    aid = _alert(db, pushes)
    W.decide(aid, "retry", now=t(19, 20))
    W.heartbeat("poll", now=t(23, 0))
    W.job_report("draft", DAY, True, now=t(23, 1))
    assert W.open_alerts() == []
    assert W.decide(aid, "retry")["status"] == "already_done"


def test_remind_pushes_again_at_7am_with_the_buttons(db, pushes):
    aid = _alert(db, pushes)
    r = W.decide(aid, "remind", now=t(19, 20))
    assert "7 AM" in r["message"]
    assert W.check(now=t(6, 59, day=25)) == []
    (body,) = W.check(now=t(7, 0, day=25))
    assert pushes[-1][2]["category"] == W.CATEGORY
    assert W.check(now=t(7, 5, day=25)) == []


def test_pushes_carry_what_nova_needs_to_open_them(db, pushes):
    aid = _alert(db, pushes)
    data = pushes[0][2]["data"]
    assert (data["kind"], data["level"], data["watchdog_id"]) == ("watchdog", "error", aid)
    assert data["detail"] == pushes[0][1]
    W.login_expired("zoom", "the 9 PM summary grab", now=t(21, 0))
    title, body, k = pushes[-1]
    assert k["kind"] == "login" and k["data"]["level"] == "error" and k["data"]["detail"] == body


def test_skip_ends_it(db, pushes):
    aid = _alert(db, pushes)
    W.decide(aid, "skip", now=t(19, 20))
    assert W.open_alerts() == [] and W.heartbeat("poll", now=t(23, 0)) == []
    assert W.check(now=t(7, 0, day=25)) == []


def test_unknown_decision_and_alert(db):
    with pytest.raises(W.AlertError):
        W.decide(1, "approve")
    with pytest.raises(W.AlertError) as e:
        W.decide(999, "skip")
    assert e.value.code == 404


# ── login expired ──

def test_login_expired_pushes_once_per_service_per_day(db, pushes):
    assert W.login_expired("zoom", "the 9 PM summary grab", now=t(21, 0))
    assert not W.login_expired("zoom", "the 8 AM login check", now=t(22, 0))
    assert W.login_expired("google", "the 7 PM drafts", now=t(22, 0))
    assert W.login_expired("zoom", "the 8 AM login check", now=t(8, 0, day=25))
    title, body, _ = pushes[0]
    assert title == "UF Zoom login expired"
    assert body.startswith("UF Zoom login expired (password change?). On the Mac run: ")
    assert "scripts/zoom_login.py" in body and "Failed job: the 9 PM summary grab." in body
    assert len(pushes) == 3


def test_evallo_login_expired_says_how_to_fix_it(db, pushes):
    assert evallo_session.login_expired("the nightly Evallo canary")
    assert not evallo_session.login_expired("marking Taylor's session Completed")
    (title, body, _), = pushes
    assert title == "Evallo login expired"
    assert "scripts/evallo_login.py" in body
    assert "scp" not in body and "evallo_state.json" not in body
    assert "Failed job: the nightly Evallo canary." in body


def test_evallo_runner_routes_login_expired_through_the_daily_alert(db, pushes, monkeypatch):
    import evallo_runner
    monkeypatch.setattr(evallo_runner, "_run",
                        lambda *a, **k: evallo_runner.Result("login_expired", "Evallo login expired, log in again"))
    monkeypatch.setattr(evallo_runner, "LOCK", evallo_runner.LOCK.parent / "test_evallo.lock")
    r = evallo_runner.set_session_status("Taylor", datetime(2026, 11, 24, 16, 15), "Completed")
    assert r.outcome == "login_expired"
    (title, body, _), = pushes
    assert title == "Evallo login expired" and "Taylor's session Completed" in body


def test_evallo_8am_check_is_silent_when_the_login_works(db, pushes, monkeypatch):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "evallo_canary", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "evallo_canary.py")
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)
    monkeypatch.setattr(canary.evallo_session, "load_state", lambda: {})
    monkeypatch.setattr(canary.evallo_runner, "next_session", lambda: None)
    assert canary.login_only() == 0 and pushes == []
    monkeypatch.setattr(canary.evallo_runner, "next_session",
                        lambda: canary.evallo_runner.Result("login_expired", "x"))
    assert canary.login_only() == 1
    assert pushes[0][0] == "Evallo login expired" and "8 AM login check" in pushes[0][1]


def test_evallo_8am_check_with_no_saved_login(db, pushes, monkeypatch):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "evallo_canary2", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "evallo_canary.py")
    canary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canary)

    def gone():
        raise evallo_session.LoginExpired("none")
    monkeypatch.setattr(canary.evallo_session, "load_state", gone)
    assert canary.login_only() == 1 and pushes[0][0] == "Evallo login expired"


# ── HTTP ──

@pytest.fixture
def client(db, monkeypatch, pushes):
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    return TestClient(server.app)


def _h(token):
    return {"Authorization": f"Bearer {token}"}


def test_agent_routes_take_the_agent_token(client):
    agent = auth.create_token("lethanial", scope=auth.AGENT_SCOPE)
    r = client.post("/agent/heartbeat", json={"source": "poll"}, headers=_h(agent))
    assert r.status_code == 200 and r.json() == {"retry": []}
    r = client.post("/agent/jobs/report", json={"job": "draft", "day": DAY, "ok": True},
                    headers=_h(agent))
    assert r.status_code == 200
    r = client.post("/agent/login-expired", json={"service": "zoom", "job": "x"}, headers=_h(agent))
    assert r.json() == {"pushed": True}
    r = client.post("/agent/login-expired", json={"service": "gmail", "job": "x"}, headers=_h(agent))
    assert r.status_code == 422


def test_the_agent_token_cannot_decide_alerts(client):
    agent = auth.create_token("lethanial", scope=auth.AGENT_SCOPE)
    assert client.post("/watchdog/1/retry", headers=_h(agent)).status_code == 401
    assert client.get("/watchdog", headers=_h(agent)).status_code == 401


def test_he_can_decide_with_his_login(client):
    W.check(now=datetime.now(ET).replace(hour=23, minute=0))
    full = auth.create_token("lethanial")
    alerts = client.get("/watchdog", headers=_h(full)).json()["alerts"]
    if alerts:
        r = client.post(f"/watchdog/{alerts[0]['id']}/skip", headers=_h(full))
        assert r.status_code == 200 and r.json()["status"] == "skip"
    assert client.post("/watchdog/1/approve", headers=_h(full)).status_code == 422


# ── the hourly Evallo sessions push ──

def test_a_failed_evallo_sessions_report_alerts_like_a_failed_draft(db, pushes):
    W.job_report("evallo_sessions", DAY, False, "Evallo answered with an unexpected shape",
                 now=t(10, 5))
    (title, body, k), = pushes
    assert title == "Mac missed a job" and k["category"] == W.CATEGORY
    assert "The hourly Evallo sessions push failed on Tue Nov 24" in body
    assert "unexpected shape" in body
    (alert,) = W.open_alerts()
    assert alert["job"] == "evallo_sessions" and k["data"]["watchdog_id"] == alert["id"]


def test_a_second_failure_the_same_day_does_not_push_twice(db, pushes):
    W.job_report("evallo_sessions", DAY, False, "one", now=t(10, 5))
    W.job_report("evallo_sessions", DAY, False, "two", now=t(11, 5))
    assert len(pushes) == 1 and len(W.open_alerts()) == 1
    W.job_report("evallo_sessions", "2026-11-25", False, "next day", now=t(10, 5, day=25))
    assert len(pushes) == 2


def test_an_ok_report_closes_the_alert(db, pushes):
    W.job_report("evallo_sessions", DAY, False, "down", now=t(10, 5))
    W.job_report("evallo_sessions", DAY, True, "stored 3", now=t(12, 5))
    assert W.open_alerts() == []


def test_no_missed_job_alert_is_ever_raised_for_it(db, pushes):
    assert "evallo_sessions" not in W.JOBS
    W.heartbeat("poll", now=t(21, 20))
    W.job_report("draft", DAY, True, now=t(19, 1))
    W.job_report("summaries", DAY, True, now=t(21, 1))
    assert W.check(now=t(23, 0)) == [] and W.check(now=t(2, 0, day=25)) == []
    assert pushes == [] and W.open_alerts() == []


def test_other_job_names_are_still_rejected(db):
    with pytest.raises(ValueError):
        W.job_report("lunch", DAY, True)


def test_the_route_accepts_evallo_sessions_and_rejects_other_names(db, pushes, monkeypatch):
    monkeypatch.setenv("MILES_JWT_SECRET", "test secret")
    server.app.dependency_overrides.clear()
    client = TestClient(server.app)
    login = auth.create_token("Lethanial")
    agent = client.post("/auth/agent", headers=_h(login)).json()["access_token"]
    ok = client.post("/agent/jobs/report", headers=_h(agent),
                     json={"job": "evallo_sessions", "day": DAY, "ok": False, "detail": "down"})
    assert ok.status_code == 200 and len(pushes) == 1
    bad = client.post("/agent/jobs/report", headers=_h(agent),
                      json={"job": "lunch", "day": DAY, "ok": True})
    assert bad.status_code == 422


def test_no_recovery_command_tells_him_to_copy_a_login_anywhere():
    for service, (_, command) in W.SERVICES.items():
        for word in ("scp", "rsync", "sftp", "evallo_state.json", "token", "cookie"):
            assert word not in command.lower(), (service, word)
