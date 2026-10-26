"""Prowl through what Nova would hear. No network: _request is replaced."""

import pytest

import pending_action as pa
import prowl_tools as prowl


@pytest.fixture(autouse=True)
def fresh():
    pa._pending, pa._turn = None, 0
    yield
    pa._pending, pa._turn = None, 0


class FakeProwl:
    """Answers GETs from a table and records every call."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, path, **params):
        self.calls.append((method, path, params))
        if method == "POST":
            return {}
        return self.routes[path]


@pytest.fixture
def fake(monkeypatch):
    def install(routes):
        server = FakeProwl(routes)
        monkeypatch.setattr(prowl, "_request", server)
        return server
    return install


def _row(**overrides):
    row = {
        "id": "a1", "organization": "Anduril", "role": "Embedded Intern",
        "status": "applied", "priority": "high",
        "posting_url": "https://example.com/job", "role_family": "embedded",
        "deadline": "2026-10-01", "days_until_deadline": 6,
        "deadline_source": "posting",
        "applied_at": "2026-09-10T12:00:00", "days_since_applied": 15,
        "days_since_last_change": 15,
    }
    row.update(overrides)
    return row


def _keys(value):
    """Every key anywhere in a nested result."""
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(v) for v in value)) if value else set()
    return set()


# ── deadlines are named by whose they are ──

@pytest.mark.parametrize("source, date_key, days_key", [
    ("posting", "applications_close", "days_until_applications_close"),
    ("self", "your_own_target_date", "days_until_your_own_target"),
    (None, "date_of_unknown_origin", "days_until_that_date"),
    ("recruiter", "date_of_unknown_origin", "days_until_that_date"),
])
def test_a_deadline_is_renamed_by_its_source(source, date_key, days_key):
    out = prowl.speakable_deadline(_row(deadline_source=source, days_until_deadline=-12))
    assert out[date_key] == "2026-10-01"
    assert out[days_key] == -12
    assert not {"deadline", "days_until_deadline", "deadline_source"} & set(out)


def test_no_deadline_adds_nothing():
    out = prowl.speakable_deadline(_row(deadline=None, days_until_deadline=None,
                                        deadline_source=None))
    assert not any("deadline" in k or "target" in k or "origin" in k for k in out)


def test_never_applied_is_words_not_a_null():
    out = prowl._application(_row(applied_at=None, days_since_applied=None))
    assert out["days_since_applied"] == "never applied"
    assert "applied_at" not in out
    assert "posting_url" not in out


def test_applied_today_stays_zero():
    assert prowl._application(_row(days_since_applied=0))["days_since_applied"] == 0


# ── the summary ──

def _summary():
    return {
        "active_count": 109,
        "counts_by_status": {"applied": 62, "discovered": 46, "rejected": 5},
        "upcoming_deadlines": [_row(deadline_source="self")],
        "overdue": [_row(id="a2", days_until_deadline=-3)],
        "stale": [_row(id="a3", deadline=None, days_until_deadline=None,
                       deadline_source=None)],
        "pending_suggestions": [],
    }


def test_summary_counts_pass_through_untouched(fake):
    fake({"summary": _summary()})
    out = prowl.get_job_search_summary()
    assert out["active_count"] == 109
    assert out["counts_by_status_including_closed"] == {"applied": 62, "discovered": 46,
                                                        "rejected": 5}
    assert "counts_by_status" not in out


def test_no_field_called_deadline_reaches_the_model(fake):
    fake({"summary": _summary()})
    out = prowl.get_job_search_summary()
    assert not {"deadline", "days_until_deadline", "deadline_source"} & _keys(out)
    assert out["upcoming_deadlines"][0]["your_own_target_date"] == "2026-10-01"
    assert out["overdue"][0]["days_until_applications_close"] == -3


def test_deadline_days_is_sent_only_when_asked(fake):
    server = fake({"summary": _summary()})
    prowl.get_job_search_summary()
    prowl.get_job_search_summary(deadline_days=14)
    assert server.calls == [("GET", "summary", {}), ("GET", "summary", {"deadline_days": 14})]


# ── lists ──

def test_the_status_filter_is_prowls_not_ours(fake):
    server = fake({"applications": [_row()]})
    prowl.list_job_applications("applied")
    assert server.calls == [("GET", "applications", {"status": "applied"})]


def test_a_long_list_is_cut_and_flagged_never_counted(fake):
    fake({"applications": [_row(id=f"a{i}") for i in range(20)]})
    out = prowl.list_job_applications("applied")
    assert [r["id"] for r in out["applications"]] == [f"a{i}" for i in range(15)]
    assert out["more_not_shown"] is True
    assert set(out) == {"applications", "more_not_shown"}


def test_a_short_list_says_nothing_is_hidden(fake):
    fake({"applications": [_row()]})
    assert prowl.list_job_applications("applied")["more_not_shown"] is False


def test_an_empty_status_says_so(fake):
    fake({"applications": []})
    assert "No applications" in prowl.list_job_applications("offer")


def test_a_detail_with_no_fit_check_says_never_run(fake):
    detail = _row(requirements_met=None, requirements_partial=None,
                  requirements_total=None, key_requirements=["C"])
    fake({"applications/a1": detail})
    out = prowl.get_job_application("a1")
    assert out["fit_check"] == "never run"
    assert "requirements_total" not in out


def test_a_detail_with_a_fit_check_keeps_the_counts(fake):
    fake({"applications/a1": _row(requirements_met=3, requirements_partial=1,
                                  requirements_total=5)})
    out = prowl.get_job_application("a1")
    assert (out["requirements_met"], out["requirements_total"]) == (3, 5)
    assert "fit_check" not in out


# ── retired ──

def test_the_retired_discovery_inbox_is_gone():
    # Prowl retired its discovery inbox (Oct 26 2026) and its MILES endpoints
    # with it. Neither tool may come back pointing at a route that 404s.
    assert not hasattr(prowl, "list_discovered_jobs")
    assert not hasattr(prowl, "accept_discovered_job")


# ── failures ──

class _Response:
    def __init__(self, status):
        self.status_code = status

    def raise_for_status(self):
        pass

    def json(self):
        return {}


@pytest.mark.parametrize("status, words", [
    (401, "rejected the MILES token"),
    (404, "stale"),
    (500, "OWNER_EMAIL"),
])
def test_each_failure_says_where_to_look(monkeypatch, status, words):
    monkeypatch.setenv("PROWL_URL", "https://prowl.test/api")
    monkeypatch.setenv("PROWL_TOKEN", "t")
    monkeypatch.setattr(prowl.requests, "request", lambda *a, **k: _Response(status))
    with pytest.raises(prowl.ProwlError, match=words):
        prowl._request("GET", "summary")


def test_the_token_goes_in_its_own_header(monkeypatch):
    seen = {}
    monkeypatch.setenv("PROWL_URL", "https://prowl.test/api/")
    monkeypatch.setenv("PROWL_TOKEN", "t")

    def request(method, url, **kwargs):
        seen.update(method=method, url=url, headers=kwargs["headers"])
        return _Response(200)

    monkeypatch.setattr(prowl.requests, "request", request)
    prowl._request("GET", "summary")
    assert seen == {"method": "GET", "url": "https://prowl.test/api/integrations/miles/summary",
                    "headers": {"X-Miles-Token": "t"}}


def test_a_missing_token_fails_without_a_call(monkeypatch):
    monkeypatch.delenv("PROWL_TOKEN", raising=False)
    monkeypatch.setattr(prowl.requests, "request", lambda *a, **k: pytest.fail("called"))
    with pytest.raises(prowl.ProwlError, match="missing"):
        prowl._request("GET", "summary")
