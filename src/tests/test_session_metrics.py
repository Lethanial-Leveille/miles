"""The nightly review's numbers: outcomes, cost at list prices, and must dos
counted only once their day is over."""

from datetime import datetime

import pytest

import session_metrics as m
import sessions


def _s(status="closed", closed_by="user", day="2026-11-24", **extra):
    return {"status": status, "closed_by": closed_by, "for_day": day, **extra}


@pytest.mark.parametrize("session, wrote, expected", [
    (_s(), 3, m.COMPLETED),
    (_s(closed_by="timer"), 3, m.ABANDONED),
    (_s(status="missed", closed_by="timer"), 0, m.MISSED),
    (_s(status="open", closed_by=None), 1, m.OPEN),
])
def test_outcomes(session, wrote, expected):
    assert m.outcome(session, wrote) == expected


def test_cost_at_list_prices_includes_the_summary_call():
    usage = [("claude-haiku-4-5", 10_000, 2_000, 50_000, 4_000, 5)]
    # 10k in at $1, 2k out at $5, 50k cache read at $0.10, 4k cache write at
    # $1.25, plus a 3k in, 200 out summary.
    expected = (10_000 * 1 + 2_000 * 5 + 50_000 * 0.10 + 4_000 * 1.25
                + 3_000 * 1 + 200 * 5) / 1e6
    assert m.cost(usage, "claude-haiku-4-5", 3_000, 200) == pytest.approx(expected)


def test_an_unpriced_model_is_none_not_a_guess():
    assert m.cost([("some-new-model", 1, 1, 0, 0, 1)]) is None


def test_summarize_rates_and_must_dos_only_for_finished_days():
    rows = [
        {"session": _s(day="2026-11-22"), "user_messages": 4, "usage": [],
         "must_dos_day": "2026-11-23", "must_dos_planned": 3, "must_dos_done": 2},
        {"session": _s(status="missed", closed_by="timer", day="2026-11-23"),
         "user_messages": 0, "usage": [],
         "must_dos_day": "2026-11-24", "must_dos_planned": 0, "must_dos_done": 0},
        # Tomorrow's must dos are not over yet, so they do not count.
        {"session": _s(day="2026-11-24"), "user_messages": 2, "usage": [],
         "must_dos_day": "2026-11-25", "must_dos_planned": 2, "must_dos_done": 0},
    ]
    report = m.summarize(rows, today="2026-11-24")
    assert report["settled"] == 3
    assert report["completion_rate"] == pytest.approx(2 / 3)
    assert (report["must_dos_planned"], report["must_dos_done"]) == (3, 2)
    assert report["nights"][2]["must_dos_done"] is None


def test_rows_from_the_database(db):
    created, _ = sessions.create("nightly_review", now=datetime(2026, 11, 24, 21, 30))
    db.add_planned_tasks([{"text": "Lab", "scheduled_for": "2026-11-25", "must_do": True}],
                         datetime(2026, 11, 24, 22, 0))
    db.finish_session(created["id"], "closed", "ok", datetime(2026, 11, 24, 22, 10), "user")
    db.record_summary_usage(created["id"], "claude-haiku-4-5", 1000, 100)
    [row] = db.session_metrics_rows("nightly_review")
    assert row["session"]["closed_by"] == "user"
    assert row["session"]["summary_input_tokens"] == 1000
    assert (row["must_dos_day"], row["must_dos_planned"], row["must_dos_done"]) == ("2026-11-25", 1, 0)
