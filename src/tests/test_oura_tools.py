"""The pure parts of the Oura tools. Nothing here touches the network."""

import datetime

import pytest

import oura_tools


@pytest.mark.parametrize("seconds,expected", [
    (38580, "10 hours 43 minutes"),
    (3600, "1 hour"),
    (60, "1 minute"),
    (0, "0 minutes"),
    (7440, "2 hours 4 minutes"),
])
def test_duration_words(seconds, expected):
    assert oura_tools.duration_words(seconds) == expected


def test_duration_words_refuses_what_is_not_a_duration():
    assert oura_tools.duration_words(None) is None
    assert oura_tools.duration_words("89") is None


def _period(kind, end, total, day="2026-09-13"):
    return {"type": kind, "day": day, "bedtime_end": end,
            "total_sleep_duration": total, "time_in_bed": total + 1800,
            "deep_sleep_duration": 7560, "rem_sleep_duration": 6240,
            "efficiency": 89}


def test_sleep_reports_durations_not_contributor_scores():
    """Regression for Sep 13 2026. daily_sleep carries contributors named
    total_sleep and deep_sleep that are scores out of a hundred. Nova read
    total_sleep 100 as a hundred minutes. The duration has to come from the
    sleep period, and the field has to say it is a duration."""
    daily = [{"day": "2026-09-13", "score": 89,
              "contributors": {"total_sleep": 100, "deep_sleep": 98}}]
    out = oura_tools.summarize_sleep([_period("long_sleep", "2026-09-13T10:02", 38580)], daily)
    assert out["total_sleep"] == "10 hours 43 minutes"
    assert out["sleep_score_out_of_100"] == 89
    assert "100" not in str(out["total_sleep"])


def test_sleep_prefers_the_night_over_a_later_nap():
    periods = [_period("long_sleep", "2026-09-13T10:02", 38580),
               _period("late_nap", "2026-09-13T15:30", 1500)]
    assert oura_tools.summarize_sleep(periods, [])["total_sleep"] == "10 hours 43 minutes"


def test_sleep_with_nothing_recorded():
    assert oura_tools.summarize_sleep([], []) is None


def test_heart_rate_is_summarized_in_code():
    now = datetime.datetime(2026, 9, 13, 19, 0, tzinfo=datetime.timezone.utc)
    samples = [
        {"timestamp": "2026-09-13T18:30:00+00:00", "bpm": 60},
        {"timestamp": "2026-09-13T18:50:00+00:00", "bpm": 72},
        {"timestamp": "2026-09-13T18:40:00+00:00", "bpm": 66},
    ]
    out = oura_tools.summarize_heart_rate(samples, now)
    assert out["latest_bpm"] == 72
    assert out["latest_reading_minutes_ago"] == 10
    assert (out["average_bpm"], out["lowest_bpm"], out["highest_bpm"]) == (66, 60, 72)
    assert out["readings"] == 3


def test_heart_rate_with_no_readings():
    assert oura_tools.summarize_heart_rate([], datetime.datetime.now(datetime.timezone.utc)) is None


def test_day_window_ends_tomorrow():
    window = oura_tools._day_window(datetime.date(2026, 9, 13))
    assert window == {"start_date": "2026-09-11", "end_date": "2026-09-14"}
