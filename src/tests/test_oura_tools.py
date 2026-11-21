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
    assert out["slept"] == "almost 11 hours"
    assert out["if_he_asks_for_detail"]["sleep_score_out_of_100"] == 89
    assert out["if_he_asks_for_detail"]["deep_sleep"] == "2 hours 6 minutes"
    assert "100" not in str(out["slept"])


def test_sleep_prefers_the_night_over_a_later_nap():
    periods = [_period("long_sleep", "2026-09-13T10:02", 38580),
               _period("late_nap", "2026-09-13T15:30", 1500)]
    assert oura_tools.summarize_sleep(periods, [])["slept"] == "almost 11 hours"


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


# ── said like a person ──

def _night(start, end, total, latency=0, kind="long_sleep", in_bed=None):
    return {"type": kind, "day": end[:10], "bedtime_start": start, "bedtime_end": end,
            "latency": latency, "total_sleep_duration": total,
            "time_in_bed": in_bed or total + latency, "efficiency": 80}


def test_last_night_reads_like_a_person_said_it():
    """Nov 20 2026, from his ring: in bed 1:43, asleep about 2:41, up at 1:27 PM."""
    night = _night("2026-11-20T01:43:26", "2026-11-20T13:26:56", 32220,
                   latency=3480, in_bed=42210)
    nap = _night("2026-11-20T16:55:31", "2026-11-20T17:26:31", 1500, kind="sleep")
    out = oura_tools.summarize_sleep([night, nap], [{"day": "2026-11-20", "score": 75}])
    assert out["slept"] == "about 9 hours"
    assert out["fell_asleep_around"] == "2:40 AM"
    assert out["woke_up_around"] == "1:25 PM"
    assert out["awake_in_bed"] == "almost 3 hours"
    assert out["night"] == "Thursday night"
    assert out["overall"] == "good"
    assert out["naps_since"] == ["about 25 minutes around 4:55 PM"]


def test_compared_with_his_usual_two_weeks():
    usual = [_night(f"2026-11-{d:02d}T00:30:00", f"2026-11-{d:02d}T08:30:00", 7 * 3600)
             for d in range(10, 16)]
    late = _night("2026-11-16T03:00:00", "2026-11-16T09:00:00", 6 * 3600)
    out = oura_tools.summarize_sleep(usual + [late], [])
    assert out["sleep_compared_to_usual"] == "about an hour less than usual"
    assert out["bedtime_compared_to_usual"] == "about 2 and a half hours later than usual"
    assert out["wake_compared_to_usual"] == "about the same as usual"


def test_too_little_history_says_nothing_about_usual():
    out = oura_tools.summarize_sleep(
        [_night("2026-11-16T03:00:00", "2026-11-16T09:00:00", 6 * 3600)], [])
    assert out["sleep_compared_to_usual"] is None


@pytest.mark.parametrize("seconds,words", [
    (7 * 3600, "about 7 hours"), (7.5 * 3600, "about 7 and a half hours"),
    (7.3 * 3600, "a little over 7 hours"), (6.8 * 3600, "almost 7 hours"),
    (3600, "about an hour"), (1500, "about 25 minutes"),
])
def test_about_duration(seconds, words):
    assert oura_tools.about_duration(seconds) == words
