"""The evening check in rules, replayed over his real week of Sep 25 to Oct 1 2026.

The workouts, nights and classes below are his own Hevy, Oura and calendar
data from that week, pulled Oct 2 2026. The nights are exact seconds, because
Tuesday's average was 7.997 hours and rounding it to eight changes the answer.
"""

import datetime

import checkin

ET = datetime.timezone(datetime.timedelta(hours=-4))
D = datetime.date

WORKOUTS = [D(2026, 9, 22), D(2026, 9, 28), D(2026, 9, 30)]

NIGHTS = {D(2026, 9, day): s for day, s in [
    (20, 29370), (21, 28710), (22, 28200), (23, 32790), (24, 35250), (25, 28650),
    (26, 27990), (27, 31560), (28, 25320), (29, 29490), (30, 27450)]}
NIGHTS[D(2026, 10, 1)] = 20190


def at(month, day, hour, minute=0):
    return datetime.datetime(2026, month, day, hour, minute, tzinfo=ET)


DSA = [(at(9, 29, 13, 55), "DSA"), (at(9, 30, 10, 40), "DSA"), (at(10, 1, 13, 55), "DSA"),
       (at(10, 1, 0, 0), "Prayer and Bible")]


def replay():
    """Each night of the week at the time the radar would have fired, carrying
    what was said forward, the way the real thing will."""
    said, out = set(), {}
    for now in (at(9, 26, 0, 0), at(9, 27, 0, 0), at(9, 27, 23, 30), at(9, 29, 0, 6),
                at(9, 29, 23, 41), at(9, 30, 23, 30), at(10, 2, 0, 50)):
        found = checkin.points(now, WORKOUTS, NIGHTS, DSA, said)
        said |= {p.key for p in found}
        out[checkin.day_lived(now)] = [p.key.split(":")[0] for p in found]
    return out


def test_the_real_week_says_little():
    assert replay() == {
        D(2026, 9, 25): ["gym_lost"],
        D(2026, 9, 26): [],
        D(2026, 9, 27): [],
        D(2026, 9, 28): ["sleep_low"],
        D(2026, 9, 29): [],
        D(2026, 9, 30): ["sleep_low"],
        D(2026, 10, 1): [],
    }


# ── the day he just lived ──

def test_after_midnight_is_still_the_night_before():
    assert checkin.day_lived(at(10, 2, 0, 30)) == D(2026, 10, 1)
    assert checkin.day_lived(at(10, 1, 23, 30)) == D(2026, 10, 1)


# ── gym ──

def test_on_pace_says_nothing():
    assert checkin.gym_point(WORKOUTS, D(2026, 10, 1)) is None


def test_no_room_left_is_said():
    point = checkin.gym_point([D(2026, 9, 28), D(2026, 9, 30)], D(2026, 10, 2))
    assert point.key == "gym_tight:2026-09-28"
    assert "two" in point.summary


def test_a_lost_week_is_said_once():
    first = checkin.gym_point(WORKOUTS, D(2026, 9, 25))
    assert first.key == "gym_lost:2026-09-21"
    assert checkin.gym_point(WORKOUTS, D(2026, 9, 26), {first.key}) is None


def test_four_this_week_is_quiet_even_on_sunday():
    four = [D(2026, 9, 28) + datetime.timedelta(days=k) for k in range(4)]
    assert checkin.gym_point(four, D(2026, 10, 4)) is None


def test_last_week_does_not_count_toward_this_one():
    assert checkin.gym_point([D(2026, 9, 27)] * 3, D(2026, 10, 3)).key == "gym_lost:2026-09-28"


# His own examples, Oct 2 2026, on the week starting Monday Oct 5.
MON, TUE, WED, THU, FRI = (D(2026, 10, 5 + k) for k in range(5))


def test_monday_then_missing_tuesday_and_wednesday_flags_wednesday():
    assert checkin.gym_point([MON], TUE) is None
    assert checkin.gym_point([MON], WED).key == "gym_gap:2026-10-06"


def test_monday_tuesday_then_missing_wednesday_and_thursday_flags_thursday():
    assert checkin.gym_point([MON, TUE], WED) is None
    point = checkin.gym_point([MON, TUE], THU)
    assert point.key == "gym_gap:2026-10-07"
    assert "two days in a row" in point.summary


def test_single_days_off_are_breaks():
    assert checkin.gym_point([MON, TUE, THU], WED) is None
    assert checkin.gym_point([MON, TUE, THU], FRI) is None


def test_a_gap_is_said_once_and_runs_across_sunday():
    saturday = D(2026, 10, 3)
    first = checkin.gym_point([saturday], MON)
    assert first.key == "gym_gap:2026-10-04"
    assert checkin.gym_point([saturday], TUE, {first.key}) is None


def test_a_gap_said_late_counts_its_days():
    point = checkin.gym_point([MON], THU, {"gym_tight:2026-10-05"})
    assert "three days in a row" in point.summary


# ── sleep ──

def test_one_bad_night_is_not_a_trend():
    nights = {D(2026, 9, 29): 9 * 3600, D(2026, 9, 30): 9 * 3600, D(2026, 10, 1): 5 * 3600}
    assert checkin.sleep_point(nights, D(2026, 10, 1)) is None


def test_a_low_stretch_is_said_once_until_it_recovers():
    # Monday is low. Tuesday is not (one short night of three), so Wednesday
    # slipping again is new, and Thursday continues Wednesday's stretch.
    monday = checkin.sleep_point(NIGHTS, D(2026, 9, 28))
    assert monday.key == "sleep_low:2026-09-27"
    assert checkin.sleep_point(NIGHTS, D(2026, 9, 29)) is None
    wednesday = checkin.sleep_point(NIGHTS, D(2026, 9, 30), {monday.key})
    assert wednesday.key == "sleep_low:2026-09-29"
    assert checkin.sleep_point(NIGHTS, D(2026, 10, 1), {monday.key, wednesday.key}) is None


def test_slipping_again_after_recovering_is_new():
    nights = dict(NIGHTS)
    nights[D(2026, 10, 2)] = 10 * 3600
    nights[D(2026, 10, 3)] = 10 * 3600
    nights[D(2026, 10, 4)] = int(3.5 * 3600)
    nights[D(2026, 10, 5)] = 4 * 3600
    point = checkin.sleep_point(nights, D(2026, 10, 5), {"sleep_low:2026-09-29"})
    assert point.key == "sleep_low:2026-10-04"


def test_a_missing_night_says_nothing():
    nights = dict(NIGHTS)
    del nights[D(2026, 9, 30)]
    assert checkin.sleep_point(nights, D(2026, 10, 1)) is None


def test_the_average_is_spoken_in_words():
    point = checkin.sleep_point(NIGHTS, D(2026, 9, 28))
    assert "seven hours fifty two minutes" in point.summary


# ── tomorrow ──

def test_a_class_with_room_to_spare_is_not_mentioned():
    assert checkin.tomorrow_point(DSA, at(9, 28, 23, 30)) is None


def test_a_class_that_limits_tonight_is_mentioned():
    point = checkin.tomorrow_point(DSA, at(9, 30, 0, 30))
    assert point.key == "tomorrow:2026-09-30"
    assert "10:40am" in point.summary and "2:10am" in point.summary


def test_a_midnight_block_is_not_tomorrows_first_class():
    assert checkin.tomorrow_point([(at(10, 2, 0, 0), "Prayer and Bible")],
                                  at(10, 1, 23, 50)) is None


# ── run: once a night, and what was said survives a restart ──

def _sources(monkeypatch, workouts=WORKOUTS, nights=NIGHTS, events=DSA, fail=()):
    def give(name, value):
        def fetch(*_):
            if name in fail:
                raise ConnectionError(f"{name} down")
            return value
        return fetch
    monkeypatch.setattr(checkin, "_workout_days", give("hevy", workouts))
    monkeypatch.setattr(checkin, "_main_sleeps", give("oura", nights))
    monkeypatch.setattr(checkin, "_tomorrow_events", give("calendar", events))


def test_runs_once_a_night(db, monkeypatch):
    _sources(monkeypatch)
    assert [p.key for p in checkin.run(at(9, 26, 0, 30))] == ["gym_lost:2026-09-21"]
    assert checkin.run(at(9, 26, 1, 15)) == []


def test_a_quiet_night_still_counts_as_run(db, monkeypatch):
    _sources(monkeypatch)
    assert checkin.run(at(9, 29, 23, 41)) == []
    assert checkin.checkin_ran(D(2026, 9, 29))


def test_what_was_said_is_not_said_the_next_night(db, monkeypatch):
    _sources(monkeypatch)
    checkin.run(at(9, 26, 0, 30))
    assert checkin.run(at(9, 26, 23, 50)) == []


def test_a_source_that_is_down_is_skipped_and_named(db, monkeypatch):
    import sqlite3
    import database
    _sources(monkeypatch, fail=("oura",))
    assert [p.key for p in checkin.run(at(9, 26, 0, 30))] == ["gym_lost:2026-09-21"]
    rows = sqlite3.connect(database.DB_PATH).execute(
        "SELECT key, skipped FROM checkin_log").fetchall()
    assert rows == [("gym_lost:2026-09-21", "oura")]


def test_hevy_down_is_not_a_missed_week(db, monkeypatch):
    _sources(monkeypatch, fail=("hevy",))
    assert all(not p.key.startswith("gym") for p in checkin.run(at(9, 26, 0, 30)))


# ── wording and delivery ──

class _FakeClaude:
    def __init__(self, text=None, error=None):
        self.text, self.error, self.asked = text, error, []
        self.messages = self

    def create(self, **kwargs):
        self.asked.append(kwargs)
        if self.error:
            raise self.error
        block = type("Block", (), {"type": "text", "text": self.text})()
        return type("Reply", (), {"content": [block]})()


def _fresh(monkeypatch, claude):
    import alerts
    alerts.clear()
    monkeypatch.setattr(checkin, "_client", claude)
    monkeypatch.setattr(checkin, "_started", set())
    return alerts


def test_outside_the_window_nothing_starts(db, monkeypatch):
    _fresh(monkeypatch, _FakeClaude("unused"))
    _sources(monkeypatch)
    assert checkin.start(at(9, 25, 22, 0)) is None
    assert checkin.start(at(9, 26, 5, 0)) is None


def test_her_words_are_queued_once(db, monkeypatch):
    claude = _FakeClaude("Heads up, the gym week finishes at one.")
    alerts = _fresh(monkeypatch, claude)
    _sources(monkeypatch)
    job = checkin.start(at(9, 26, 0, 30))
    job.result(timeout=5)
    checkin.deliver(job)
    spoken = alerts.take_for_speech()
    assert [a.text for a in spoken] == ["Heads up, the gym week finishes at one."]
    assert "out of reach" in claude.asked[0]["messages"][0]["content"]
    assert checkin.start(at(9, 26, 1, 0)) is None


def test_a_slow_check_in_is_queued_when_it_finishes(db, monkeypatch):
    import threading
    alerts = _fresh(monkeypatch, _FakeClaude("Late, but here."))
    gate = threading.Event()
    _sources(monkeypatch)
    slow = checkin._workout_days
    monkeypatch.setattr(checkin, "_workout_days", lambda: (gate.wait(5), slow())[1])
    job = checkin.start(at(9, 26, 0, 30))
    checkin.deliver(job)
    assert alerts.pending_count() == 0
    gate.set()
    job.result(timeout=5)
    # The callback runs on the worker right after the result is set.
    for _ in range(50):
        if alerts.pending_count():
            break
        threading.Event().wait(0.02)
    assert [a.text for a in alerts.take_for_speech()] == ["Late, but here."]


def test_a_good_night_queues_nothing(db, monkeypatch):
    alerts = _fresh(monkeypatch, _FakeClaude("unused"))
    _sources(monkeypatch)
    checkin.deliver(checkin.start(at(9, 29, 23, 41)))
    assert alerts.pending_count() == 0


def test_claude_down_speaks_the_written_lines(db, monkeypatch):
    alerts = _fresh(monkeypatch, _FakeClaude(error=ConnectionError("offline")))
    _sources(monkeypatch)
    job = checkin.start(at(9, 26, 0, 30))
    job.result(timeout=5)
    checkin.deliver(job)
    assert [a.text for a in alerts.take_for_speech()] == [
        "The gym week finishes at one. Monday starts over."]


def test_the_written_tomorrow_line_says_times_in_words():
    point = checkin.tomorrow_point(DSA, at(9, 30, 0, 30))
    assert point.text == "DSA is at ten forty tomorrow, so asleep by two ten for eight hours."
