"""The bridge has its own measurement, and never marks a turn as local."""

import time

import timing


def test_the_bridge_is_timed_without_touching_perceived(monkeypatch):
    timing.begin_turn("initial")
    timing.note_speech_end(time.monotonic() - 3.0)
    timing.note_bridge()
    turn = timing._turn
    assert 2900 < turn["stages"]["bridge_ms"] < 3500
    assert "total_perceived_ms" not in turn["stages"]
    assert turn["local_intent"] is False
    timing.note_bridge()
    assert turn["stages"]["bridge_ms"] < 3500, "only the first bridge counts"
    timing.abandon_turn()


def test_a_bridge_with_no_turn_is_ignored():
    timing.abandon_turn()
    timing.note_bridge()


def test_the_bridge_is_written_to_the_log(db, monkeypatch):
    timing.begin_turn("initial")
    timing.note_speech_end(time.monotonic() - 1.0)
    timing.note_bridge()
    timing.end_turn(transcript="how did I sleep", response="Fine.")
    import sqlite3
    row = sqlite3.connect(db.DB_PATH).execute(
        "SELECT bridge_ms, local_intent FROM timing_log").fetchone()
    assert row[0] > 900 and row[1] == 0
