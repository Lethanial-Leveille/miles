"""get_system_state, Nova's view of her own health.

The contract that matters is degradation: a missing thermal zone or a checkout
without a git directory must produce a null field, never an exception. "How are
you doing" turning into a traceback is a worse outcome than not knowing the
core temperature.
"""

import pytest

import system_state
from tools import Permission, registry


def test_registered_as_a_read_tool():
    spec = registry.get("get_system_state")
    assert spec.permission is Permission.READ
    assert spec.returns_to_model is True
    assert spec.input_schema["properties"] == {}


def test_returns_every_field(db, monkeypatch):
    monkeypatch.setattr(system_state, "DB_PATH", db.DB_PATH)
    out = system_state.get_system_state()
    assert set(out) == {
        "uptime_hours", "turns_served", "median_latency_ms", "whisper_model",
        "active_memories", "core_temp_f", "voice_service_restarts", "git_commit",
    }


def test_whisper_model_is_a_basename_not_a_path():
    """Nova reads this aloud. A full home directory path is not an answer."""
    assert "/" not in system_state.get_system_state()["whisper_model"]


@pytest.mark.parametrize("reader", [
    "_uptime_hours", "_core_temp_f", "_git_commit", "_service_restarts", "_db_stats",
])
def test_a_failing_reader_yields_null_rather_than_raising(monkeypatch, reader):
    def boom():
        raise OSError("unavailable")

    monkeypatch.setattr(system_state, reader, boom)
    out = system_state.get_system_state()          # must not raise
    assert isinstance(out, dict)
    assert "whisper_model" in out                  # unaffected fields survive


def test_database_failure_nulls_only_the_database_fields(monkeypatch):
    monkeypatch.setattr(system_state, "_db_stats", lambda: (_ for _ in ()).throw(OSError()))
    out = system_state.get_system_state()
    assert out["turns_served"] is None
    assert out["active_memories"] is None
    assert out["median_latency_ms"] is None
    assert out["whisper_model"] is not None


def test_median_latency_is_none_when_there_is_no_data(db, monkeypatch):
    """A fresh database has no timed turns. Reporting zero would read as
    instant rather than as unknown."""
    monkeypatch.setattr(system_state, "DB_PATH", db.DB_PATH)
    stats = system_state._db_stats()
    assert stats["median_latency_ms"] is None
    assert stats["turns"] == 0


def test_counts_come_from_the_database(db, monkeypatch):
    monkeypatch.setattr(system_state, "DB_PATH", db.DB_PATH)
    db.log_timing(
        turn_type="wake", action_fired=False, transcript="hi",
        speech_end_to_endpoint_ms=900, transcribe_ms=None, verify_ms=None,
        claude_ttft_ms=None, claude_total_ms=None, tts_ttfb_ms=None,
        tts_first_audio_ms=None, action_ms=None, total_perceived_ms=5000,
    )
    db.log_timing(
        turn_type="wake", action_fired=False, transcript="hi",
        speech_end_to_endpoint_ms=900, transcribe_ms=None, verify_ms=None,
        claude_ttft_ms=None, claude_total_ms=None, tts_ttfb_ms=None,
        tts_first_audio_ms=None, action_ms=None, total_perceived_ms=3000,
    )
    stats = system_state._db_stats()
    assert stats["turns"] == 2
    assert stats["median_latency_ms"] == 4000
