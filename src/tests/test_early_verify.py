"""Verification started beside the speculative transcript. The module is
importable while miles-voice holds the microphone; audio.py is not, so the
wiring there is pinned by parsing it."""

import ast
import pathlib

import numpy as np
import pytest

import early_verify

AUDIO_PY = pathlib.Path(__file__).resolve().parent.parent / "audio.py"


def test_the_wake_word_goes_ahead_of_the_command(monkeypatch):
    seen = {}
    monkeypatch.setattr(early_verify.speaker_encoder, "trim",
                        lambda wav, source_sr: seen.setdefault("wav", wav) * 2)
    wake = np.array([16384, 16384], dtype=np.int16)
    command = np.array([-16384], dtype=np.int16)
    out = early_verify.prepare(command, wake)
    assert seen["wav"].tolist() == [0.5, 0.5, -0.5]
    assert out.tolist() == [1.0, 1.0, -1.0]


def test_a_follow_up_is_the_command_alone(monkeypatch):
    monkeypatch.setattr(early_verify.speaker_encoder, "trim", lambda wav, source_sr: wav)
    assert early_verify.prepare(np.array([16384], dtype=np.int16)).tolist() == [0.5]


def test_the_embedding_is_computed_on_a_copy_of_the_frames(monkeypatch):
    monkeypatch.setattr(early_verify.speaker_encoder, "trim", lambda wav, source_sr: wav)
    monkeypatch.setattr(early_verify.speaker_encoder, "get_encoder",
                        lambda name: (lambda wav: np.array([wav.sum()])))
    frames = [np.array([16384], dtype=np.int16).tobytes(),
              np.array([16384], dtype=np.int16).tobytes()]
    wav, embedding = early_verify.start(frames).result(timeout=5)
    assert wav.tolist() == [0.5, 0.5] and embedding.tolist() == [1.0]


def test_one_embedding_at_a_time():
    """Two encoder runs would fight each other and whisper for four cores."""
    assert early_verify._worker._max_workers == 1


def _source(name):
    tree = ast.parse(AUDIO_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.get_source_segment(AUDIO_PY.read_text(), node)
    raise AssertionError(f"{name} is gone from audio.py")


def test_each_loop_says_which_kind_of_turn_it_is():
    assert "turn_type='initial'" in _source("record_command")
    assert "turn_type='followup'" in _source("listen_for_followup")


def test_only_a_speculation_that_held_is_used():
    source = _source("transcribe")
    assert "not speculation.stale else None" in source


def test_verify_prepares_audio_the_same_way_as_the_early_run():
    source = _source("verify_voice")
    assert "early_verify.prepare(command, prepended)" in source
    assert "speaker_encoder.trim(" not in source


def test_a_resumed_sentence_drops_the_early_embedding():
    assert "self.early.cancel()" in _source("_Speculation")
    assert "_early_verification = None" in _source("cancel_speculation")
