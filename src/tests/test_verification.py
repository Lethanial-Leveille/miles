"""The verification decision, apart from everything verify_voice does around it.

A speculative turn asks this of the early embedding before the endpoint, and
verify_voice asks it afterwards. If the two ever disagreed, a speculation could
send a personal prompt for a voice the real check refuses, so there is one rule
and verify_voice is pinned to it.
"""

import ast
import pathlib

import numpy as np
import pytest

import config
import verification as v

AUDIO_PY = pathlib.Path(__file__).resolve().parent.parent / "audio.py"

PRINT = np.array([1.0, 0.0])


def _at(similarity):
    """A unit vector whose cosine with PRINT is exactly this."""
    return np.array([similarity, np.sqrt(1.0 - similarity ** 2)])


def test_too_little_voice_is_no_audio_not_a_rejection():
    verdict = v.before_embedding(v.MIN_EMBED_SECONDS - 0.01)
    assert verdict == v.Verdict(v.NO_AUDIO, None, v.NO_AUDIO)


def test_a_short_clip_in_a_trusted_session_is_trusted():
    verdict = v.before_embedding(v.MIN_TRUSTWORTHY_SECONDS - 0.01, session_trusted=True)
    assert verdict == v.Verdict(v.VERIFIED, None, "session_trust")


def test_a_short_clip_outside_a_session_must_be_scored():
    assert v.before_embedding(v.MIN_TRUSTWORTHY_SECONDS - 0.01) is None


def test_a_long_clip_is_scored_even_in_a_session():
    assert v.before_embedding(v.MIN_TRUSTWORTHY_SECONDS, session_trusted=True) is None


def test_an_encoder_that_declines_logs_nothing():
    assert v.on_embedding(None, PRINT) == v.Verdict(v.NO_AUDIO, None, None)


@pytest.mark.parametrize("similarity,outcome,logged", [
    (config.VERIFY_THRESHOLD + 0.01, v.VERIFIED, "scored"),
    (config.VERIFY_THRESHOLD, v.VERIFIED, "scored"),
    ((config.VERIFY_THRESHOLD + config.VERIFY_RETRY_THRESHOLD) / 2, v.RETRY, "retry"),
    (config.VERIFY_RETRY_THRESHOLD - 0.01, v.REJECTED, "scored"),
])
def test_the_score_lands_in_the_right_band(similarity, outcome, logged):
    verdict = v.on_embedding(_at(similarity), PRINT)
    assert (verdict.outcome, verdict.logged_as) == (outcome, logged)
    assert verdict.similarity == pytest.approx(similarity)


def test_the_thresholds_are_read_when_deciding(monkeypatch):
    """Read from config at call time, as verify_voice always did, so a threshold
    change can never leave the two answering differently."""
    monkeypatch.setattr(config, "VERIFY_THRESHOLD", 0.9)
    monkeypatch.setattr(config, "VERIFY_RETRY_THRESHOLD", 0.85)
    assert v.on_embedding(_at(0.8), PRINT).outcome == v.REJECTED


def test_decide_skips_scoring_when_the_clip_is_too_short():
    assert v.decide(0.1, None, PRINT).outcome == v.NO_AUDIO


def test_decide_scores_when_it_has_to():
    assert v.decide(3.0, _at(0.99), PRINT).outcome == v.VERIFIED


def test_verify_voice_decides_through_this_module_and_scores_nothing_itself():
    """audio.py cannot be imported while miles-voice holds the mic, so this reads
    it as text. A cosine or a threshold comparison written back into
    verify_voice would be a second copy of the rule."""
    tree = ast.parse(AUDIO_PY.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "verify_voice")
    source = ast.unparse(fn)
    assert "verification.before_embedding(" in source
    assert "verification.on_embedding(" in source
    assert "np.dot" not in source and "linalg" not in source
    assert "similarity >= VERIFY_THRESHOLD" not in source
    assert "VERIFY_RETRY_THRESHOLD" not in source
