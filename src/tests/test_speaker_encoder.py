"""Two embedding spaces must never be mixed, and the label is what stops it.

Resemblyzer cannot separate Lethanial from his sister: measured EER 22.3%
against ECAPA's 4.8%, with an impostor max of 0.843 sitting above his own 0.764
genuine median. So an encoder swap is coming, and the swap is the dangerous
part rather than the destination.

Resemblyzer produces 256 dimensions and ECAPA 192, in unrelated spaces. A
centroid built from one and scored by the other does not give a slightly wrong
similarity, it gives a meaningless one, and every turn afterwards is decided by
noise. Nothing about that looks wrong from the outside, which is exactly the
shape of the failure that poisoned the April voiceprint.

These tests pin the guards. Neither encoder is loaded here; that needs torch
and several seconds, and none of what is being checked depends on it.
"""

import json
import os

import numpy as np
import pytest

import config
import speaker_encoder as se


def _voiceprint(tmp_path, dimensions):
    path = str(tmp_path / "voiceprint.npy")
    vector = np.ones(dimensions, dtype=np.float32)
    return path, vector / np.linalg.norm(vector)


# ── provenance is written down ──

def test_saving_records_the_encoder(tmp_path):
    path, vector = _voiceprint(tmp_path, 256)
    se.save_voiceprint(vector, "resemblyzer", path)

    assert se.voiceprint_encoder(path) == "resemblyzer"
    with open(str(tmp_path / "voiceprint.json")) as f:
        assert json.load(f)["dimensions"] == 256


def test_saving_a_mislabelled_voiceprint_is_refused(tmp_path):
    """The label is the only thing standing between a centroid and being
    scored by the wrong encoder, so a wrong one is worse than none."""
    path, vector = _voiceprint(tmp_path, 192)
    with pytest.raises(ValueError, match="256 dimensions"):
        se.save_voiceprint(vector, "resemblyzer", path)


def test_no_file_is_left_behind_by_a_refused_save(tmp_path):
    path, vector = _voiceprint(tmp_path, 192)
    with pytest.raises(ValueError):
        se.save_voiceprint(vector, "resemblyzer", path)
    assert not os.path.exists(path)


# ── loading refuses a mismatch ──

def test_loading_under_the_wrong_encoder_fails_hard(tmp_path):
    """SystemExit rather than a warning.

    The alternative is a service that starts happily and then decides every
    verification by noise. A boot that refuses to come up is the loud version
    of the same problem and is trivially diagnosable."""
    path, vector = _voiceprint(tmp_path, 256)
    se.save_voiceprint(vector, "resemblyzer", path)

    with pytest.raises(SystemExit, match="ecapa"):
        se.load_voiceprint(path, "ecapa")


def test_the_dimension_is_checked_even_when_the_label_agrees(tmp_path):
    """Both checks exist because either can pass while the other fails.

    The label catches two encoders that happen to share a dimension. The
    dimension catches a sidecar that is stale, hand edited, or was copied from
    another machine alongside the wrong .npy."""
    path, vector = _voiceprint(tmp_path, 192)
    np.save(path, vector)
    with open(str(tmp_path / "voiceprint.json"), "w") as f:
        json.dump({"encoder": "resemblyzer", "dimensions": 256}, f)

    with pytest.raises(SystemExit, match="different encoder"):
        se.load_voiceprint(path, "resemblyzer")


def test_an_unlabelled_voiceprint_loads_with_a_warning(tmp_path, capsys):
    """Every voiceprint built before Sep 6 2026 has no sidecar, including the
    one in production. Refusing them would mean this change could not ship
    until after a re enrollment, which is the wrong order."""
    path, vector = _voiceprint(tmp_path, 256)
    np.save(path, vector)

    loaded = se.load_voiceprint(path, "resemblyzer")
    assert loaded.shape == (256,)
    assert "no recorded encoder" in capsys.readouterr().out


def test_an_unlabelled_voiceprint_of_the_wrong_size_is_still_refused(tmp_path):
    """The dimension check is what makes the unlabelled case safe. Without it,
    accepting unlabelled voiceprints would accept an ECAPA one under
    Resemblyzer purely because nobody had written the label."""
    path, vector = _voiceprint(tmp_path, 192)
    np.save(path, vector)
    with pytest.raises(SystemExit):
        se.load_voiceprint(path, "resemblyzer")


def test_a_missing_sidecar_reports_none_rather_than_guessing(tmp_path):
    path, vector = _voiceprint(tmp_path, 256)
    np.save(path, vector)
    assert se.voiceprint_encoder(path) is None


def test_an_unreadable_sidecar_reports_none(tmp_path):
    path, vector = _voiceprint(tmp_path, 256)
    np.save(path, vector)
    (tmp_path / "voiceprint.json").write_text("{not json")
    assert se.voiceprint_encoder(path) is None


# ── the registry ──

def test_an_unknown_encoder_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown speaker encoder"):
        se.get_encoder("wav2vec")


def test_every_declared_encoder_has_a_dimension():
    """A loader without a declared dimension silently disables the size check,
    which is half the protection."""
    assert set(se.LOADERS) <= set(se.DIMENSIONS)


# ── thresholds are per encoder, and today's behaviour is unchanged ──

def test_the_live_thresholds_are_exactly_what_they_were():
    """The regression that matters most in this change.

    VERIFY_RETRY_THRESHOLD and VOICEPRINT_LEARN_MIN_SIMILARITY became ratios of
    the accept bar rather than absolutes. Under resemblyzer they must still
    produce 0.45 and 0.75, the values that were tuned against real rejections,
    or this refactor quietly retuned the live system."""
    assert config.SPEAKER_ENCODER == "resemblyzer"
    assert config.VERIFY_THRESHOLD == 0.5
    assert config.VERIFY_RETRY_THRESHOLD == pytest.approx(0.45)
    assert config.VOICEPRINT_LEARN_MIN_SIMILARITY == pytest.approx(0.75)


def test_the_bands_scale_with_the_encoder():
    """Why they are ratios at all.

    A fixed 0.45 retry band sits just under resemblyzer's 0.5 accept bar and
    far ABOVE ecapa's 0.30, where it would make every accepted turn also look
    ambiguous. A fixed 0.75 learn bar is unreachable under ecapa, whose genuine
    median is 0.542, so the voiceprint would silently stop learning forever
    with no error and no log line."""
    ecapa = config.VERIFY_THRESHOLDS["ecapa"]
    assert ecapa * config.VERIFY_RETRY_RATIO < ecapa
    assert ecapa * config.VOICEPRINT_LEARN_RATIO > ecapa
    assert ecapa * config.VOICEPRINT_LEARN_RATIO < 0.542, (
        "the learn bar must sit below his measured ecapa genuine median, or "
        "nothing will ever clear it")


def test_every_encoder_has_a_threshold():
    assert set(se.LOADERS) == set(config.VERIFY_THRESHOLDS)
