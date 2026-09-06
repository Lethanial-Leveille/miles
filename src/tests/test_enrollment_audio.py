"""The enrollment recordings are kept, and they match the embeddings.

enroll.py wrote every sample to one temp file and overwrote it, saving only the
embeddings. An embedding is locked to the encoder that produced it: Resemblyzer
is 256 dimensions, ECAPA is 192, and there is no conversion between them. So
changing encoder meant re recording all twelve samples, and would again for the
encoder after that.

This is the April voiceprint lesson one step further on. That failure was that
only the mean was saved, so a bad sample could not be identified afterwards, and
keeping the individual embeddings fixed it for analysis while leaving the same
hole for re embedding. Audio is the only artifact every future encoder can read.

The property under test is correspondence: file N is the recording that
produced embedding N. A directory that merely contains twelve wavs is worthless
if nobody can say which is which.
"""

import os
import wave

import pytest

import enroll


def _samples(count, condition="near, normal voice"):
    """Distinguishable audio per sample, so a mixed up order is detectable.

    Each recording is a different length, which is enough to tell them apart
    without decoding anything."""
    return [(bytes((i + 1) * 200), f"{condition} {i}") for i in range(count)]


def _written(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".wav"))


def test_every_accepted_sample_is_written(tmp_path):
    names = enroll.save_enrollment_audio(_samples(12), 2, str(tmp_path))
    assert len(names) == 12
    assert len(_written(str(tmp_path))) == 12


def test_filenames_carry_the_embedding_index(tmp_path):
    """Index first, so the mapping to the npz arrays is readable off the name.

    Without it the correspondence is implied by sort order, which survives
    exactly until a condition is renamed."""
    names = enroll.save_enrollment_audio(_samples(3), 2, str(tmp_path))
    assert names[0].startswith("sample_00_")
    assert names[1].startswith("sample_01_")
    assert names[2].startswith("sample_02_")


def test_the_returned_order_matches_the_input_order(tmp_path):
    """What makes file N the recording behind embedding N.

    Both lists are appended in the same branch of the accept loop, so this
    holds as long as nothing writes files independently of that list."""
    samples = _samples(4)
    names = enroll.save_enrollment_audio(samples, 2, str(tmp_path))
    for (audio_bytes, _), name in zip(samples, names):
        with wave.open(os.path.join(str(tmp_path), name), 'rb') as wf:
            assert wf.getnframes() * 2 == len(audio_bytes)


def test_a_previous_run_is_cleared_first(tmp_path):
    """The reason all of this is written at the end rather than per sample.

    An old sample_07 left beside a fresh sample_00 through sample_05 is
    silently wrong: the directory looks like a seven sample enrollment and
    nothing about it appears off. Clearing and writing in one step from one
    list makes that impossible rather than merely unlikely."""
    enroll.save_enrollment_audio(_samples(8), 2, str(tmp_path))
    assert len(_written(str(tmp_path))) == 8

    enroll.save_enrollment_audio(_samples(3), 2, str(tmp_path))
    assert _written(str(tmp_path)) == [
        "sample_00_near_normal_voice_0.wav",
        "sample_01_near_normal_voice_1.wav",
        "sample_02_near_normal_voice_2.wav",
    ]


def test_non_wav_files_are_left_alone(tmp_path):
    """Only the artifacts this function owns are removed. Anything else in the
    directory belongs to someone else."""
    note = tmp_path / "README.txt"
    note.write_text("do not delete these, they are what a future encoder reads")
    enroll.save_enrollment_audio(_samples(2), 2, str(tmp_path))
    assert note.exists()


def test_the_directory_is_created_if_absent(tmp_path):
    target = str(tmp_path / "not_yet")
    enroll.save_enrollment_audio(_samples(2), 2, target)
    assert len(_written(target)) == 2


def test_conditions_become_readable_filenames(tmp_path):
    """The condition is in the name because per condition analysis is one of
    the stated reasons for keeping the samples at all."""
    names = enroll.save_enrollment_audio(
        [(b"\x00\x00", "far, projected")], 2, str(tmp_path))
    assert names == ["sample_00_far_projected.wav"]


def test_no_samples_writes_nothing_and_does_not_raise(tmp_path):
    assert enroll.save_enrollment_audio([], 2, str(tmp_path)) == []
    assert _written(str(tmp_path)) == []
