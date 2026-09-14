"""Whisper's window by clip length, and where long recordings are cut."""

import numpy as np

import audio_segments as seg
from config import WHISPER_AUDIO_CTX, WHISPER_AUDIO_CTX_LONG

RATE = 16000


def test_normal_commands_keep_the_fast_window():
    """Nine in ten real recordings are under 8.2s. None of them may get slower."""
    assert seg.audio_ctx_for(2.9) == WHISPER_AUDIO_CTX
    assert seg.audio_ctx_for(14.0) == WHISPER_AUDIO_CTX


def test_past_what_was_validated_gets_the_whole_window():
    """The fast window was validated to fourteen seconds, and at eighteen it
    dropped trailing words from a real recording."""
    assert seg.audio_ctx_for(18.0) == WHISPER_AUDIO_CTX_LONG
    assert seg.audio_ctx_for(25.0) == WHISPER_AUDIO_CTX_LONG


def test_a_recording_within_one_window_is_one_piece():
    samples = np.zeros(RATE * 10, dtype=np.int16)
    assert seg.segment_bounds(samples, RATE, 28.0) == [(0, len(samples))]


def test_pieces_cover_everything_and_each_fits_a_window():
    samples = np.random.default_rng(0).normal(0, 3000, RATE * 70).astype(np.int16)
    bounds = seg.segment_bounds(samples, RATE, 28.0)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(samples)
    assert all(nxt[0] == cur[1] for cur, nxt in zip(bounds, bounds[1:])), "a gap or overlap loses words"
    assert all(end - start <= 28.0 * RATE for start, end in bounds)
    assert len(bounds) == 3


def test_a_cut_lands_in_the_breath_not_through_a_word():
    samples = np.random.default_rng(1).normal(0, 3000, RATE * 40).astype(np.int16)
    samples[RATE * 26: RATE * 26 + RATE // 2] = 0   # half a second of breath at 26s
    (_, cut), _ = seg.segment_bounds(samples, RATE, 28.0)
    assert RATE * 26 <= cut <= RATE * 26 + RATE // 2
