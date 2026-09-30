import numpy as np
import pytest

from mic_channel import ChannelPick


class FakeStream:
    def __init__(self, data):
        self.data = data
        self.closed = False

    def read(self, frames, exception_on_overflow=True):
        return self.data

    def close(self):
        self.closed = True


def _interleave(left, right):
    return np.column_stack([left, right]).astype(np.int16).tobytes()


def test_picks_channel_zero_out_of_stereo():
    raw = _interleave([1, 2, 3], [-1, -2, -3])
    out = ChannelPick(FakeStream(raw), channels=2, pick=0).read(3)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [1, 2, 3]


def test_picks_channel_one_out_of_stereo():
    raw = _interleave([1, 2, 3], [-1, -2, -3])
    out = ChannelPick(FakeStream(raw), channels=2, pick=1).read(3)
    assert np.frombuffer(out, dtype=np.int16).tolist() == [-1, -2, -3]


def test_a_read_of_n_frames_is_still_n_frames():
    """Every reader counts in frames, so halving the bytes must not halve them."""
    raw = _interleave(np.arange(1280), np.zeros(1280))
    out = ChannelPick(FakeStream(raw), channels=2, pick=0).read(1280)
    assert len(out) == 1280 * 2


def test_mono_passes_through_untouched():
    raw = np.array([5, 6, 7], dtype=np.int16).tobytes()
    assert ChannelPick(FakeStream(raw), channels=1, pick=0).read(3) == raw


def test_everything_else_reaches_the_stream():
    fake = FakeStream(b"")
    ChannelPick(fake, channels=2, pick=0).close()
    assert fake.closed


def test_a_channel_that_does_not_exist_is_refused():
    with pytest.raises(ValueError):
        ChannelPick(FakeStream(b""), channels=2, pick=2)
