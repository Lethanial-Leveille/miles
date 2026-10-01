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


def test_one_read_can_ask_for_the_other_channel():
    """Barge in reads the recognition channel; the next plain read is back to the pick."""
    raw = _interleave([1, 2, 3], [-1, -2, -3])
    stream = ChannelPick(FakeStream(raw), channels=2, pick=0)
    assert np.frombuffer(stream.read(3, channel=1), dtype=np.int16).tolist() == [-1, -2, -3]
    assert np.frombuffer(stream.read(3), dtype=np.int16).tolist() == [1, 2, 3]


def test_a_mono_mic_ignores_the_channel_request():
    """So rolling back to a single capsule never breaks barge in."""
    raw = np.array([5, 6, 7], dtype=np.int16).tobytes()
    assert ChannelPick(FakeStream(raw), channels=1, pick=0).read(3, channel=1) == raw


# ── 48kHz down to 16kHz, for the 2.1.1 firmware ──

from scipy import signal

from mic_channel import decimation


class StreamOf:
    """Serves interleaved audio a read at a time, like the device would."""

    def __init__(self, samples):
        self.samples = samples
        self.at = 0
        self.asked = []

    def read(self, frames, exception_on_overflow=True):
        self.asked.append(frames)
        out = self.samples[self.at:self.at + frames]
        self.at += frames
        return out.astype(np.int16).tobytes()

    def get_read_available(self):
        return len(self.samples) - self.at


def _tone(hz, seconds=1.0, rate=48000, amplitude=8000):
    t = np.arange(int(rate * seconds)) / rate
    return amplitude * np.sin(2 * np.pi * hz * t)


def _stereo(left, right):
    return np.column_stack([left, right])


def _read_all(stream, chunk=1280, channel=None):
    reads = []
    while stream._stream.get_read_available() >= chunk * 3:
        reads.append(np.frombuffer(stream.read(chunk, channel=channel), dtype=np.int16))
    return np.concatenate(reads)


def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)) + 1e-9)


def test_only_a_whole_ratio_is_accepted():
    assert decimation(48000, 16000) == 3
    assert decimation(16000, 16000) == 1
    with pytest.raises(ValueError):
        decimation(44100, 16000)


def test_a_read_of_n_asks_the_device_for_three_n():
    fake = StreamOf(_stereo(_tone(1000), _tone(1000)))
    out = ChannelPick(fake, 2, 0, capture_rate=48000).read(1280)
    assert fake.asked == [3840]
    assert len(out) == 1280 * 2


def test_reads_join_without_a_seam():
    """The filter remembers the previous read. Restarted per read, every 80ms
    frame would click, and no test of a single read would see it."""
    left = _tone(440) + _tone(3000, amplitude=3000)
    stream = ChannelPick(StreamOf(_stereo(left, np.zeros_like(left))), 2, 0,
                         capture_rate=48000)
    chunked = _read_all(stream)

    taps = stream._taps
    whole = np.round(signal.lfilter(taps, 1.0, left.astype(np.int16))[::3])
    assert np.max(np.abs(chunked - whole[:len(chunked)])) <= 1


def test_speech_band_keeps_its_level():
    stream = ChannelPick(StreamOf(_stereo(_tone(1000), _tone(1000))), 2, 0,
                         capture_rate=48000)
    out = _read_all(stream)[200:]   # past the filter's first taps
    assert abs(_db(out) - _db(_tone(1000, rate=16000))) < 0.2


def test_a_tone_above_eight_kilohertz_does_not_fold_into_speech():
    """12kHz kept as every third sample would read as a loud 4kHz tone."""
    stream = ChannelPick(StreamOf(_stereo(_tone(12000), _tone(12000))), 2, 0,
                         capture_rate=48000)
    out = _read_all(stream)[200:]
    assert _db(out) < _db(_tone(12000)) - 60


def test_channel_zero_is_seamless_after_barge_in_read_channel_one():
    """Barge in reads channel 1 for a whole reply. Channel 0's filter has to
    have kept up, or the next listen starts with a click."""
    left = _tone(440)
    stream = ChannelPick(StreamOf(_stereo(left, _tone(700))), 2, 0, capture_rate=48000)
    for _ in range(5):
        stream.read(1280, channel=1)
    after = np.frombuffer(stream.read(1280), dtype=np.int16)

    whole = np.round(signal.lfilter(stream._taps, 1.0, left.astype(np.int16))[::3])
    assert np.max(np.abs(after - whole[5 * 1280:6 * 1280])) <= 1


def test_available_is_counted_in_the_pipelines_frames():
    """flush_input reads what is available; counted at 48kHz it would ask for
    three times what is queued and block."""
    fake = StreamOf(_stereo(np.zeros(3840 + 2), np.zeros(3840 + 2)))
    assert ChannelPick(fake, 2, 0, capture_rate=48000).get_read_available() == 1280


def test_at_sixteen_kilohertz_nothing_is_filtered():
    """The 2.0.6 firmware and the rollback: audio passes exactly as before."""
    left = np.arange(-640, 640)
    fake = StreamOf(_stereo(left, np.zeros_like(left)))
    out = ChannelPick(fake, 2, 0).read(1280)
    assert np.frombuffer(out, dtype=np.int16).tolist() == left.tolist()
    assert fake.asked == [1280]
