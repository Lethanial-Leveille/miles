"""Reads one channel out of a multichannel microphone, so the pipeline stays mono.

The ReSpeaker XVF3800 only offers two channels, and everything downstream (the
wake model, webrtcvad, Whisper, Deepgram, every WAV written) was built for one.
Picking the channel at the stream means none of the six readers change, and a
read of N frames still returns N frames, so get_read_available stays true too.

Channel 0 is the one used. Sep 30 2026, both channels scored against the live
Seiren on the same takes: channel 0 woke 10 of 12, channel 1 missed one channel
0 caught (0.29 against 0.69). One session in one room, so worth rechecking.

It also brings the array's 48kHz down to the pipeline's 16kHz. The 2.1.1 firmware,
which the jack needs to play at full quality, offers 48kHz only, for capture
too. Every reader still gets RATE: a read of N returns N frames at 16kHz.

Kept out of audio.py so it can be tested while miles-voice holds the microphone.
"""

import numpy as np
from scipy import signal

# Flat to 6.5kHz, 82dB down from 8.8kHz, whose content would otherwise fold into
# the speech band. Measured on the Pi Oct 1 2026: 0.6ms per 80ms read for both
# channels, 1.25ms of delay. scipy's own resample_poly filter rejects only 22dB.
_TAPS = 121
_CUTOFF_HZ = 7500
_KAISER_BETA = 8.0


def decimation(capture_rate, rate):
    """How many captured samples make one the pipeline hears. Only a whole
    number works: keeping every third sample is exact, every 2.76th is not."""
    if capture_rate % rate:
        raise ValueError(f"{capture_rate}Hz is not a whole multiple of {rate}Hz")
    return capture_rate // rate


class ChannelPick:
    """The PyAudio input stream, returning only one of its channels at the
    pipeline's rate. Everything else passes straight through, so no caller
    changes."""

    def __init__(self, stream, channels, pick, capture_rate=16000, rate=16000):
        if not 0 <= pick < channels:
            raise ValueError(f"channel {pick} does not exist in {channels}")
        self._stream = stream
        self._channels = channels
        self._pick = pick
        self._decimate = decimation(capture_rate, rate)
        if self._decimate > 1:
            self._taps = signal.firwin(_TAPS, _CUTOFF_HZ, window=("kaiser", _KAISER_BETA),
                                       fs=capture_rate)
            # The filter's memory of the previous read, one column per channel.
            # Every channel is filtered on every read, even the one not returned:
            # barge in reads channel 1 for a whole reply, and channel 0 coming
            # back with memory seconds old would click at the start of a listen.
            self._state = np.zeros((_TAPS - 1, channels))

    def read(self, frames, *args, channel=None, **kwargs):
        """channel overrides the pick for this read only. Barge in uses it: the
        processed channel chops his voice while hers plays, and the recognition
        channel does not. See BARGE_IN_CHANNEL in config.py."""
        data = self._stream.read(frames * self._decimate, *args, **kwargs)
        if self._channels == 1 and self._decimate == 1:
            return data
        pick = self._pick if channel is None else channel
        if self._channels == 1:
            pick = 0
        # Interleaved S16_LE: one frame is every channel's sample, side by side.
        samples = np.frombuffer(data, dtype=np.int16).reshape(-1, self._channels)
        if self._decimate == 1:
            return samples[:, pick].tobytes()
        filtered, self._state = signal.lfilter(self._taps, 1.0, samples, axis=0,
                                               zi=self._state)
        # Every read is a whole multiple of decimate, so [::decimate] lands on
        # the same phase each time and the reads join without a seam.
        kept = filtered[::self._decimate, pick]
        return np.clip(np.round(kept), -32768, 32767).astype(np.int16).tobytes()

    def get_read_available(self):
        """In the pipeline's frames, so flush_input never asks for more than is
        queued and blocks."""
        return self._stream.get_read_available() // self._decimate

    def __getattr__(self, name):
        return getattr(self._stream, name)
