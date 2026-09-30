"""Reads one channel out of a multichannel microphone, so the pipeline stays mono.

The ReSpeaker XVF3800 only offers two channels, and everything downstream (the
wake model, webrtcvad, Whisper, Deepgram, every WAV written) was built for one.
Picking the channel at the stream means none of the six readers change, and a
read of N frames still returns N frames, so get_read_available stays true too.

Channel 0 is the one used. Sep 30 2026, both channels scored against the live
Seiren on the same takes: channel 0 woke 10 of 12, channel 1 missed one channel
0 caught (0.29 against 0.69). One session in one room, so worth rechecking.

Kept out of audio.py so it can be tested while miles-voice holds the microphone.
"""

import numpy as np


class ChannelPick:
    """The PyAudio input stream, returning only one of its channels. Everything
    else passes straight through, so no caller changes."""

    def __init__(self, stream, channels, pick):
        if not 0 <= pick < channels:
            raise ValueError(f"channel {pick} does not exist in {channels}")
        self._stream = stream
        self._channels = channels
        self._pick = pick

    def read(self, *args, channel=None, **kwargs):
        """channel overrides the pick for this read only. Barge in uses it: the
        processed channel chops his voice while hers plays, and the recognition
        channel does not. See BARGE_IN_CHANNEL in config.py."""
        data = self._stream.read(*args, **kwargs)
        if self._channels == 1:
            return data
        pick = self._pick if channel is None else channel
        # Interleaved S16_LE: one frame is every channel's sample, side by side.
        samples = np.frombuffer(data, dtype=np.int16).reshape(-1, self._channels)
        return samples[:, pick].tobytes()

    def __getattr__(self, name):
        return getattr(self._stream, name)
