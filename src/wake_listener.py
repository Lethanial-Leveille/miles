"""Hearing the wake word while a command is being recorded.

The capture loops read 30ms frames for webrtcvad; openWakeWord needs 80ms
chunks. This buffers one into the other and reports a detection. The model is
passed in rather than imported, so this can be tested without a microphone or
the model file.
"""

import numpy as np


class WakeListener:
    def __init__(self, predict, threshold, chunk_samples=1280):
        self._predict = predict
        self._threshold = threshold
        self._chunk_bytes = chunk_samples * 2   # int16 is two bytes a sample
        self._buffer = b""

    def feed(self, frame):
        """Add one capture frame. True when the wake word was just heard.

        The buffer is cleared on a detection, so the audio that held the wake
        word cannot fire it a second time on the next frame."""
        self._buffer += frame
        while len(self._buffer) >= self._chunk_bytes:
            chunk = self._buffer[:self._chunk_bytes]
            self._buffer = self._buffer[self._chunk_bytes:]
            scores = self._predict(np.frombuffer(chunk, dtype=np.int16))
            if max(scores.values(), default=0.0) > self._threshold:
                self._buffer = b""
                return True
        return False
