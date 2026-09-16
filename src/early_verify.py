"""Voice verification started during the endpoint wait, beside the speculative
transcription.

Verification used to start only after the transcript came back, and it has
nothing to wait for: the transcript is only logged. Measured Sep 16 2026 on six
archived recordings, whisper alone took 1019ms and the encoder 150ms; run
together, 1055ms and 212ms, so the encoder finishes inside whisper's time for
about 36ms of whisper's. Live, verify_ms was a median 264ms on the path to the
Claude call.

Kept out of audio.py so it can be imported and tested while miles-voice holds
the microphone.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import speaker_encoder
from config import RATE, SPEAKER_ENCODER

# One at a time. Two encoder runs would compete with each other and with
# whisper for the same four cores.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="early_verify")


def prepare(command, prepended=None):
    """What verification scores: the command as int16 samples, with the wake
    word ahead of it on a first turn, trimmed. The one place this is built, so
    the early run and the late one can never prepare it differently. Checked
    against the old path based version on four real recordings: identical
    samples, embedding cosine 1.0."""
    samples = command if prepended is None else np.concatenate([prepended, command])
    return speaker_encoder.trim(samples.astype(np.float32) / 32768.0, source_sr=RATE)


def _embed(command, prepended):
    wav = prepare(command, prepended)
    return wav, speaker_encoder.get_encoder(SPEAKER_ENCODER)(wav)


def start(frames, prepended=None):
    """Embed captured frames on the worker. Returns a future of (wav, embedding).

    The frames are copied into an array here, because the speculative file on
    disk can be overwritten by the next speculation while this is still
    running."""
    command = np.frombuffer(b"".join(frames), dtype=np.int16).copy()
    return _worker.submit(_embed, command, prepended)
