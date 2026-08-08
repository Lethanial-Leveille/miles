import os
import sys
import ctypes
import fcntl
import warnings
import logging
from contextlib import contextmanager

# ── Silence noisy native libs before any audio/ML imports ──
os.environ["JACK_NO_START_SERVER"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

_ALSA_HANDLER = ctypes.CFUNCTYPE(
    None,
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p
)
def _alsa_silent(filename, line, function, err, fmt):
    pass
_alsa_cb = _ALSA_HANDLER(_alsa_silent)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(_alsa_cb)
except OSError:
    pass

_JACK_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _jack_silent(msg):
    pass
_jack_cb = _JACK_HANDLER(_jack_silent)
try:
    _jack = ctypes.cdll.LoadLibrary("libjack.so.0")
    _jack.jack_set_error_function(_jack_cb)
    _jack.jack_set_info_function(_jack_cb)
except OSError:
    pass


@contextmanager
def silence_stderr():
    sys.stderr.flush()
    saved   = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


# ── Audio/ML imports (silenced above) ──
import pyaudio
import numpy as np
import wave
import subprocess

with silence_stderr():
    from openwakeword.model import Model
    from resemblyzer import VoiceEncoder, preprocess_wav

from config import (
    CHUNK, CHANNELS, RATE,
    WHISPER_MODEL, WHISPER_CLI, TEMP_WAV,
    WAKE_MODEL_PATH, VOICEPRINT_PATH,
)

FORMAT = pyaudio.paInt16

# ── Hardware init ──
print("Loading wake word model...", flush=True)
with silence_stderr():
    wake_model = Model(wakeword_model_paths=[WAKE_MODEL_PATH])

print("Loading voice encoder...", flush=True)
with silence_stderr():
    voice_encoder = VoiceEncoder()
voiceprint = np.load(VOICEPRINT_PATH)

# ── Mic lock ──
# Exclusive OS level lock, held for the life of this process. Guards against
# a future regression where some other process (like the FastAPI server)
# tries to open the mic again. Without this, a conflict shows up as PyAudio
# silently dropping the busy device from its enumeration, which then looks
# like a missing microphone instead of the real cause.
MIC_LOCK_PATH = os.path.expanduser("~/miles/build/mic.lock")

def _acquire_mic_lock():
    lock_fd = open(MIC_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Mic is already locked by another process. See {MIC_LOCK_PATH}. Refusing to open a second capture stream.", flush=True)
        raise SystemExit(1)
    lock_fd.write(str(os.getpid()))
    lock_fd.flush()
    return lock_fd

_mic_lock_fd = _acquire_mic_lock()

with silence_stderr():
    _audio = pyaudio.PyAudio()

mic_index = None
for i in range(_audio.get_device_count()):
    info = _audio.get_device_info_by_index(i)
    if "Razer" in info["name"] or "Seiren" in info["name"]:
        mic_index = i
        print(f"Found mic: {info['name']} (index {i})", flush=True)
        break

if mic_index is None:
    print("Razer mic not found!", flush=True)
    raise SystemExit(1)

with silence_stderr():
    stream = _audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK,
    )


# ── Audio functions ──

def record_command():
    print("Listening...", flush=True)

    VAD_FRAME         = 480
    SILENCE_THRESHOLD = 200
    SILENCE_LIMIT     = 3.0
    MAX_RECORD        = 15
    MIN_RECORD        = 1.0

    frames          = []
    silent_chunks   = 0
    chunks_for_silence = int(SILENCE_LIMIT / 0.03)
    max_chunks      = int(MAX_RECORD / 0.03)
    min_chunks      = int(MIN_RECORD / 0.03)
    total_chunks    = 0

    while total_chunks < max_chunks:
        data  = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        energy = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        if energy < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if total_chunks > min_chunks and silent_chunks >= chunks_for_silence:
            break

    print(f"Recorded {total_chunks * 0.03:.1f}s", flush=True)
    _write_wav(frames)
    return TEMP_WAV


def listen_for_followup(timeout=10):
    VAD_FRAME         = 480
    SILENCE_THRESHOLD = 200
    SPEECH_THRESHOLD  = 150
    SILENCE_LIMIT     = 3.0
    timeout_chunks    = int(timeout / 0.03)
    max_chunks        = int(15 / 0.03)
    min_chunks        = int(0.5 / 0.03)

    frames         = []
    waiting_chunks = 0
    speech_started = False

    # Phase 1: wait up to timeout for speech to begin
    while waiting_chunks < timeout_chunks:
        data   = stream.read(VAD_FRAME, exception_on_overflow=False)
        energy = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        waiting_chunks += 1
        if energy > SPEECH_THRESHOLD:
            frames.append(data)
            speech_started = True
            break

    if not speech_started:
        return None

    # Phase 2: record until silence (mirrors record_command)
    total_chunks  = 1
    silent_chunks = 0
    while total_chunks < max_chunks:
        data   = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        energy = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        if energy < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if total_chunks > min_chunks and silent_chunks >= int(SILENCE_LIMIT / 0.03):
            break

    print(f"Follow up recorded {(waiting_chunks + total_chunks) * 0.03:.1f}s", flush=True)
    _write_wav(frames)
    return TEMP_WAV


def _write_wav(frames):
    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(_audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
         "-bs", "1", "-bo", "1", "--no-prints", "--no-timestamps"],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def verify_voice(wav_path):
    from config import VERIFY_THRESHOLD
    wav       = preprocess_wav(wav_path)
    embedding = voice_encoder.embed_utterance(wav)
    similarity = np.dot(embedding, voiceprint) / (
        np.linalg.norm(embedding) * np.linalg.norm(voiceprint)
    )
    print(f"Voice similarity: {similarity:.3f}", flush=True)
    return similarity >= VERIFY_THRESHOLD
