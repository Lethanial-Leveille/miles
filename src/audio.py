import os
import re
import sys
import time
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
import shutil
from collections import deque
from datetime import datetime

with silence_stderr():
    from openwakeword.model import Model
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.hparams import sampling_rate as RESEMBLYZER_SR
    import webrtcvad

from config import (
    CHUNK, CHANNELS, RATE,
    WHISPER_MODEL, WHISPER_CLI, WHISPER_AUDIO_CTX, TEMP_WAV,
    WAKE_MODEL_PATH, VOICEPRINT_PATH,
    VAD_MODE, VAD_PREROLL_MS, VAD_ONSET_FRAMES, SILENCE_LIMIT, MAX_RECORD,
    EXPECTED_MIC_GAIN, MIC_MIXER_CARD, MIC_MIXER_CONTROL,
    ARCHIVE_RECORDINGS, ARCHIVE_DIR, ARCHIVE_MAX_FILES,
    TTS_FLUSH_MARGIN_MS,
)
from database import log_verification
import timing

FORMAT = pyaudio.paInt16

# ── Voice activity detection ──
# webrtcvad accepts only 10, 20, or 30ms frames. 480 samples at 16kHz is
# exactly 30ms, which is what the capture loops already read, so the frame
# size did not have to change to adopt this.
VAD_FRAME     = 480
VAD_FRAME_MS  = 30
_vad          = webrtcvad.Vad(VAD_MODE)

PREROLL_FRAMES = VAD_PREROLL_MS // VAD_FRAME_MS


def _is_speech(frame):
    """True when webrtcvad classifies this 30ms frame as speech.

    Spectral rather than amplitude based, so steady broadband noise (an AC
    unit) reads as non speech no matter how loud it is. Not speaker aware:
    a television still counts as speech."""
    return _vad.is_speech(frame, RATE)


def flush_input(margin_ms=None):
    """Discard buffered mic input after Nova speaks.

    The input stream fills the whole time she is talking, so her own voice is
    waiting in the buffer when the next listen begins. webrtcvad classifies it
    as speech, correctly, and opens a recording on it. That is the same
    runaway the AC used to cause, except a spectral VAD cannot reject this one
    because Nova's voice really is speech. A shared mic and speaker enclosure
    makes it worse by adding a structure borne path.

    Drains what is queued rather than discarding a fixed duration: a ten
    second response leaves ten seconds buffered, and a fixed flush would leave
    most of it in place."""
    margin_ms = TTS_FLUSH_MARGIN_MS if margin_ms is None else margin_ms

    while True:
        available = stream.get_read_available()
        if available < VAD_FRAME:
            break
        # Capped at what is actually available so this can never block.
        stream.read(min(available, 4096), exception_on_overflow=False)

    # Reading in real time both waits out the tail and throws it away.
    for _ in range(int(margin_ms / VAD_FRAME_MS)):
        stream.read(VAD_FRAME, exception_on_overflow=False)

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

    MIN_RECORD        = 1.0

    frames          = []
    silent_chunks   = 0
    longest_pause   = 0
    chunks_for_silence = int(SILENCE_LIMIT / 0.03)
    max_chunks      = int(MAX_RECORD / 0.03)
    min_chunks      = int(MIN_RECORD / 0.03)
    total_chunks    = 0
    last_speech_at  = None

    while total_chunks < max_chunks:
        data  = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        # Endpoint on speech absence rather than quiet. Room noise loud enough
        # to stay above an amplitude floor used to hold the recording open
        # until MAX_RECORD.
        if _is_speech(data):
            # Longest pause the speaker took and then spoke through. This is
            # the number SILENCE_LIMIT has to clear; if it creeps up toward
            # the limit, turns are ending on hesitation rather than on a
            # finished thought.
            longest_pause = max(longest_pause, silent_chunks)
            silent_chunks = 0
            last_speech_at = time.monotonic()
        else:
            silent_chunks += 1

        if total_chunks > min_chunks and silent_chunks >= chunks_for_silence:
            break

    timing.mark('max_pause_ms', longest_pause * 30.0)
    if total_chunks >= max_chunks:
        print(f"NOTE: hit the {MAX_RECORD:.0f}s recording cap, speech may be "
              f"truncated.", flush=True)

    # The user's wait starts the moment they stop talking, not when this
    # returns. Captured here because nothing downstream can recover it.
    timing.note_speech_end(last_speech_at)

    print(f"Recorded {total_chunks * 0.03:.1f}s", flush=True)
    _write_wav(frames)
    return TEMP_WAV


def listen_for_followup(timeout=10):
    timeout_chunks    = int(timeout / 0.03)
    max_chunks        = int(MAX_RECORD / 0.03)
    min_chunks        = int(0.5 / 0.03)

    # Sized to hold the full pre roll plus the frames that confirm onset, so
    # the kept audio starts VAD_PREROLL_MS before the first speech frame
    # rather than at it.
    preroll = deque(maxlen=PREROLL_FRAMES + VAD_ONSET_FRAMES)

    waiting_chunks = 0
    speech_chunks  = 0
    speech_started = False

    # Phase 1: wait up to timeout for speech to begin
    while waiting_chunks < timeout_chunks:
        data = stream.read(VAD_FRAME, exception_on_overflow=False)
        preroll.append(data)
        waiting_chunks += 1

        if _is_speech(data):
            speech_chunks += 1
        else:
            speech_chunks = 0

        if speech_chunks >= VAD_ONSET_FRAMES:
            speech_started = True
            break

    if not speech_started:
        return None

    frames = list(preroll)

    # Phase 2: record until silence (mirrors record_command)
    total_chunks   = len(frames)
    silent_chunks  = 0
    longest_pause  = 0
    last_speech_at = time.monotonic()
    while total_chunks < max_chunks:
        data   = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        if _is_speech(data):
            longest_pause = max(longest_pause, silent_chunks)
            silent_chunks = 0
            last_speech_at = time.monotonic()
        else:
            silent_chunks += 1

        if total_chunks > min_chunks and silent_chunks >= int(SILENCE_LIMIT / 0.03):
            break

    timing.note_speech_end(last_speech_at)
    timing.mark('max_pause_ms', longest_pause * 30.0)
    if total_chunks >= max_chunks:
        print(f"NOTE: hit the {MAX_RECORD:.0f}s recording cap, speech may be "
              f"truncated.", flush=True)

    # Counts written frames only. This used to add waiting_chunks, reporting
    # time spent waiting for speech as though it were recorded audio, which is
    # why the logs showed follow ups of 11 to 13 seconds that were nothing of
    # the sort. The wav itself was always correct.
    print(f"Follow up recorded {total_chunks * 0.03:.1f}s", flush=True)
    _write_wav(frames)
    return TEMP_WAV


def archive_recording(wav_path, turn_type):
    """Copy a captured command into the archive and prune the oldest.

    Returns the archived path, or None when archiving is off or fails.
    Failure is never fatal: this exists for diagnostics, and losing a
    recording is not a reason to lose the turn."""
    if not ARCHIVE_RECORDINGS:
        return None
    try:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
        dest  = os.path.join(ARCHIVE_DIR, f"{stamp}_{turn_type}.wav")
        shutil.copy2(wav_path, dest)

        # Prune by name, which sorts chronologically given the timestamp
        # prefix, so this never has to stat every file.
        existing = sorted(f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".wav"))
        for stale in existing[:max(0, len(existing) - ARCHIVE_MAX_FILES)]:
            os.remove(os.path.join(ARCHIVE_DIR, stale))
        return dest
    except OSError as exc:
        print(f"Could not archive recording: {exc}", flush=True)
        return None


def _write_wav(frames):
    wf = wave.open(TEMP_WAV, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(_audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcribe(wav_path):
    with timing.stopwatch('transcribe_ms'):
        result = subprocess.run(
            [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
             "-bs", "1", "-bo", "1", "--no-prints", "--no-timestamps",
             "-ac", str(WHISPER_AUDIO_CTX)],
            capture_output=True, text=True
        )
    return result.stdout.strip()


_MIXER_VALUE = re.compile(r'Capture (\d+) \[(\d+)%\](?: \[([-\d.]+)dB\])?')


def log_mic_gain():
    """Print the capture gain at startup and shout if it has drifted.

    ALSA settings can revert on a kernel or firmware update, or if alsactl
    restore runs against a stale state file. A silent revert does not fail
    loudly, it just quietly halves the data quality of everything recorded
    afterward, so it gets checked where it will be seen."""
    try:
        result = subprocess.run(
            ["amixer", "-c", MIC_MIXER_CARD, "sget", MIC_MIXER_CONTROL],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Mic gain check failed to run: {exc}", flush=True)
        return None

    match = _MIXER_VALUE.search(result.stdout)
    if not match:
        print(f"Mic gain check could not parse amixer output for "
              f"card {MIC_MIXER_CARD} control {MIC_MIXER_CONTROL}.", flush=True)
        return None

    value   = int(match.group(1))
    percent = match.group(2)
    db      = match.group(3)
    detail  = f"{value} [{percent}%]" + (f" [{db}dB]" if db else "")

    if value == EXPECTED_MIC_GAIN:
        print(f"Mic gain: {detail}", flush=True)
    else:
        print(f"WARNING: mic gain is {detail}, expected {EXPECTED_MIC_GAIN}. "
              f"Capture level has changed and logged data from this session is "
              f"not comparable to earlier runs. Restore with: "
              f"amixer -c {MIC_MIXER_CARD} sset {MIC_MIXER_CONTROL} "
              f"{EXPECTED_MIC_GAIN} && sudo alsactl store", flush=True)
    return value


def measure_acoustics(wav_path):
    """Level, signal to noise ratio, and spectral tilt of a captured clip.

    These separate variables that are confounded in normal use. Distance,
    loudness, and vocal effort all move together when you step back from the
    mic and raise your voice to compensate, so block labels cannot tell them
    apart. These three can.

    snr_db compares loud frames to quiet ones within the same clip, so it
    tracks how far the speech sits above the room rather than how loud it is
    in absolute terms. Raising mic gain lifts both and leaves this unchanged,
    which is exactly the property that makes it the useful number.

    spectral_tilt is energy above 1kHz over energy below it, in dB. Projecting
    flattens the glottal pulse and pushes relative energy upward, so tilt
    rises with vocal effort. Distance barely touches it at room scale, so it
    separates "loud because close" from "loud because projecting"."""
    with wave.open(wav_path, 'rb') as wf:
        rate = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    if samples.size < 480:
        return None, None, None

    x = samples.astype(np.float64) / 32768.0

    rms = np.sqrt(np.mean(x ** 2))
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else None

    # Frame energies, then loud versus quiet percentiles. Avoids needing a
    # separate noise recording or a VAD pass to find the silent stretches.
    frames = np.array([
        np.sqrt(np.mean(x[i:i + 480] ** 2)) for i in range(0, len(x) - 480, 480)
    ])
    snr_db = None
    if frames.size >= 4:
        loud  = np.percentile(frames, 90)
        quiet = np.percentile(frames, 10)
        if quiet > 0 and loud > 0:
            snr_db = float(20 * np.log10(loud / quiet))

    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    low  = spectrum[(freqs >= 100) & (freqs < 1000)].sum()
    high = spectrum[(freqs >= 1000) & (freqs < 8000)].sum()
    spectral_tilt = float(10 * np.log10(high / low)) if low > 0 and high > 0 else None

    return (float(rms_dbfs) if rms_dbfs is not None else None), snr_db, spectral_tilt


# verify_voice outcomes. NO_AUDIO is distinct from REJECTED on purpose: one
# means "not your voice", the other means "there was no voice to compare".
VERIFIED = 'verified'
REJECTED = 'rejected'
NO_AUDIO = 'no_audio'
RETRY = 'retry'

# Resemblyzer computes one partial utterance per 1.6s window and zero pads the
# last one. Below roughly a third of that there is not enough voiced audio for
# the embedding to mean anything.
MIN_EMBED_SECONDS = 0.5

# Below this, an embedding is too unstable to reject anyone on. Measured: a
# 0.9s follow up of "Okay, thank you" scored 0.460 against a voiceprint whose
# median live score is 0.781, and was refused. The utterance was genuine; the
# embedding simply had too little to work with.
#
# Short conversational follow ups ("yeah", "thanks", "what about tomorrow")
# are exactly the turns that cannot be scored reliably, so scoring them at all
# produces false rejections and nothing else.
MIN_TRUSTWORTHY_SECONDS = 2.0


def wake_word_audio(seconds=1.5):
    """The tail of openWakeWord's ring buffer, which still holds "hey nova".

    Verification failures are a duration problem and nothing else. Across 149
    scored attempts every rejection came from an utterance under two seconds,
    and there were none at all in 94 attempts above it. Commands like "All
    right" and "Okay, thank you" are simply too short to embed stably, and no
    threshold tuning fixes an embedding that had a second of audio to work with.

    The wake word is free audio of the right speaker, already captured, sitting
    in a buffer nobody reads. openWakeWord fills raw_data_buffer inside
    predict(), and nothing calls predict() between detection and verification,
    so the buffer is frozen with "hey nova" at its tail while the command is
    being recorded. reset() clears the prediction smoothing, not this.

    Returns int16 at RATE, or None if the buffer is unavailable. Never raises:
    losing the prepend costs accuracy, and an exception here costs the turn.
    """
    try:
        buffer = wake_model.preprocessor.raw_data_buffer
        wanted = int(RATE * seconds)
        tail = np.array(list(buffer)[-wanted:], dtype=np.int16)
        return tail if len(tail) else None
    except Exception:
        return None


def verify_voice(wav_path, transcript=None, turn_type='initial', wake_confidence=None,
                 session_trusted=False, recording_path=None):
    from config import VERIFY_THRESHOLD, VERIFY_RETRY_THRESHOLD

    verify_started = time.monotonic()

    # Prepend the wake word for initial turns. A follow up has no wake word,
    # and the buffer would be stale from the turn before, so this is scoped to
    # the only case where the audio genuinely belongs to this utterance.
    prepended = None
    if turn_type == 'initial':
        prepended = wake_word_audio()

    if prepended is not None:
        with wave.open(wav_path, 'rb') as wf:
            command = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        wav = preprocess_wav(np.concatenate([prepended, command]).astype(np.float32) / 32768.0,
                             source_sr=RATE)
    else:
        wav = preprocess_wav(wav_path)

    with wave.open(wav_path, 'rb') as wf:
        duration_seconds = wf.getnframes() / wf.getframerate()
    embedded_duration_seconds = len(wav) / RESEMBLYZER_SR

    rms_dbfs, snr_db, spectral_tilt = measure_acoustics(wav_path)

    # Guard before embedding, not after. embed_utterance() zero pads whatever
    # it is handed up to one partial and returns a confident looking vector
    # for silence, so there is no downstream signal that the input was empty.
    if embedded_duration_seconds < MIN_EMBED_SECONDS:
        print(f"No voiced audio to verify ({embedded_duration_seconds:.2f}s "
              f"after trim, need {MIN_EMBED_SECONDS}s).", flush=True)
        log_verification(
            similarity=0.0,
            accepted=False,
            threshold_used=VERIFY_THRESHOLD,
            transcript=transcript,
            duration_seconds=duration_seconds,
            embedded_duration_seconds=embedded_duration_seconds,
            turn_type=turn_type,
            wake_confidence=wake_confidence,
            outcome=NO_AUDIO,
            rms_dbfs=rms_dbfs,
            snr_db=snr_db,
            spectral_tilt=spectral_tilt,
            recording_path=recording_path,
        )
        return NO_AUDIO

    # Too short to score, but inside a session that already authenticated on a
    # longer utterance. Trust the session rather than reject a genuine speaker
    # on an embedding that cannot support the decision either way.
    #
    # The exposure this accepts is someone speaking into the mic within the
    # follow up window, in the same room, immediately after Lethanial. Against
    # the current action set (weather, timers, reminders, conversation) that is
    # a narrower risk than refusing him every time he says "thanks". Revisit
    # when an action with real consequence is added.
    if session_trusted and embedded_duration_seconds < MIN_TRUSTWORTHY_SECONDS:
        print(f"Too short to verify ({embedded_duration_seconds:.2f}s), "
              f"trusting session.", flush=True)
        log_verification(
            similarity=0.0, accepted=True, threshold_used=VERIFY_THRESHOLD,
            transcript=transcript, duration_seconds=duration_seconds,
            embedded_duration_seconds=embedded_duration_seconds,
            turn_type=turn_type, wake_confidence=wake_confidence,
            outcome='session_trust', rms_dbfs=rms_dbfs, snr_db=snr_db,
            spectral_tilt=spectral_tilt, recording_path=recording_path,
        )
        timing.mark('verify_ms', (time.monotonic() - verify_started) * 1000.0)
        return VERIFIED

    embedding = voice_encoder.embed_utterance(wav)
    similarity = np.dot(embedding, voiceprint) / (
        np.linalg.norm(embedding) * np.linalg.norm(voiceprint)
    )
    # Covers preprocess_wav, the acoustic measures, and embed_utterance: all
    # the work that sits between transcription and the Claude call.
    timing.mark('verify_ms', (time.monotonic() - verify_started) * 1000.0)

    print(f"Voice similarity: {similarity:.3f}", flush=True)
    accepted = similarity >= VERIFY_THRESHOLD
    # An ambiguous band rather than a hard line. The scores between these two
    # values are the model saying it does not know, and answering "I do not
    # know" with silence is the worst of the three available responses.
    ambiguous = (not accepted) and similarity >= VERIFY_RETRY_THRESHOLD

    log_verification(
        similarity=float(similarity),
        accepted=accepted,
        threshold_used=VERIFY_THRESHOLD,
        transcript=transcript,
        duration_seconds=duration_seconds,
        embedded_duration_seconds=embedded_duration_seconds,
        turn_type=turn_type,
        wake_confidence=wake_confidence,
        outcome='retry' if ambiguous else 'scored',
        rms_dbfs=rms_dbfs,
        snr_db=snr_db,
        spectral_tilt=spectral_tilt,
        recording_path=recording_path,
    )

    # Any of these can be None for a degenerate clip, so format defensively
    # rather than letting a log line take down the voice loop.
    def _fmt(value, spec):
        return format(value, spec) if value is not None else "n/a"

    print(f"  level {_fmt(rms_dbfs, '.1f')} dBFS, "
          f"SNR {_fmt(snr_db, '.1f')} dB, "
          f"tilt {_fmt(spectral_tilt, '+.1f')} dB", flush=True)

    if accepted:
        return VERIFIED
    return RETRY if ambiguous else REJECTED
