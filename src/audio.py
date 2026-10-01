import os
import re
import sys
import time
import threading
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
    from resemblyzer.hparams import sampling_rate as RESEMBLYZER_SR
    import webrtcvad

import speaker_encoder
import early_verify
import verification
from verification import (VERIFIED, REJECTED, NO_AUDIO, RETRY,
                          MIN_EMBED_SECONDS, MIN_TRUSTWORTHY_SECONDS)

from config import (
    CHUNK, CHANNELS, RATE, MIC_STALL_S, WHISPER_FALLBACK_AFTER_MS,
    WHISPER_MODEL, WHISPER_CLI, WHISPER_SEGMENT_SECONDS, WHISPER_INITIAL_PROMPT,
    TEMP_WAV,
    WAKE_MODEL_PATH, VOICEPRINT_PATH, WAKE_THRESHOLD, BARGE_IN_THRESHOLD,
    BARGE_IN_CHANNEL,
    VAD_MODE, VAD_PREROLL_MS, VAD_ONSET_FRAMES, SILENCE_LIMIT, MAX_RECORD,
    EXPECTED_MIC_GAIN, MIC_MIXER_CARD, MIC_MIXER_CONTROL, MIC_NAME_HINT,
    MIC_CAPTURE_CHANNELS, MIC_CHANNEL, MIC_CAPTURE_RATE,
    capsule_name, SPEAKER_ENCODER,
    ARCHIVE_RECORDINGS, ARCHIVE_DIR, ARCHIVE_MAX_FILES,
    TTS_FLUSH_MARGIN_MS,
    SPECULATIVE_TRANSCRIBE, SPECULATIVE_SILENCE_MS, SPECULATIVE_THREADS,
    WHISPER_BEAM_SIZE,
    SPECULATIVE_WAV,
    CAPTURE_WAKE_MISSES, WAKE_MISS_DIR, WAKE_MISS_MAX_FILES,
    CAPTURE_WAKE_HITS, WAKE_HIT_DIR, WAKE_HIT_MAX_FILES,
    WAKE_GATED_DIR, WAKE_GATED_MAX_FILES,
)
from audio_segments import audio_ctx_for, segment_bounds
from wake_listener import WakeListener
import wake_capture
import deepgram_stt
import speculative_turn
import mic_channel
import mic_watchdog
from database import keep_voiceprint_sample, log_verification
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
    # A second copy, fed only while a command is being recorded. The main
    # model's buffer has to stay frozen on "hey nova" until verification
    # reads it back out (see wake_word_audio), so listening during a
    # recording cannot use it.
    recording_wake_model = Model(wakeword_model_paths=[WAKE_MODEL_PATH])

print(f"Loading voice encoder ({SPEAKER_ENCODER})...", flush=True)
with silence_stderr():
    embed_voice = speaker_encoder.get_encoder(SPEAKER_ENCODER)

# Refuses a voiceprint built by a different encoder rather than scoring against
# it. Two embedding spaces are unrelated, so a mismatch does not produce a
# slightly wrong similarity, it produces a meaningless one, and every turn
# afterwards would be decided by noise. Failing at boot is the loud version.
voiceprint = speaker_encoder.load_voiceprint(VOICEPRINT_PATH, SPEAKER_ENCODER)

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
# Recorded so that a voiceprint sample can be tagged with the capsule that
# produced it. A voiceprint does not transfer across microphones, and samples
# gathered on one must never be folded into a centroid built on another. When
# the array arrives this string changes and the old samples become identifiable
# as belonging to the previous mic rather than silently averaging in.
MIC_NAME = "unknown"
for i in range(_audio.get_device_count()):
    info = _audio.get_device_info_by_index(i)
    if MIC_NAME_HINT in info["name"] and info["maxInputChannels"] > 0:
        mic_index = i
        MIC_NAME = capsule_name(info["name"])
        print(f"Found mic: {info['name']} (index {i})", flush=True)
        break

if mic_index is None:
    print(f"Mic not found: no input device matching {MIC_NAME_HINT!r}.", flush=True)
    raise SystemExit(1)

with silence_stderr():
    stream = _audio.open(
        format=FORMAT,
        channels=MIC_CAPTURE_CHANNELS,
        rate=MIC_CAPTURE_RATE,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=CHUNK * mic_channel.decimation(MIC_CAPTURE_RATE, RATE),
    )

# Every reader below expects mono at RATE. See mic_channel.py.
stream = mic_channel.ChannelPick(stream, MIC_CAPTURE_CHANNELS, MIC_CHANNEL,
                                 capture_rate=MIC_CAPTURE_RATE, rate=RATE)

# A read that never returns is a deaf assistant with every service reporting
# active. The guard ends the process instead, and systemd reopens the device.
stream = mic_watchdog.GuardedStream(stream, MIC_STALL_S)
stream.start_watchdog()


# ── Audio functions ──

def _recording_wake():
    """A fresh listener on the recording copy of the wake model."""
    recording_wake_model.reset()
    return WakeListener(recording_wake_model.predict, WAKE_THRESHOLD,
                        reset=recording_wake_model.reset)


def record_command(on_wake=None):
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

    cancel_speculation()
    spec_chunks = int(SPECULATIVE_SILENCE_MS / 30.0)
    wake = _recording_wake()

    while total_chunks < max_chunks:
        data  = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        if wake.feed(data):
            # "Hey Nova" in the middle of a recording. What came before it was
            # not the command, usually the room, so it is dropped and the
            # command starts here. On Sep 13 2026 a busy room held the mic for
            # the whole cap while he said the wake word into the recording.
            print("Wake word during recording, starting over.", flush=True)
            if on_wake is not None:
                on_wake()
            cancel_speculation()
            frames, total_chunks, silent_chunks, longest_pause = [], 0, 0, 0
            last_speech_at = time.monotonic()
            continue

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

            # He spoke through the pause, so anything started on it covers only
            # part of the turn. Discard and let the next pause try again.
            cancel_speculation()

            # Keeps the Deepgram connection open while he talks, so the request
            # at the pause does not pay a handshake. Rate limited inside.
            deepgram_stt.keep_warm()
        else:
            silent_chunks += 1

            # Start Whisper on what is captured so far, while endpointing keeps
            # waiting out the rest of SILENCE_LIMIT.
            maybe_speculate(frames, silent_chunks, spec_chunks,
                            total_chunks > min_chunks, turn_type='initial')

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


def listen_for_followup(timeout=10, on_wake=None):
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

    cancel_speculation()
    spec_chunks = int(SPECULATIVE_SILENCE_MS / 30.0)
    wake = _recording_wake()

    while total_chunks < max_chunks:
        data   = stream.read(VAD_FRAME, exception_on_overflow=False)
        frames.append(data)
        total_chunks += 1

        if wake.feed(data):
            # "Hey Nova" in the middle of a recording. What came before it was
            # not the command, usually the room, so it is dropped and the
            # command starts here. On Sep 13 2026 a busy room held the mic for
            # the whole cap while he said the wake word into the recording.
            print("Wake word during recording, starting over.", flush=True)
            if on_wake is not None:
                on_wake()
            cancel_speculation()
            frames, total_chunks, silent_chunks, longest_pause = [], 0, 0, 0
            last_speech_at = time.monotonic()
            continue

        if _is_speech(data):
            longest_pause = max(longest_pause, silent_chunks)
            silent_chunks = 0
            last_speech_at = time.monotonic()
            cancel_speculation()
            deepgram_stt.keep_warm()
        else:
            silent_chunks += 1
            maybe_speculate(frames, silent_chunks, spec_chunks,
                            total_chunks > min_chunks, turn_type='followup')

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


def _save_wake_clip(frames, score, directory, max_files, label):
    """Keep the audio behind a wake decision, named by score so it sorts.

    Shared by both directions because they are the same operation and only the
    question differs. wake_misses asks why she did not hear him; wake_hits asks
    why she answered when he said nothing. Neither is answerable from a score
    alone, which is the whole reason wake_log was insufficient.

    Never fatal. These are diagnostics, and losing a clip is not a reason to
    lose the wake loop."""
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")[:-3]
        dest  = os.path.join(directory, f"{score:.3f}_{stamp}.wav")
        _write_wav(frames, dest)

        # Never by name: the name leads with the score so it sorts by severity,
        # which has nothing to do with what is worth keeping. wake_capture
        # prefers the unlabelled, because pruning by age alone spent eleven days
        # eating hand labelled clips.
        for stale in wake_capture.clips_to_evict(directory, max_files):
            try:
                os.remove(stale)
            except OSError:
                # Tolerated per file rather than per call. One clip that has
                # already gone is not a reason to leave the rest of the surplus
                # on disk, which is what a single try around the loop did.
                pass
        return dest
    except OSError as exc:
        print(f"Could not save wake {label}: {exc}", flush=True)
        return None


def save_wake_miss(frames, score):
    """Audio behind a score that did NOT reach the threshold."""
    if not CAPTURE_WAKE_MISSES:
        return None
    return _save_wake_clip(frames, score, WAKE_MISS_DIR, WAKE_MISS_MAX_FILES,
                           "miss")


def save_wake_gated(frames, score):
    """Audio behind a score that cleared the threshold and failed the level gate.

    Kept in its own directory rather than with the hits, because `wake_hits`
    means "she answered" and these are the ones she did not. A real "hey nova"
    in here is the only way the gate's cost becomes visible, and it is the
    reason the gate is allowed to exist."""
    return _save_wake_clip(frames, score, WAKE_GATED_DIR, WAKE_GATED_MAX_FILES,
                           "gated")


def save_wake_hit(frames, score):
    """Audio behind a score that DID reach the threshold.

    The false wake case had no record at all. archive_recording keeps the
    command spoken after the wake, which is a different sound: it cannot tell
    you whether "hey nova" was ever said. Sep 6 2026 produced four wakes in a
    row whose commands read as television and overheard conversation, and there
    was no way to check what triggered any of them."""
    if not CAPTURE_WAKE_HITS:
        return None
    return _save_wake_clip(frames, score, WAKE_HIT_DIR, WAKE_HIT_MAX_FILES,
                           "hit")


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


def _wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def _whisper_cmd(wav_path, threads=None):
    """The one place the Whisper command line is built.

    Both the speculative run and the full one have to pass identical decoding
    settings or the speculation is transcribing under different rules than the
    fallback it stands in for, and the two would disagree for reasons nobody
    would think to look for. They differ only in thread count.

    The audio window follows the clip's own length, read from the file rather
    than passed in, so every caller gets it right without knowing it exists."""
    cmd = [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path,
           "-bs", str(WHISPER_BEAM_SIZE), "-bo", str(WHISPER_BEAM_SIZE),
           "--no-prints", "--no-timestamps",
           "-ac", str(audio_ctx_for(_wav_seconds(wav_path)))]
    if threads is not None:
        cmd += ["-t", str(threads)]
    if WHISPER_INITIAL_PROMPT:
        cmd += ["--prompt", WHISPER_INITIAL_PROMPT]
    return cmd


class _Speculation:
    """A Whisper run started before the recording finished.

    Holds a snapshot of the frames captured up to the moment speech stopped.
    The final clip differs only by the trailing silence that endpointing was
    still waiting out, and silence carries no words, so the transcript is the
    one the full clip would have produced.

    A Deepgram request goes out on the same frames, in parallel rather than
    instead: it is more accurate and finishes inside the same window, and whisper
    running anyway means the fallback costs nothing extra. Deepgram is a socket
    read, not a core, so the two do not contend."""

    def __init__(self, frames, turn_type='initial'):
        _write_wav(frames, SPECULATIVE_WAV)
        self.stale = False
        self.turn_type = turn_type
        self.turn = None          # the speculative Claude call, once started
        self._lock = threading.Lock()
        # Whisper is the fallback, started only when Deepgram fails or is late;
        # see _start_whisper. It used to start on every speculation, and on four
        # cores it slowed the ECAPA embedding beside it from 238ms to 617ms
        # median, measured Sep 28 2026, for a transcript Deepgram almost always
        # made unnecessary.
        self.proc = None
        # The voice embedding starts beside it, on the same frames; see
        # early_verify.py. A failure here only loses the head start. Started
        # before the Deepgram request, because the speculative Claude call is
        # decided on both the moment the transcript arrives.
        try:
            prepended = wake_word_audio() if turn_type == 'initial' else None
            self.early = early_verify.start(frames, prepended)
        except Exception as exc:
            print(f"Early verification unavailable: {exc}", flush=True)
            self.early = None
        # From the frames rather than SPECULATIVE_WAV, which the next speculation
        # overwrites while this request may still be reading it. Same hazard
        # early_verify.prepare exists to avoid.
        self.remote_run = deepgram_stt.start(frames, on_done=self._on_transcript)
        if self.remote_run is None:
            # No Deepgram to wait for: whisper is the transcript, as before it.
            self._start_whisper()
        else:
            threading.Timer(WHISPER_FALLBACK_AFTER_MS / 1000.0,
                            self._whisper_if_still_waiting).start()

    def _start_whisper(self):
        """Start the whisper run once, unless he has spoken again. Safe from any
        thread: the Deepgram hook, the late timer, and result() can all ask."""
        with self._lock:
            if self.proc is not None or self.stale:
                return
            self.proc = subprocess.Popen(
                _whisper_cmd(SPECULATIVE_WAV, threads=SPECULATIVE_THREADS),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def _whisper_if_still_waiting(self):
        """Deepgram has not answered in WHISPER_FALLBACK_AFTER_MS: start the
        fallback now, so a slow request costs at most this much over whisper
        alone rather than its full timeout."""
        if self.remote_run is not None and not self.remote_run.finished():
            self._start_whisper()

    def _on_transcript(self, text):
        """The Deepgram transcript is in: maybe open the Claude call now, while
        endpointing is still waiting out the silence. On the request's thread.

        None is Deepgram failing, which is the moment the whisper fallback
        starts. An empty string is an answer, heard no words, and needs none.

        The embedding usually finished long before Deepgram; if it has not, no
        speculation, rather than holding this thread for it."""
        if text is None:
            self._start_whisper()
            return
        if self.stale or self.early is None:
            return
        try:
            wav, embedding = self.early.result(timeout=0.3)
        except Exception:
            return
        turn = speculative_turn.consider(text, self.turn_type,
                                         len(wav) / RESEMBLYZER_SR, embedding, voiceprint)
        if turn is None:
            return
        # He may have spoken again while it was being decided. cancel() reads
        # self.turn under the same lock, so one of the two always cancels it.
        with self._lock:
            stale = self.stale
            if not stale:
                self.turn = turn
        if stale:
            turn.cancel()

    def cancel(self):
        """He spoke again, so this covers only part of the turn."""
        with self._lock:
            self.stale = True
            turn = self.turn
        if turn is not None:
            turn.cancel()
        if self.early is not None:
            self.early.cancel()
        if self.remote_run is not None:
            self.remote_run.cancel()
        with self._lock:
            proc = self.proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop_whisper(self):
        """Deepgram answered, so whisper's transcript will never be read.

        Left running, it holds three cores past the endpoint at beam 5, which is
        exactly when recall and local intent embed the transcript: recall
        measured 87ms on a quiet CPU against 209ms beside it, Sep 25 2026. Only
        the process stops. The early embedding is left alone, because
        verify_voice still reads it.

        Collected on a thread: a terminated child stays a zombie until someone
        waits on it, and waiting here would put its shutdown on the turn.
        Usually there is nothing to stop: whisper only runs when Deepgram was
        late or failed."""
        with self._lock:
            proc = self.proc
        if proc is None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        threading.Thread(target=proc.wait, daemon=True).start()

    def remote(self):
        """The Deepgram transcript, or None to fall back to whisper.

        **An empty string is an answer, not a failure**, and the two must not
        collapse. Deepgram returning no words on a false wake is correct and
        better than whisper's invented sentence; is_noise_transcript already
        reads it as noise and ends the turn quietly. Only None means ask
        whisper."""
        if self.remote_run is None:
            return None
        return self.remote_run.result()

    def result(self):
        """The transcript, or None if the run failed. Blocks if it is still
        going, which is the point: it started earlier, so it ends earlier.
        Only asked for when Deepgram gave nothing, so the fallback has usually
        started already; if the hook has not got there yet, it starts here."""
        self._start_whisper()
        if self.proc is None:
            return None
        try:
            out, _ = self.proc.communicate(timeout=30)
        except Exception:
            self.cancel()
            return None
        return out.strip() if self.proc.returncode == 0 else None


# Handed from the capture loops to transcribe. Module state rather than a return
# value because audio.py owns both ends and threading it through would change
# the signature every caller already uses.
_pending_speculation = None

# The early embedding of a speculation that held, handed from transcribe to
# verify_voice the same way. None whenever the speculation did not hold.
_early_verification = None


def cancel_speculation():
    """Drop any pending run. Safe to call when there is nothing pending.

    Also drops a speculative Claude call left from an earlier recording: every
    recording starts here, so nothing can be claimed across two of them."""
    global _pending_speculation, _early_verification
    speculative_turn.discard()
    _early_verification = None
    if _pending_speculation is not None:
        _pending_speculation.cancel()
        _pending_speculation = None


def maybe_speculate(frames, silent_chunks, spec_chunks, past_minimum,
                    turn_type='initial'):
    """Start Whisper on what is captured so far, mid pause.

    Called from both capture loops rather than inlined in each, because it was
    inlined in each and the follow up loop never got it. Phase 2 of
    listen_for_followup is a deliberate copy of record_command's loop, and a
    copy only stays correct until one side changes: follow ups paid the full
    transcription cost for as long as speculation had existed, and they are the
    majority of turns.

    Equality on silent_chunks rather than a threshold is what makes this fire
    once per pause. The caller cancels on resumed speech, which resets the
    count and lets the next pause try again."""
    global _pending_speculation
    if not (SPECULATIVE_TRANSCRIBE and past_minimum
            and _pending_speculation is None and silent_chunks == spec_chunks):
        return
    # A recording that will be transcribed in pieces cannot be stood in for by
    # one speculative run, so none is started for it.
    if len(frames) * 0.03 > WHISPER_SEGMENT_SECONDS:
        return
    try:
        _pending_speculation = _Speculation(list(frames), turn_type)
    except Exception as exc:
        print(f"Speculative transcribe unavailable: {exc}", flush=True)


def _write_wav(frames, path=TEMP_WAV):
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(_audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return path


def transcribe(wav_path):
    """Transcribe, collecting a speculative run if one is still valid.

    A speculation that survived to here started during the endpoint wait and is
    already partly or wholly done, so the stopwatch records only what was left.
    Anything that failed or went stale falls back to a full run, which is
    exactly today's behaviour."""
    global _pending_speculation, _early_verification
    speculation, _pending_speculation = _pending_speculation, None
    # Valid whenever the speculation is, whatever whisper made of it: the
    # frames are the turn's, less trailing silence, which trimming removes.
    _early_verification = (speculation.early if speculation is not None
                           and not speculation.stale else None)

    with timing.stopwatch('transcribe_ms'):
        if speculation is not None and not speculation.stale:
            remote = speculation.remote()
            if remote is not None:
                speculation.stop_whisper()
                print("(deepgram transcript)", flush=True)
                return remote
            text = speculation.result()
            if text:
                # Printed rather than marked: end_turn only reads known keys, so
                # an unrecognised stage is silently dropped. The hit rate is what
                # tunes SPECULATIVE_SILENCE_MS, so it needs to reach the journal.
                print("(speculative transcript hit)", flush=True)
                return text
        elif speculation is not None:
            speculation.cancel()

        # No speculation to ride along with: a cancelled one, a disabled one, or
        # a recording too long for a single window. Deepgram has no window limit,
        # so it is tried before the piecewise path rather than after it.
        remote = deepgram_stt.transcribe_file(wav_path)
        if remote is not None:
            print("(deepgram transcript)", flush=True)
            return remote

        if _wav_seconds(wav_path) > WHISPER_SEGMENT_SECONDS:
            return _transcribe_in_pieces(wav_path)

        result = subprocess.run(_whisper_cmd(wav_path),
                                capture_output=True, text=True)
    return result.stdout.strip()


def _transcribe_in_pieces(wav_path):
    """Past one Whisper window, transcribe piece by piece and join the text.

    Pieces are cut by segment_bounds at the quietest moment near each limit.
    Each is written beside the original, transcribed with the same settings as
    any other clip, and removed."""
    with wave.open(wav_path, "rb") as w:
        params = w.getparams()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    texts = []
    for i, (start, end) in enumerate(segment_bounds(samples, params.framerate,
                                                    WHISPER_SEGMENT_SECONDS)):
        piece = f"{wav_path}.part{i}.wav"
        with wave.open(piece, "wb") as out:
            out.setparams(params)
            out.writeframes(samples[start:end].tobytes())
        try:
            result = subprocess.run(_whisper_cmd(piece), capture_output=True, text=True)
            texts.append(result.stdout.strip())
        finally:
            os.remove(piece)
    print(f"(transcribed in {len(texts)} pieces)", flush=True)
    return " ".join(t for t in texts if t)


_MIXER_VALUE = re.compile(r'Capture (\d+) \[(\d+)%\](?: \[([-\d.]+)dB\])?')


def log_mic_gain():
    """Print the capture gain at startup and shout if it has drifted.

    ALSA settings can revert on a kernel or firmware update, or if alsactl
    restore runs against a stale state file. A silent revert does not fail
    loudly, it just quietly halves the data quality of everything recorded
    afterward, so it gets checked where it will be seen."""
    # None means no card matched MIC_NAME_HINT. Saying so is the honest
    # outcome: the previous behaviour was to fall back to card 0, which is the
    # speaker adapter here, and report its level as though it were the mic.
    if MIC_MIXER_CARD is None:
        print(f"Mic gain unchecked: no ALSA card matching {MIC_NAME_HINT!r} in "
              f"/proc/asound/cards.", flush=True)
        return None

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


# The outcomes and the two duration floors live in verification.py with the
# decision, and are imported at the top so callers keep using audio.RETRY.


# The last few things the barge in watcher heard, one per sentence, so a miss
# can be listened to. A long reply is two or three sentences, so four covers one.
BARGE_IN_WAV = os.path.expanduser("~/miles/build/barge_in_{}.wav")
BARGE_IN_KEEP = 4
_barge_in_count = 0


def watch_for_interrupt(stop_event):
    """Listen for the wake word while Nova is speaking. Returns True if heard.

    Runs on its own thread during playback, when the main loop is blocked on
    aplay and nothing else is reading the stream. Two readers on one PyAudio
    stream would interleave frames and produce intermittent deafness rather
    than an error, so the caller must guarantee exclusivity.

    No echo cancellation, and none is needed. Her own voice reaches the
    microphone but the wake model only fires on one phrase, which her speech
    does not contain. That is the whole reason this is cheap when general barge
    in is not.

    A fresh model instance rather than the shared one: the main loop's
    detection state must not be disturbed by frames captured mid sentence."""
    try:
        with silence_stderr():
            watcher = Model(wakeword_model_paths=[WAKE_MODEL_PATH])
    except Exception as e:
        print(f"  barge in: wake model did not load: {e!r}", flush=True)
        return False

    # Logged per sentence, because a watcher that quietly scores low, or quietly
    # dies on an exception, looks exactly like one that heard nothing. The first
    # live test on Sep 30 2026 failed with no way to tell which.
    heard = False
    peak, frames = 0.0, 0
    kept = []
    while not stop_event.is_set():
        try:
            # The recognition channel, not the one the rest of the pipeline
            # hears: see BARGE_IN_CHANNEL.
            data = stream.read(CHUNK, exception_on_overflow=False,
                               channel=BARGE_IN_CHANNEL)
        except Exception as e:
            print(f"  barge in: mic read failed: {e!r}", flush=True)
            break
        frames += 1
        kept.append(data)
        frame = np.frombuffer(data, dtype=np.int16)
        try:
            for _, score in watcher.predict(frame).items():
                peak = max(peak, score)
                if score > BARGE_IN_THRESHOLD:
                    heard = True
                    stop_event.set()
                    break
        except Exception as e:
            print(f"  barge in: wake model failed: {e!r}", flush=True)
            break
    global _barge_in_count
    _barge_in_count += 1
    path = BARGE_IN_WAV.format(_barge_in_count % BARGE_IN_KEEP)
    level = None
    if kept:
        audio_bytes = b"".join(kept)
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
        level = 20 * np.log10(np.abs(samples).max() / 32768 + 1e-9)
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)
        except OSError:
            pass
    level_text = f", loudest {level:.1f} dBFS" if level is not None else ""
    print(f"  barge in: {'HEARD' if heard else 'watched'} {frames * CHUNK / RATE:.1f}s, "
          f"peak {peak:.2f}{level_text} ({os.path.basename(path)})", flush=True)
    return heard


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
    from config import (VERIFY_THRESHOLD,
                        VOICEPRINT_LEARN_MIN_SIMILARITY,
                        VOICEPRINT_LEARN_MIN_SECONDS)

    global _early_verification
    verify_started = time.monotonic()

    # Started during the endpoint wait, if the speculation held. Its wav and
    # embedding are the ones this function would compute, so the rest of the
    # decision is unchanged.
    early, _early_verification = _early_verification, None
    wav = embedding = None
    if early is not None:
        try:
            wav, embedding = early.result(timeout=10)
            print("(early verification hit)", flush=True)
        except Exception as exc:
            print(f"Early verification failed, verifying now: {exc}", flush=True)
            wav = embedding = None

    if wav is None:
        # Prepend the wake word for initial turns. A follow up has no wake
        # word, and the buffer would be stale from the turn before, so this is
        # scoped to the only case where the audio genuinely belongs to this
        # utterance.
        prepended = wake_word_audio() if turn_type == 'initial' else None
        with wave.open(wav_path, 'rb') as wf:
            command = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        wav = early_verify.prepare(command, prepended)

    with wave.open(wav_path, 'rb') as wf:
        duration_seconds = wf.getnframes() / wf.getframerate()
    embedded_duration_seconds = len(wav) / RESEMBLYZER_SR

    rms_dbfs, snr_db, spectral_tilt = measure_acoustics(wav_path)

    # The decision itself is verification.py's, so the speculative turn asks the
    # same question of the early embedding and gets the same answer.
    unscored = verification.before_embedding(embedded_duration_seconds, session_trusted)
    if unscored is not None and unscored.outcome == NO_AUDIO:
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
    # longer utterance.
    #
    # The exposure this accepts is someone speaking into the mic within the
    # follow up window, in the same room, immediately after Lethanial. Against
    # the current action set (weather, timers, reminders, conversation) that is
    # a narrower risk than refusing him every time he says "thanks". Revisit
    # when an action with real consequence is added.
    if unscored is not None:
        print(f"Too short to verify ({embedded_duration_seconds:.2f}s), "
              f"trusting session.", flush=True)
        log_verification(
            similarity=0.0, accepted=True, threshold_used=VERIFY_THRESHOLD,
            transcript=transcript, duration_seconds=duration_seconds,
            embedded_duration_seconds=embedded_duration_seconds,
            turn_type=turn_type, wake_confidence=wake_confidence,
            outcome=unscored.logged_as, rms_dbfs=rms_dbfs, snr_db=snr_db,
            spectral_tilt=spectral_tilt, recording_path=recording_path,
        )
        timing.mark('verify_ms', (time.monotonic() - verify_started) * 1000.0)
        return VERIFIED

    if embedding is None:
        embedding = embed_voice(wav)
    verdict = verification.on_embedding(embedding, voiceprint)
    if verdict.logged_as is None:
        print("Encoder declined the clip as too short to embed.", flush=True)
        return NO_AUDIO
    similarity = verdict.similarity
    # Covers preprocess_wav, the acoustic measures, and embed_utterance: all
    # the work that sits between transcription and the Claude call.
    timing.mark('verify_ms', (time.monotonic() - verify_started) * 1000.0)

    print(f"Voice similarity: {similarity:.3f}", flush=True)
    accepted  = verdict.outcome == VERIFIED
    ambiguous = verdict.outcome == RETRY

    log_verification(
        similarity=float(similarity),
        accepted=accepted,
        threshold_used=VERIFY_THRESHOLD,
        transcript=transcript,
        duration_seconds=duration_seconds,
        embedded_duration_seconds=embedded_duration_seconds,
        turn_type=turn_type,
        wake_confidence=wake_confidence,
        outcome=verdict.logged_as,
        rms_dbfs=rms_dbfs,
        snr_db=snr_db,
        spectral_tilt=spectral_tilt,
        recording_path=recording_path,
    )

    # Keep the embedding when this turn is well clear of the accept bar and
    # long enough to have embedded stably. Both conditions matter and neither
    # is the accept threshold: a sample that merely passed is not evidence
    # strong enough to change what "he sounds like" means, and the profile is
    # not correctable once a wrong one is averaged in.
    if (accepted
            and similarity >= VOICEPRINT_LEARN_MIN_SIMILARITY
            and embedded_duration_seconds >= VOICEPRINT_LEARN_MIN_SECONDS):
        # Tagged with whichever encoder is live, never a literal. The column
        # exists so embeddings from two encoders are never averaged into one
        # centroid, and a hardcoded name would defeat it on the day it matters.
        keep_voiceprint_sample(embedding, similarity, embedded_duration_seconds,
                               model=SPEAKER_ENCODER, mic=MIC_NAME)

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
