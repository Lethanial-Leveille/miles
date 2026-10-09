"""Transcribe through Deepgram, with whisper as the floor underneath.

Measured Sep 25 2026 on the 58 hand labelled clips: 7.6% WER against 17.5% for
base.en q8_0 at beam 5, 8.8% against 11.8% word weighted on real speech, follow
ups 10.7% against 33.3%, and it invents words in 2 of 20 non speech clips against
5. Faster too: p50 520ms and p90 826ms both fit inside the 990ms speculation head
start, where whisper at beam 5 costs 1137ms and does not. The tables are in
docs/AUDIO_PIPELINE.md.

**This never becomes the only path.** Every failure returns None and the caller
falls back to whisper, which is what the local model is for and why it stays. A
remote engine that is down, slow, or answering strangely must cost nothing more
than today's behaviour.

**It sends the room to a third party, which was decided rather than overlooked.**
Every recording leaves the Pi, including the 45 percent of wakes that are false,
so other people in the apartment are transcribed by Deepgram too. The decision
and the alternative that was declined are in docs/SESSION_START.md. Do not
quietly widen what is sent; do not send anything the wake loop did not already
record.

Runs on a thread rather than in the caller, so it can start beside the whisper
speculation on the same frames. Nothing here holds the microphone or the GIL for
longer than a socket read.
"""

import io
import threading
import time
import wave

import requests

from config import (DEEPGRAM_STT, DEEPGRAM_API_KEY, DEEPGRAM_MODEL,
                    DEEPGRAM_TIMEOUT_S, DEEPGRAM_KEYTERMS, RATE, CHANNELS)

_ENDPOINT = "https://api.deepgram.com/v1/listen"

# Free and carries no audio. Only here to open or refresh the connection.
_WARM_URL = "https://api.deepgram.com/v1/projects"

# A new connection costs about 225ms of every request (380ms against 153ms
# reused, measured Sep 25 2026), and Deepgram closes one that sits idle for
# about 5s: a reuse at 5s failed with RemoteDisconnected. So a connection is
# reused only inside 4s of its last response, and kept warm while he talks by
# a request every 3s.
_REUSE_WITHIN_S = 4.0
_REWARM_AFTER_S = 3.0

_lock = threading.Lock()
_session = None
_last_answer = 0.0     # monotonic time the connection last finished a response
_warming = False

# punctuate, and deliberately not smart_format. Smart format rewrites numbers as
# digits, and Nova's spoken replies and the eval truths are both written in words,
# so "1 minute" would be a difference of formatting masquerading as one of
# hearing. The eval scored it this way, so the live path has to match or the
# measurement stops describing what runs.
_PARAMS = {"punctuate": "true", "language": "en", "keyterm": DEEPGRAM_KEYTERMS}


def available():
    """Whether a remote transcript is even worth attempting."""
    return bool(DEEPGRAM_STT and DEEPGRAM_API_KEY)


def _wav_bytes(frames):
    """Raw capture frames as a WAV in memory.

    In memory because the speculative WAV on disk is overwritten by the next
    speculation while this request may still be reading it, which is the same
    hazard early_verify.prepare was written to avoid."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def _session_for(fresh=False):
    """The shared connection when it is recent enough to trust, else a new one.

    Past _REUSE_WITHIN_S the old one is closed rather than tried, because the
    server has likely dropped it and finding that out costs a failed request."""
    global _session
    with _lock:
        stale = time.monotonic() - _last_answer >= _REUSE_WITHIN_S
        if _session is None or fresh or stale:
            if _session is not None:
                _session.close()
            _session = requests.Session()
        return _session


def _answered():
    global _last_answer
    with _lock:
        _last_answer = time.monotonic()


def _headers(content_type=None):
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def keep_warm():
    """Open or refresh the connection while he is still talking.

    Called on speech frames by both capture loops and rate limited here, so it
    costs a thread start at most every _REWARM_AFTER_S. Sends no audio: the
    recording still leaves the Pi only at the pause, as before."""
    global _warming
    if not available():
        return
    with _lock:
        if _warming or time.monotonic() - _last_answer < _REWARM_AFTER_S:
            return
        _warming = True
    threading.Thread(target=_warm, daemon=True).start()


def _warm():
    global _warming
    try:
        _session_for().get(_WARM_URL, headers=_headers(), timeout=DEEPGRAM_TIMEOUT_S)
        _answered()
    except requests.RequestException:
        pass  # the transcription request opens its own connection, as it always did
    finally:
        with _lock:
            _warming = False


def _post(audio, fresh):
    return _session_for(fresh).post(
        _ENDPOINT,
        params=dict(_PARAMS, model=DEEPGRAM_MODEL),
        headers=_headers("audio/wav"),
        data=audio, timeout=DEEPGRAM_TIMEOUT_S)


def transcribe_bytes(audio):
    """One request. The transcript, or None on any failure whatsoever.

    None and the empty string mean different things here. None is "ask whisper";
    an empty string is Deepgram saying it heard no words, which on a false wake
    is the correct answer and better than whisper's invented sentence.

    A reused connection that turns out closed is retried once on a new one. A
    timeout is not, so a slow Deepgram can never double the wait."""
    if not available():
        return None
    try:
        try:
            r = _post(audio, fresh=False)
        except requests.Timeout:
            raise
        except requests.ConnectionError:
            r = _post(audio, fresh=True)
        r.raise_for_status()
        _answered()
        alts = r.json()["results"]["channels"][0]["alternatives"]
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"Deepgram unavailable, using whisper: {exc}", flush=True)
        return None
    return alts[0].get("transcript", "") if alts else ""


class Run:
    """A request in flight on its own thread."""

    def __init__(self, audio, on_done=None):
        self._text = None
        self._done = threading.Event()
        self._cancelled = False
        self._on_done = on_done
        self._thread = threading.Thread(target=self._work, args=(audio,),
                                        daemon=True)
        self._thread.start()

    def _work(self, audio):
        try:
            self._text = transcribe_bytes(audio)
        finally:
            self._done.set()
        # For the speculative turn, which starts the moment the transcript
        # exists rather than when someone next asks for it. Not called for a
        # run already cancelled, and a failure in it never reaches the result.
        if self._on_done is not None and not self._cancelled:
            try:
                self._on_done(self._text)
            except Exception as exc:
                print(f"Deepgram completion hook failed: {exc}", flush=True)

    def cancel(self):
        """Stop waiting on it. The request itself is left to finish and be
        discarded, because requests has no safe interrupt and abandoning the
        socket is cheaper than the complexity of one."""
        self._cancelled = True

    def finished(self):
        """Whether the request has come back, with an answer or a failure."""
        return self._done.is_set()

    def result(self, timeout=None):
        """The transcript, or None if it failed, was cancelled, or is still out.

        The timeout is a second line of defence behind the request's own, for the
        case where the thread is wedged somewhere requests did not time out."""
        if self._cancelled:
            return None
        if not self._done.wait(
                timeout if timeout is not None else DEEPGRAM_TIMEOUT_S):
            print("Deepgram did not answer in time, using whisper.", flush=True)
            return None
        return self._text


def transcribe_file(path):
    """One request for a WAV already on disk, for the path with no speculation
    to ride along with. The transcript, or None to fall back to whisper."""
    if not available():
        return None
    try:
        with open(path, "rb") as f:
            audio = f.read()
    except OSError as exc:
        print(f"Deepgram could not read {path}: {exc}", flush=True)
        return None
    return transcribe_bytes(audio)


def start(frames, on_done=None):
    """Begin a request on captured frames. None when it is not worth trying.
    on_done, when given, is called with the transcript (or None) on the
    request's own thread as soon as it arrives."""
    if not available():
        return None
    try:
        return Run(_wav_bytes(frames), on_done)
    except Exception as exc:
        print(f"Deepgram could not start: {exc}", flush=True)
        return None
