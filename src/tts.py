import os
import queue
import re
import subprocess
import threading
import time
from elevenlabs.client import ElevenLabs

import timing

from database import get_pronunciations
from config import (
    ELEVENLABS_API_KEY, TTS_VOICE_ID,
    DEFAULT_TTS_MODEL, TTS_OUTPUT_FORMAT, TTS_APP_OUTPUT_FORMAT, TTS_PHONEME_TAGS,
    TTS_VOICE_SETTINGS, SPEAKER_NAME_HINT, speak_lock,
)

CHIME_PATH = os.path.expanduser("~/miles/assets/wake_chime.wav")

_BRACKET_CUE   = re.compile(r'\[.*?\]')
# The dotted form has no trailing \b on purpose. With one, "M.I.L.E.S. is
# online" matched only "M.I.L.E.S" and left the final period behind, producing
# "Miles. is online" and a full stop read aloud in the middle of the sentence.
# There is no word boundary between "." and " ", so the boundary anchor could
# never sit where it was assumed to.
_MILES_ACRONYM = re.compile(r'\bM\.I\.L\.E\.S\.?|\bMILES\b')


def normalize_pronunciation(text):
    """Replace graphemes with the respellings the synthesizer says correctly.

    Voice channel only. This must never touch text the app displays: an alias
    is a phonetic hack, so "Luhthanyul" on screen is simply a misspelling of
    Lethanial's name. The caller enforces that; this function does the work.

    Whole word only, case insensitive, longest grapheme first.

    Whole word matters because a grapheme inside a longer word is a different
    word. Longest first matters because a shorter entry that is a prefix of a
    longer one would otherwise consume it and leave the remainder unreplaced.
    The database returns rows in that order already.

    Case is matched insensitively but the alias is substituted verbatim: the
    synthesizer is reading sound, not spelling, so preserving the original
    capitalization would mean nothing to it.

    A failure here returns the text unchanged rather than raising. Aliases are
    user data added at runtime, and mispronouncing a word is a far smaller
    problem than a bad row taking down every turn."""
    try:
        rows = get_pronunciations()
    except Exception as e:
        print(f"Pronunciation lookup failed, speaking as written: {e}", flush=True)
        return text

    for row in rows:
        grapheme, alias = row[0], row[1]
        arpabet = row[2] if len(row) > 2 else None

        # Phonemes are exact; an alias is a respelling tuned by ear. Prefer the
        # phonemes when the model can honor them and the row has them.
        replacement = alias
        if TTS_PHONEME_TAGS and arpabet:
            replacement = (f'<phoneme alphabet="cmu-arpabet" ph="{arpabet}">'
                           f'{grapheme}</phoneme>')

        # The replacement is a lambda, not the alias string, because re.sub
        # interprets backslash escapes in a replacement template. An alias
        # containing \1 raised "invalid group reference" and one containing \n
        # would have silently inserted a newline. Aliases are phonetic
        # respellings typed by hand, so they must be substituted literally.
        text = re.sub(rf'\b{re.escape(grapheme)}\b',
                      lambda _, r=replacement: r, text, flags=re.IGNORECASE)
    return text

_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def _resolve_speaker_device(name_hint, fallback="plughw:0,0"):
    # Card numbers shift between boots, so find the card by name instead of
    # trusting a fixed index. /proc/asound/cards is ALSA's own device list,
    # read only, no exclusive lock involved (unlike opening the device itself).
    try:
        with open("/proc/asound/cards") as f:
            cards = f.read()
    except OSError:
        print(f"Could not read /proc/asound/cards, falling back to {fallback}", flush=True)
        return fallback

    # Each card is two lines: " N [ID  ]: driver - description" then an
    # indented detail line. Split on the start of each card entry and search
    # both lines of a block for the name hint.
    for block in re.split(r'\n(?=\s*\d+\s+\[)', cards):
        match = re.match(r'\s*(\d+)\s+\[', block)
        if match and name_hint in block:
            card_num = int(match.group(1))
            device = f"plughw:{card_num},0"
            print(f"Found speaker: {name_hint} ({device})", flush=True)
            return device

    print(f"Speaker '{name_hint}' not found, falling back to {fallback}", flush=True)
    return fallback


SPEAKER_DEVICE = _resolve_speaker_device(SPEAKER_NAME_HINT)


def play_chime():
    # Fire-and-forget: no speak_lock so it never blocks timer/reminder threads.
    # record_command() starts immediately while the chime plays in the background.
    subprocess.Popen(
        ["aplay", "-D", SPEAKER_DEVICE, CHIME_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _prepare(text):
    """The text exactly as the synthesizer should receive it, or None when
    nothing is left to say."""
    clean = _BRACKET_CUE.sub('', text).strip()
    clean = _MILES_ACRONYM.sub('Miles', clean)
    if not clean:
        return None
    if not clean.endswith(('?', '!', '.')):
        clean += '.'

    # Last thing before the API call, so nothing downstream can undo it and
    # nothing upstream ever sees an alias. What gets returned, saved to
    # history, and shown in the app is the real spelling.
    return normalize_pronunciation(clean)


class Synthesis:
    """One utterance being synthesized now, to be played later.

    Splitting synthesis from playback is what lets the next sentence be fetched
    while the current one plays. speak() used to do both in order, so a
    sentence was not even requested until the one before it had finished.

    The stream is drained on a daemon thread into a queue. A failure is kept
    and reported by the player, so it surfaces where it always did rather than
    dying quietly on a background thread."""

    _DONE = object()

    def __init__(self, text, voice_settings=None, model=None, seed=None,
                 output_format=None):
        self.text = text
        self.error = None
        self.requested_at = time.monotonic()
        self._chunks = queue.Queue()
        self._request = dict(
            voice_id=TTS_VOICE_ID,
            text=text,
            model_id=model or DEFAULT_TTS_MODEL,
            voice_settings=voice_settings or TTS_VOICE_SETTINGS,
            # Defaults to the room speaker's raw PCM. Only a caller that is not
            # feeding aplay passes anything else, and play() cannot read one
            # that does, because the format it opens aplay with is fixed.
            output_format=output_format or TTS_OUTPUT_FORMAT,
            **({"seed": seed} if seed is not None else {}),
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self):
        try:
            for chunk in _elevenlabs.text_to_speech.stream(**self._request):
                if chunk:
                    self._chunks.put(chunk)
        except Exception as e:
            self.error = e
        finally:
            self._chunks.put(self._DONE)

    def chunks(self):
        """Yield audio as it arrives, until the stream ends or fails."""
        while True:
            chunk = self._chunks.get()
            if chunk is self._DONE:
                return
            yield chunk


def join_for_speech(sentences):
    """Several sentences as one request, with a breath between them.

    Chosen by ear, Sep 13 2026. On eleven_v3 at stability 1.0 every sentence
    came out in the same tone with almost no pause. A full stop written as an
    ellipsis made v3 pause 550ms between sentences against 210ms without, and
    nothing else about the delivery changed. A question or an exclamation keeps
    its own mark, because "?..." reads as uncertainty rather than a pause.

    Only what is sent to ElevenLabs is shaped. History and the app keep the
    punctuation Nova actually wrote."""
    parts = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if i < len(sentences) - 1 and sentence.endswith(".") and not sentence.endswith("..."):
            sentence = sentence[:-1] + "..."
        parts.append(sentence)
    return " ".join(parts)


def start_synthesis(text, voice_settings=None, model=None, seed=None,
                    output_format=None):
    """Begin synthesizing now. None when there is nothing to say."""
    clean = _prepare(text)
    if clean is None:
        return None
    return Synthesis(clean, voice_settings, model, seed, output_format)


def stream_audio(text, output_format=None):
    """Synthesize for somewhere that is not the room speaker, and yield the
    audio as it arrives. None when there is nothing to say.

    This is the app's path. It shares everything that decides how Nova sounds,
    _prepare and so the pronunciation table, the voice, the model and the voice
    settings, and shares none of the playback: no speak_lock, no aplay, no
    timing.note_tts. A phone asking for audio must never be able to block the
    room speaker mid sentence, and the lock is what would let it.

    One request per call, never one per sentence. On eleven_v3 each request is
    voiced on its own, so a reply synthesized a sentence at a time comes back in
    several slightly different deliveries, heard as a different person partway
    through. _tts_consumer solves the same problem the same way.

    A failure before any audio arrives is raised, so the caller can still answer
    with an error rather than a truncated file. A failure partway through is
    logged and ends the stream, because by then the caller has already committed
    to sending audio."""
    synthesis = start_synthesis(text, output_format=output_format or TTS_APP_OUTPUT_FORMAT)
    if synthesis is None:
        return None

    def chunks():
        sent = False
        for chunk in synthesis.chunks():
            sent = True
            yield chunk
        if synthesis.error is not None:
            print(f"TTS error (ElevenLabs, app): {synthesis.error}", flush=True)
            if not sent:
                raise synthesis.error

    return chunks()


def _open_aplay():
    return subprocess.Popen(
        [
            "aplay", "-D", SPEAKER_DEVICE,
            "-f", "S16_LE", "-r", "22050", "-c", "1",
            "--buffer-size=8192", "--period-size=1024",
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def play(synthesis, interrupt=None):
    """Play a started synthesis to the end.

    Returns how long it waited before its first audio reached aplay, in ms, or
    None if nothing played. That wait is the silence heard before this
    utterance. For the first sentence of a turn it is roughly the time to first
    byte; for a later one it is only whatever of its synthesis was unfinished
    when the sentence before it ended, which is the number fetching ahead
    exists to drive toward zero.

    aplay is opened on the first chunk rather than up front, so a request that
    fails outright never starts the audio device, the same as before synthesis
    and playback were split."""
    if synthesis is None:
        return None

    with speak_lock:
        started = time.monotonic()
        aplay, waited_ms = None, None
        try:
            for chunk in synthesis.chunks():
                if aplay is None:
                    ttfb_ms = (time.monotonic() - synthesis.requested_at) * 1000.0
                    aplay = _open_aplay()
                # Flush after every write. Without it Python buffers up to 64KB
                # and adds about 1.5s of phantom latency; see VOICE_OUTPUT.md.
                aplay.stdin.write(chunk)
                aplay.stdin.flush()
                if waited_ms is None:
                    waited_ms = (time.monotonic() - started) * 1000.0
                    timing.note_tts(ttfb_ms,
                                    (time.monotonic() - synthesis.requested_at) * 1000.0)
        except Exception as e:
            print(f"TTS playback error: {e}", flush=True)
        finally:
            if aplay is not None:
                aplay.stdin.close()
                # Barge in. The watcher thread reads the microphone while aplay
                # drains; if it hears the wake word it sets the event, and
                # killing aplay here is what actually cuts her off, because the
                # ALSA buffer holds roughly 185ms of audio already written.
                if interrupt is not None and interrupt.is_set():
                    aplay.kill()
                aplay.wait()

    if synthesis.error is not None:
        print(f"TTS error (ElevenLabs): {synthesis.error}", flush=True)
    return waited_ms


def speak(text, voice_settings=None, model=None, seed=None,
          interrupt=None):
    """Synthesize and play one utterance.

    seed pins the generation. Without it ElevenLabs produces a different
    rendition every call, and the spread between two renditions of identical
    input is wide enough that the same phoneme string can sound right once and
    wrong the next time. Production leaves it None, because varied delivery is
    desirable in conversation. Comparisons must set it, or they are measuring
    luck rather than the thing being compared."""
    play(start_synthesis(text, voice_settings, model, seed), interrupt)
