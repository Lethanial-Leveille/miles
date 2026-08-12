import os
import re
import subprocess
import time
from elevenlabs.client import ElevenLabs

import timing

from database import get_pronunciations
from config import (
    ELEVENLABS_API_KEY, TTS_VOICE_ID,
    DEFAULT_TTS_MODEL, TTS_OUTPUT_FORMAT, TTS_PHONEME_TAGS,
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


def speak(text, voice_settings=None, model=None, seed=None,
          interrupt=None):
    """Synthesize and play one utterance.

    seed pins the generation. Without it ElevenLabs produces a different
    rendition every call, and the spread between two renditions of identical
    input is wide enough that the same phoneme string can sound right once and
    wrong the next time. Production leaves it None, because varied delivery is
    desirable in conversation. Comparisons must set it, or they are measuring
    luck rather than the thing being compared."""
    with speak_lock:
        clean = _BRACKET_CUE.sub('', text).strip()
        clean = _MILES_ACRONYM.sub('Miles', clean)
        if not clean:
            return
        if not clean.endswith(('?', '!', '.')):
            clean += '.'

        # Last thing before the API call, so nothing downstream can undo it and
        # nothing upstream ever sees an alias. What gets returned, saved to
        # history, and shown in the app is the real spelling.
        clean = normalize_pronunciation(clean)

        settings  = voice_settings or TTS_VOICE_SETTINGS
        tts_model = model or DEFAULT_TTS_MODEL

        requested_at = time.monotonic()
        try:
            audio_iter = _elevenlabs.text_to_speech.stream(
                voice_id=TTS_VOICE_ID,
                text=clean,
                model_id=tts_model,
                voice_settings=settings,
                output_format=TTS_OUTPUT_FORMAT,
                **({"seed": seed} if seed is not None else {}),
            )
        except Exception as e:
            print(f"TTS error (ElevenLabs): {e}", flush=True)
            return

        aplay = subprocess.Popen(
            [
                "aplay", "-D", SPEAKER_DEVICE,
                "-f", "S16_LE", "-r", "22050", "-c", "1",
                "--buffer-size=8192", "--period-size=1024",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        # Split so network time and local write time are separable: ttfb is
        # ElevenLabs, the gap between the two is ours.
        first_chunk_done = False
        try:
            for chunk in audio_iter:
                if chunk:
                    if not first_chunk_done:
                        ttfb_ms = (time.monotonic() - requested_at) * 1000.0
                        aplay.stdin.write(chunk)
                        aplay.stdin.flush()
                        timing.note_tts(
                            ttfb_ms,
                            (time.monotonic() - requested_at) * 1000.0,
                        )
                        first_chunk_done = True
                        continue
                    aplay.stdin.write(chunk)
                    aplay.stdin.flush()
        except Exception as e:
            print(f"TTS playback error: {e}", flush=True)
        finally:
            aplay.stdin.close()

            # Barge in. The watcher thread reads the microphone while aplay
            # drains; if it hears the wake word it sets the event, and killing
            # aplay here is what actually cuts her off, because the ALSA buffer
            # holds roughly 185ms of audio that has already been written.
            if interrupt is not None and interrupt.is_set():
                aplay.kill()
            aplay.wait()
