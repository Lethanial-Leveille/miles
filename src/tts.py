import os
import re
import subprocess
import time
from elevenlabs.client import ElevenLabs

import timing

from config import (
    ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID,
    DEFAULT_TTS_MODEL, TTS_OUTPUT_FORMAT,
    EMMA_NEUTRAL, SPEAKER_NAME_HINT, speak_lock,
)

CHIME_PATH = os.path.expanduser("~/miles/assets/wake_chime.wav")

_BRACKET_CUE   = re.compile(r'\[.*?\]')
_MILES_ACRONYM = re.compile(r'\b(?:M\.I\.L\.E\.S\.?|MILES)\b')

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


def speak(text, voice_settings=None, model=None):
    with speak_lock:
        clean = _BRACKET_CUE.sub('', text).strip()
        clean = _MILES_ACRONYM.sub('Miles', clean)
        if not clean:
            return
        if not clean.endswith(('?', '!', '.')):
            clean += '.'

        settings  = voice_settings or EMMA_NEUTRAL
        tts_model = model or DEFAULT_TTS_MODEL

        requested_at = time.monotonic()
        try:
            audio_iter = _elevenlabs.text_to_speech.stream(
                voice_id=ELEVENLABS_VOICE_ID,
                text=clean,
                model_id=tts_model,
                voice_settings=settings,
                output_format=TTS_OUTPUT_FORMAT,
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
            aplay.wait()
