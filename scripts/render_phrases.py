#!/usr/bin/env python3
"""Render the phrase bank to disk, so Nova can still speak with no network.

    python3 scripts/render_phrases.py list             # what exists, with durations
    python3 scripts/render_phrases.py render           # fill in anything missing
    python3 scripts/render_phrases.py render --force   # re-render everything
    python3 scripts/render_phrases.py render ack       # one key only
    python3 scripts/render_phrases.py trim             # re-trim on disk, no API call
    python3 scripts/render_phrases.py preview ack      # hear every ack, in order
    python3 scripts/render_phrases.py audition dismiss 0    # hear 6 seeds of one phrase
    python3 scripts/render_phrases.py pick dismiss 0 3      # commit the one that read best

Run it while online. It costs one ElevenLabs call per variant and then never
costs anything again.

Durations are printed because they are the thing that decides whether a wake
acknowledgement is usable. The chime is 0.320s and overlaps recording; a spoken
ack has to finish before the mic opens, so its duration lands directly between
the wake word and the command.
"""

import os
import subprocess
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from elevenlabs.client import ElevenLabs                 # noqa: E402

import phrasebank                                        # noqa: E402
import tts                                               # noqa: E402
from database import init_db                             # noqa: E402
from config import (ELEVENLABS_API_KEY, TTS_VOICE_ID,    # noqa: E402
                    DEFAULT_TTS_MODEL, TTS_VOICE_SETTINGS,
                    TTS_OUTPUT_FORMAT, PHRASE_DIR)

CHIME_SECONDS = 0.320   # what an ack is competing with

# pcm_22050 means 22050Hz mono signed 16 bit little endian. Parsed rather than
# hardcoded so that changing TTS_OUTPUT_FORMAT cannot quietly produce WAV
# headers that disagree with their own samples.
SAMPLE_RATE = int(TTS_OUTPUT_FORMAT.rsplit("_", 1)[1])

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


def duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


# Kept rather than trimmed flush. Cutting a plosive or a fricative tail at the
# sample it drops under the threshold sounds clipped, and the leading pad keeps
# a soft onset from starting mid consonant.
LEAD_PAD_MS  = 30
TRAIL_PAD_MS = 60


def trim_silence(pcm):
    """Strip the dead air ElevenLabs pads onto each clip.

    It measured up to 0.232s on a 0.557s acknowledgement, which is not a
    cosmetic problem: an ack has to finish before the mic opens, so trailing
    silence is time spent waiting to be allowed to speak. "Yeah?" was 0.308s of
    speech inside a 0.557s file.

    Amplitude rather than webrtcvad on purpose. This is studio audio with a true
    digital silence floor, not a room, so there is nothing here for a VAD to be
    better at."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return pcm

    threshold = max(200, abs(samples).max() * 0.02)
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return pcm

    lead  = int(SAMPLE_RATE * LEAD_PAD_MS / 1000)
    trail = int(SAMPLE_RATE * TRAIL_PAD_MS / 1000)
    start = max(0, loud[0] - lead)
    end   = min(samples.size, loud[-1] + trail)
    return samples[start:end].tobytes()


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def _synthesize(text, seed):
    # Same normalization speak() applies immediately before its own call, so a
    # rendered phrase and a spoken one pronounce names identically.
    chunks = client.text_to_speech.stream(
        voice_id=TTS_VOICE_ID,
        text=tts.normalize_pronunciation(text),
        model_id=DEFAULT_TTS_MODEL,
        voice_settings=TTS_VOICE_SETTINGS,
        output_format=TTS_OUTPUT_FORMAT,
        seed=seed,
    )
    return b"".join(c for c in chunks if c)


def render_one(key, index, text):
    """Synthesize one variant and write it as a WAV.

    The seed is pinned to the index. Production leaves seed None because varied
    delivery is what makes conversation sound alive, but this is a stored
    artifact rather than a conversation: pinning means re-rendering after a
    phrase edit changes only the phrase that was edited, instead of quietly
    redrawing every other reading of the same voice."""
    pcm  = _synthesize(text, seed=index)
    path = phrasebank.path_for(key, index)
    write_wav(path, trim_silence(pcm))
    return path


def cmd_preview(key=None):
    """Play what is actually on disk, in index order, announcing each number.

    Reading a phrase in a source file tells you nothing about how it lands. This
    is for listening through a set and deciding which ones to keep, which is the
    only way to judge an acknowledgement: it is 300ms of tone and delivery, and
    no amount of looking at "Uh huh?" predicts whether it reads as a question."""
    keys = [key] if key else list(phrasebank.PHRASES)
    for k in keys:
        variants = phrasebank.PHRASES.get(k)
        if not variants:
            print(f"no such key: {k}")
            return
        print(f"\n{k}")
        for index, text in enumerate(variants):
            path = phrasebank.path_for(k, index)
            if not os.path.exists(path):
                print(f"  {index}      --   {text!r}  NOT RENDERED")
                continue
            print(f"  {index}  {duration(path):5.3f}s  {text!r}", flush=True)
            subprocess.run(["aplay", "-D", tts.SPEAKER_DEVICE, path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
    print(f"\n  drop one:   edit PHRASES in src/phrasebank.py")
    print(f"  reroll one: render_phrases.py audition {keys[0]} <index>  then  pick {keys[0]} <index> <seed>")


AUDITION_DIR = os.path.join(PHRASE_DIR, ".audition")


def cmd_audition(key, index, seeds=6):
    """Render one phrase at several seeds and play each, so a bad draw can be
    heard and rejected before it is committed.

    This exists because pinning the seed cuts both ways. Reproducibility is
    worth having, but a pinned seed also freezes whatever rendition it drew,
    and his name is documented as the unstable part: at stability 0.80 the band
    between 0.75 and 0.90 trades holding the name against reading flat. Live,
    a bad reading of it passes in one turn. Rendered, it is permanent.

    Auditioning turns that around. Pick once by ear and it is right every time
    after, which live synthesis never guarantees."""
    text = phrasebank.PHRASES[key][index]
    os.makedirs(AUDITION_DIR, exist_ok=True)
    print(f"{key}.{index}  {text!r}\n")

    for seed in range(seeds):
        path = os.path.join(AUDITION_DIR, f"{key}.{index}.seed{seed}.wav")
        if not os.path.exists(path):
            write_wav(path, trim_silence(_synthesize(text, seed)))
        print(f"  seed {seed}  ({duration(path):.3f}s)", flush=True)
        subprocess.run(["aplay", "-D", tts.SPEAKER_DEVICE, path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.4)

    print(f"\n  python3 scripts/render_phrases.py pick {key} {index} <seed>")


def cmd_pick(key, index, seed):
    """Commit an auditioned seed. Copies rather than re-rendering, because
    ElevenLabs treats seed as best effort and a re-render could draw something
    other than the reading that was just approved."""
    source = os.path.join(AUDITION_DIR, f"{key}.{index}.seed{seed}.wav")
    if not os.path.exists(source):
        print(f"No audition at seed {seed}. Run audition first.")
        return
    with wave.open(source, "rb") as w:
        pcm = w.readframes(w.getnframes())
    write_wav(phrasebank.path_for(key, index), pcm)
    print(f"{key}.{index} committed from seed {seed}. Live on the next play, no restart.")


def cmd_trim():
    """Re-trim what is already on disk, without spending an API call.

    Separate from render because ElevenLabs treats seed as best effort rather
    than a guarantee, so re-rendering to pick up a trimming change risks
    quietly redrawing the voice. Trimming in place cannot."""
    for key, texts in phrasebank.PHRASES.items():
        for index in range(len(texts)):
            path = phrasebank.path_for(key, index)
            if not os.path.exists(path):
                continue
            before = duration(path)
            with wave.open(path, "rb") as w:
                pcm = w.readframes(w.getnframes())
            write_wav(path, trim_silence(pcm))
            after = duration(path)
            if after < before:
                print(f"{key}.{index}  {before:5.3f}s -> {after:5.3f}s  "
                      f"({(before - after) * 1000:.0f}ms of dead air)")


def cmd_list():
    for key, texts in phrasebank.PHRASES.items():
        print(f"\n{key}")
        for index, text in enumerate(texts):
            path = phrasebank.path_for(key, index)
            if os.path.exists(path):
                secs = duration(path)
                flag = "" if key != "ack" else (
                    "  near the chime" if secs <= CHIME_SECONDS + 0.15 else "  slower than the chime")
                print(f"  {index}  {secs:5.3f}s  {text!r}{flag}")
            else:
                print(f"  {index}      --   {text!r}  NOT RENDERED")


def cmd_render(only_key=None, force=False):
    os.makedirs(PHRASE_DIR, exist_ok=True)

    made = skipped = 0
    for key, texts in phrasebank.PHRASES.items():
        if only_key and key != only_key:
            continue
        for index, text in enumerate(texts):
            path = phrasebank.path_for(key, index)
            if os.path.exists(path) and not force:
                skipped += 1
                continue
            print(f"rendering {key}.{index}  {text!r} ...", flush=True)
            written = render_one(key, index, text)
            print(f"  {duration(written):.3f}s  {written}", flush=True)
            made += 1

    print(f"\n{made} rendered, {skipped} already present.")
    if made:
        print("Nothing to restart. The files are read at playback.")


if __name__ == "__main__":
    init_db()   # normalize_pronunciation reads the pronunciations table
    args = sys.argv[1:]
    command = args[0] if args else "list"

    if command == "list":
        cmd_list()
    elif command == "trim":
        cmd_trim()
    elif command == "preview":
        cmd_preview(args[1] if len(args) > 1 else None)
    elif command == "audition":
        cmd_audition(args[1], int(args[2]),
                     seeds=int(args[3]) if len(args) > 3 else 6)
    elif command == "pick":
        cmd_pick(args[1], int(args[2]), int(args[3]))
    elif command == "render":
        rest = [a for a in args[1:] if a != "--force"]
        cmd_render(only_key=rest[0] if rest else None,
                   force="--force" in args)
    else:
        print(__doc__)
        sys.exit(1)
