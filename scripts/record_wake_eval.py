#!/usr/bin/env python3
"""Record the wake word evaluation set: him saying "hey nova", over her and not.

A retrained wake model can only be trusted if it beats the current one on the
same recordings, so this set is recorded once, kept, and every candidate model
is scored on it. Rerun it after a change to the room (a lower volume, the
enclosure) into a new session directory, never over an old one.

Three kinds of take, all recorded on both of the array's channels, because
which channel the barge in watcher should read is one of the questions:

    over     one of her real replies plays, with the echo reference on as barge
             in would run it, and a cue appears twice; he says "hey nova" once
             per cue. The cue times are the ground truth.
    her      her reply alone. Nothing here may fire, since a barge in that
             fires on her own voice cuts off every reply.
    quiet    no playback, one cue: the normal wake, so a model that learns her
             voice and forgets the quiet room shows up as a regression.

After every take he confirms whether he said it at each cue, and that answer
is the label. Sep 30 2026: a survey's cues never showed on his terminal and he
talked whenever she did; the log said otherwise. A cue is an intention, only
his answer says what happened.

These clips are evaluation only. They must never be used to train a model, or
the score measures memory instead of hearing.

    python3 scripts/record_wake_eval.py             # a full session, about fifteen minutes
    python3 scripts/record_wake_eval.py --takes 3   # a short try of each kind
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import tts
from config import MIC_NAME_HINT
from label_wake import _voice_service_paused

EVAL_DIR = os.path.join(ROOT, "data", "wake_eval")
SPOKEN_DIR = os.path.join(ROOT, "data", "spoken")

README = """# Wake word evaluation set

Recorded by scripts/record_wake_eval.py. Him saying "hey nova" at cues, over
her replies and in a quiet room, plus her replies alone. Both array channels,
16kHz stereo: channel 0 processed, channel 1 the recognition beam.

EVALUATION ONLY. Never train on these clips. A model trained on its own test
set scores itself on memory, and the number stops meaning anything.

Never prune this directory. A label with no audio cannot be checked again
(INCIDENTS.md, the wake_misses eviction).

manifest.jsonl, one line per take: file, kind, spot, clip played, where
playback starts and ends in the recording, and each cue with whether he
confirmed saying it.
"""

# Where he is. Spread across the takes so the set is not one position.
SPOTS = ["your usual seat", "your other usual spot", "wherever you would naturally be"]


def _clips(minimum_s=10.0, maximum_s=30.0):
    """Her replies of a length that leaves room for two cues."""
    found = []
    for name in sorted(os.listdir(SPOKEN_DIR)):
        if not name.endswith(".wav"):
            continue
        with wave.open(os.path.join(SPOKEN_DIR, name)) as w:
            seconds = w.getnframes() / w.getframerate()
        if minimum_s <= seconds <= maximum_s:
            found.append(name)
    return found


class _Clip:
    """A recorded reply, handed to play() the way a synthesis would be."""

    def __init__(self, pcm):
        self.pcm = pcm
        self.requested_at = time.monotonic()
        self.error = None

    def chunks(self):
        for i in range(0, len(self.pcm), 4096):
            yield self.pcm[i:i + 4096]


def _cue(at, t0, times):
    time.sleep(max(0.0, t0 + at - time.monotonic()))
    times.append(round(time.monotonic() - t0, 3))
    # Written to the real terminal and flushed at once. Printed from a thread
    # while play() runs, which is the one place a cue went missing before.
    sys.__stdout__.write("\n" + "#" * 40 + "\n###   SAY  \"HEY NOVA\"  NOW   ###\n"
                         + "#" * 40 + "\n")
    sys.__stdout__.flush()


def _confirm(n_cues):
    said = []
    for i in range(n_cues):
        answer = input(f"    cue {i + 1}: did you say it? [Enter = yes, n = no] ").strip().lower()
        said.append(answer != "n")
    return said


def _take(session, number, kind, spot, clip, mic):
    path = os.path.join(session, f"{number:02d}_{kind}.wav")
    pcm, seconds = b"", 5.0
    if clip is not None:
        with wave.open(os.path.join(SPOKEN_DIR, clip)) as w:
            pcm = w.readframes(w.getnframes())
        seconds = len(pcm) / 2 / 22050

    # Cues at random moments, kept clear of the start and end and of each
    # other, so the model is heard over different parts of her sentences.
    if kind == "over":
        first = random.uniform(2.0, seconds / 2 - 1.5)
        cues_at = [first, random.uniform(max(first + 3.0, seconds / 2), seconds - 2.0)]
    elif kind == "quiet":
        cues_at = [1.5]
    else:
        cues_at = []

    where = f" from {spot}" if spot else ""
    what = {"over": "she plays; say it at each cue", "her": "she plays; stay silent",
            "quiet": "no playback; say it at the cue"}[kind]
    print(f"\n=== take {number}: {kind}{where}, {seconds:.0f}s. {what}")
    input("    press Enter when ready... ")

    rec = subprocess.Popen(["arecord", "-q", "-D", mic, "-f", "S16_LE", "-r", "16000",
                            "-c", "2", "-d", str(int(seconds + 3)), path])
    t0 = time.monotonic()
    time.sleep(1.0)
    play_at = time.monotonic() - t0
    heard_at = []
    cue_threads = [threading.Thread(target=_cue, args=(1.0 + at, t0, heard_at))
                   for at in cues_at]
    for c in cue_threads:
        c.start()
    if pcm:
        tts.play(_Clip(pcm))
    for c in cue_threads:
        c.join()
    rec.wait()

    said = _confirm(len(cues_at))
    return {"file": os.path.basename(path), "kind": kind, "spot": spot, "clip": clip,
            "play_at": round(play_at, 3), "play_s": round(seconds, 3) if pcm else 0.0,
            "cues": [{"at": at, "said": s} for at, s in zip(heard_at, said)]}


def _plan(takes):
    """The order he goes through: over her by spot, her alone, then quiet."""
    plan = []
    for spot in SPOTS:
        plan += [("over", spot)] * takes
    plan += [("her", None)] * max(1, takes * 4 // 5)
    for spot in SPOTS:
        plan += [("quiet", spot)] * max(1, takes * 2 // 3)
    return plan


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--takes", type=int, default=5,
                        help="over her takes per spot (default 5)")
    args = parser.parse_args()

    mic_card = tts._resolve_speaker_device(MIC_NAME_HINT, fallback=None)
    if mic_card is None:
        sys.exit("The array was not found; nothing recorded.")
    tts.REFERENCE_DEVICE = mic_card          # the echo reference, as barge in runs it
    tts.timing.note_tts = lambda *a, **k: None
    mic = mic_card.replace("plughw", "hw")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "README.md"), "w") as f:
        f.write(README)
    session = os.path.join(EVAL_DIR, time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(session)

    clips = _clips()
    random.shuffle(clips)
    plan = _plan(args.takes)
    print(f"{len(plan)} takes into {session}. Ctrl+C stops; takes already done are kept.")

    manifest = os.path.join(session, "manifest.jsonl")
    with _voice_service_paused():
        for number, (kind, spot) in enumerate(plan, 1):
            clip = clips[number % len(clips)] if kind in ("over", "her") else None
            row = _take(session, number, kind, spot, clip, mic)
            # One line per take, written at once, so an interrupted session
            # keeps everything it recorded.
            with open(manifest, "a") as f:
                f.write(json.dumps(row) + "\n")
    print(f"\nDone: {session}")


if __name__ == "__main__":
    main()
