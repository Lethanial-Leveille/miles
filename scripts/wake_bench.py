#!/usr/bin/env python3
"""Compare wake word engines on the same audio: nothing runs live.

Asked Oct 30 2026. hey_nova.onnx wakes on his roommates' conversation several
times a day, and a speaker specific verifier was ruled out because guests must
still be able to wake her. Before switching engines or retraining, every
candidate gets scored here on identical audio, so the choice is measured
rather than taken on a vendor's reputation.

Every engine is reduced to one question per frame, did it fire. openWakeWord
reports a score and Porcupine only a detection, so fired or not is the only
comparison both can enter honestly.

Three sets, each reported on its own:

  eval    data/wake_eval quiet takes, channel 0 (the cold wake path). A fire
          near a cue he confirmed is a catch; a fire anywhere else is false.
          The fairest set: nothing in it was chosen by the current model.
  hits    data/wake_hits, his labels. Biased toward the current model, since
          every clip is something that already made it fire, and the window
          ends where it fired, so a slower engine may find the phrase cut off.
  room    every WAV under --room, false wakes per hour. Empty until room audio
          is kept; it is the only set that can show what a NEW engine falsely
          wakes on.

The takes over her playback are left out: barge in reads channel 1 at its own
threshold, and is a separate question.

    python3 scripts/wake_bench.py
    PICOVOICE_ACCESS_KEY=... python3 scripts/wake_bench.py --porcupine models/hey_nova_pi.ppn
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import (CHUNK, RATE, WAKE_HIT_DIR, WAKE_MODEL_PATH, WAKE_THRESHOLD,
                    WAKE_MISS_PREROLL_MS)
import wake_gate

EVAL_DIR = os.path.expanduser("~/miles/data/wake_eval")
ROOM_DIR = os.path.expanduser("~/miles/data/room_audio")

# The live loop records after a wake and cannot fire again until it is done, so
# one phrase must count once. Two seconds covers the tail of "hey nova" without
# hiding a second, separate false wake.
REFRACTORY_S = 2.0

# A catch is a fire from just before his cue to a little after it: the cue is
# when he was prompted, and he speaks a beat later.
CUE_BEFORE_S = 0.5
CUE_AFTER_S = 3.0


class OpenWakeWord:
    """The live model at the live threshold, nothing else."""

    frame_samples = CHUNK

    def __init__(self, path=WAKE_MODEL_PATH, threshold=WAKE_THRESHOLD):
        from openwakeword.model import Model
        self.model = Model(wakeword_model_paths=[path])
        self.name_in_model = next(iter(self.model.models))
        self.threshold = threshold
        self.label = f"openwakeword {os.path.basename(path)} @ {threshold}"
        self.last_score = 0.0

    def reset(self):
        self.model.reset()

    def fired(self, frame):
        self.last_score = self.model.predict(frame)[self.name_in_model]
        return self.last_score >= self.threshold


class LiveGates:
    """openWakeWord with the level gate and the wake rule, as voice_main runs it.

    Wraps the engine rather than copying its threshold, so a candidate model
    can be scored with or without the gates by passing it in."""

    def __init__(self, engine):
        self.engine = engine
        self.frame_samples = engine.frame_samples
        # voice_main's own expression, so the gates see the window they see live
        self.window = collections.deque(
            maxlen=max(1, int(WAKE_MISS_PREROLL_MS / (CHUNK / RATE * 1000))))
        self.label = engine.label + " + live gates"

    def reset(self):
        self.engine.reset()
        self.window.clear()

    def fired(self, frame):
        self.window.append(frame.tobytes())
        if not self.engine.fired(frame):
            return False
        level = wake_gate.window_peak_dbfs(list(self.window))
        if wake_gate.too_quiet(level, len(self.window), self.window.maxlen):
            return False
        return not wake_gate.low_score_needs_louder(
            self.engine.last_score, level, len(self.window), self.window.maxlen)


class Porcupine:
    """A Picovoice keyword file. pvporcupine is imported here, not at the top,
    so the bench runs without it installed."""

    def __init__(self, keyword_path, sensitivity=0.5):
        import pvporcupine
        key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not key:
            sys.exit("Porcupine needs PICOVOICE_ACCESS_KEY in the environment.")
        self.handle = pvporcupine.create(access_key=key, keyword_paths=[keyword_path],
                                         sensitivities=[sensitivity])
        self.frame_samples = self.handle.frame_length
        self.label = f"porcupine {os.path.basename(keyword_path)} @ {sensitivity}"

    def reset(self):
        # Porcupine keeps no state an application can clear; feeding it a
        # second of silence flushes whatever the last clip left behind.
        silence = np.zeros(self.frame_samples, dtype=np.int16)
        for _ in range(RATE // self.frame_samples):
            self.handle.process(silence)

    def fired(self, frame):
        return self.handle.process(frame) >= 0


def fire_times(engine, samples):
    """Seconds at which the engine fired, one per refractory window."""
    engine.reset()
    n = engine.frame_samples
    times, quiet_until = [], 0.0
    for start in range(0, len(samples) - n + 1, n):
        t = (start + n) / RATE
        if engine.fired(samples[start:start + n]) and t >= quiet_until:
            times.append(t)
            quiet_until = t + REFRACTORY_S
    return times


def match_cues(fires, cues):
    """(cues caught, fires that belong to no cue). One fire catches one cue."""
    unused = list(fires)
    caught = 0
    for at in cues:
        hit = next((t for t in unused if at - CUE_BEFORE_S <= t <= at + CUE_AFTER_S), None)
        if hit is not None:
            caught += 1
            unused.remove(hit)
    return caught, len(unused)


def read_wav(path, channel=0):
    """int16 mono at RATE: the channel the cold wake path reads."""
    with wave.open(path) as w:
        if w.getframerate() != RATE or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 bit at {RATE}Hz")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return np.ascontiguousarray(data.reshape(-1, w.getnchannels())[:, channel])


def bench_eval(engine):
    cues = caught = false = 0
    seconds = 0.0
    by_spot = collections.defaultdict(lambda: [0, 0])
    for manifest in sorted(glob.glob(os.path.join(EVAL_DIR, "*", "manifest.jsonl"))):
        folder = os.path.dirname(manifest)
        with open(manifest) as f:
            takes = [json.loads(line) for line in f if line.strip()]
        for take in takes:
            if take["kind"] != "quiet":
                continue
            samples = read_wav(os.path.join(folder, take["file"]))
            said = [c["at"] for c in take["cues"] if c.get("said")]
            got, extra = match_cues(fire_times(engine, samples), said)
            cues += len(said); caught += got; false += extra
            seconds += len(samples) / RATE
            by_spot[take["spot"]][0] += got
            by_spot[take["spot"]][1] += len(said)
    print(f"  eval:  caught {caught} of {cues} cued wakes, "
          f"{false} false in {seconds / 60:.1f} min of takes")
    for spot, (got, total) in sorted(by_spot.items()):
        print(f"         {spot}: {got} of {total}")


def bench_hits(engine):
    counts = collections.Counter()
    path = os.path.join(WAKE_HIT_DIR, "labels.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clip = os.path.join(WAKE_HIT_DIR, row["file"])
            if row["label"] not in ("wake_phrase", "not_wake_phrase") or not os.path.exists(clip):
                continue
            fired = bool(fire_times(engine, read_wav(clip)))
            counts[(row["label"], fired)] += 1
    his = counts[("wake_phrase", True)] + counts[("wake_phrase", False)]
    other = counts[("not_wake_phrase", True)] + counts[("not_wake_phrase", False)]
    print(f"  hits:  his {counts[('wake_phrase', True)]} of {his} fired, "
          f"false {counts[('not_wake_phrase', True)]} of {other} fired (biased to the current model)")


def bench_room(engine, room_dir):
    files = sorted(glob.glob(os.path.join(room_dir, "**", "*.wav"), recursive=True))
    if not files:
        print(f"  room:  no audio under {room_dir}")
        return
    fires, seconds = 0, 0.0
    for path in files:
        samples = read_wav(path)
        fires += len(fire_times(engine, samples))
        seconds += len(samples) / RATE
    print(f"  room:  {fires} false in {seconds / 3600:.2f} h, "
          f"{fires / (seconds / 3600):.1f} per hour")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--porcupine", help="a .ppn keyword file to bench beside the live model")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Porcupine's, 0 to 1")
    parser.add_argument("--room", default=ROOM_DIR)
    parser.add_argument("--sets", default="eval,hits,room")
    args = parser.parse_args()

    live = OpenWakeWord()
    engines = [live, LiveGates(live)]
    if args.porcupine:
        engines.append(Porcupine(args.porcupine, args.sensitivity))

    sets = args.sets.split(",")
    for engine in engines:
        print(engine.label)
        if "eval" in sets:
            bench_eval(engine)
        if "hits" in sets:
            bench_hits(engine)
        if "room" in sets:
            bench_room(engine, args.room)


if __name__ == "__main__":
    main()
