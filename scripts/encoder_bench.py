#!/usr/bin/env python3
"""Compare speaker encoders on the clips that actually failed.

The question is narrow and answerable today: does a modern encoder hold up on
short utterances, where Resemblyzer collapses? Across 149 logged verification
attempts every rejection came from under two seconds, and there were none at
all in 94 attempts above it.

Method. Build a centroid from the long archived clips, which is a stand in for
enrollment, then score every clip against it and look at how similarity decays
as clips get shorter. A model that holds its score on a one second clip is the
one worth swapping to.

What this cannot measure, and it matters: every archived recording is
Lethanial. There is no impostor audio, so this says nothing about false
acceptance. A model that scored everything at 0.99 would look perfect here and
be useless. Treat the result as one half of the picture, and run the real EER
after the move when there is somebody else's voice to test against.

    python3 scripts/encoder_bench.py
    python3 scripts/encoder_bench.py --models resemblyzer ecapa
"""

import argparse
import os
import sqlite3
import statistics
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from config import DB_PATH

SR = 16000
ENROLL_MIN_SECONDS = 3.0     # clips long enough to stand in for enrollment
BUCKETS = ((0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0))


def clips():
    """Archived recordings that still exist, with the duration already logged."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT recording_path, embedded_duration_seconds, similarity "
        "FROM verification_log "
        "WHERE recording_path IS NOT NULL AND outcome = 'scored' "
        "AND embedded_duration_seconds > 0"
    ).fetchall()
    conn.close()
    return [(p, d, s) for p, d, s in rows if p and os.path.exists(p)]


def _read(path):
    import wave
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ── encoders ──────────────────────────────────────────────────────────────
# Each returns a function taking a float32 waveform at 16k and giving a unit
# length vector, so the comparison is like for like.

def load_resemblyzer():
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()

    def embed(wav):
        processed = preprocess_wav(wav, source_sr=SR)
        if len(processed) < SR * 0.2:
            return None
        vector = encoder.embed_utterance(processed)
        return vector / np.linalg.norm(vector)
    return embed


def load_ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain-ecapa"),
        run_opts={"device": "cpu"})

    def embed(wav):
        if len(wav) < SR * 0.2:
            return None
        with torch.no_grad():
            vector = model.encode_batch(
                torch.from_numpy(wav).unsqueeze(0)).squeeze().numpy()
        return vector / np.linalg.norm(vector)
    return embed


ENCODERS = {"resemblyzer": load_resemblyzer, "ecapa": load_ecapa}


def evaluate(name, samples):
    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    try:
        embed = ENCODERS[name]()
    except Exception as exc:
        print(f"  unavailable: {exc}")
        return None

    vectors = {}
    for path, duration, _ in samples:
        vector = embed(_read(path))
        if vector is not None:
            vectors[path] = vector

    enroll = [v for p, v in vectors.items()
              if dict((x[0], x[1]) for x in samples)[p] >= ENROLL_MIN_SECONDS]
    if len(enroll) < 3:
        print(f"  only {len(enroll)} clips over {ENROLL_MIN_SECONDS}s, "
              f"not enough to stand in for enrollment")
        return None

    centroid = np.mean(enroll, axis=0)
    centroid /= np.linalg.norm(centroid)
    print(f"  centroid from {len(enroll)} clips of {ENROLL_MIN_SECONDS}s or more")

    scored = []
    for path, duration, _ in samples:
        if path in vectors:
            scored.append((duration, float(np.dot(vectors[path], centroid))))

    print(f"\n  {'duration':12} {'n':>3} {'median':>8} {'min':>8}")
    result = {}
    for low, high in BUCKETS:
        group = [s for d, s in scored if low <= d < high]
        if not group:
            continue
        label = f"{low:.0f} to {high:.0f}s" if high < 90 else f"{low:.0f}s and up"
        median = statistics.median(group)
        result[label] = median
        print(f"  {label:12} {len(group):>3} {median:>8.3f} {min(group):>8.3f}")

    short = [s for d, s in scored if d < 2.0]
    long_ = [s for d, s in scored if d >= 3.0]
    if short and long_ and len(short) > 1 and len(long_) > 1:
        drop = statistics.median(long_) - statistics.median(short)

        # Raw drop is not comparable across encoders and comparing it was a
        # mistake. Resemblyzer's embeddings are non negative and even unrelated
        # audio floors around 0.75, so its whole usable range is roughly a
        # quarter as wide as one that can go negative. A compressed scale shows
        # a smaller absolute drop by construction, which made the older model
        # look more robust when it was only more cramped.
        #
        # Cohen's d divides the gap by the pooled spread of the two groups, so
        # it asks how far short clips fall relative to how much this encoder's
        # scores vary anyway. That is scale free and comparable.
        s_sd, l_sd = statistics.stdev(short), statistics.stdev(long_)
        pooled = (((len(short) - 1) * s_sd ** 2 + (len(long_) - 1) * l_sd ** 2)
                  / (len(short) + len(long_) - 2)) ** 0.5
        d = drop / pooled if pooled else float("inf")
        result["_drop"] = drop
        result["_d"] = d
        result["_spread"] = pooled
        print(f"\n  raw drop, long to under 2s : {drop:.3f}")
        print(f"  this encoder's own spread   : {pooled:.3f}")
        print(f"  drop in units of spread (d) : {d:.2f}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(ENCODERS),
                   choices=list(ENCODERS))
    args = p.parse_args()

    samples = clips()
    print(f"{len(samples)} archived clips on disk with a logged duration")
    if len(samples) < 10:
        print("Not enough to say anything. Use Nova for a few days first.")
        return

    results = {name: evaluate(name, samples) for name in args.models}

    usable = {k: v for k, v in results.items() if v and "_d" in v}
    if len(usable) > 1:
        print(f"\n{'=' * 66}\nVERDICT\n{'=' * 66}")
        print(f"  {'encoder':14} {'raw drop':>9} {'spread':>8} {'d':>7}")
        for name, r in sorted(usable.items(), key=lambda kv: kv[1]["_d"]):
            print(f"  {name:14} {r['_drop']:>9.3f} {r['_spread']:>8.3f} {r['_d']:>7.2f}")
        best = min(usable, key=lambda k: usable[k]["_d"])
        print(f"\n  {best} degrades least, measured against its own spread.")
        print("\n  Read raw drop only within one encoder, never across them: an")
        print("  encoder whose scores are squeezed into a narrow band shows a")
        print("  smaller drop for free. d is the column that compares.")
        print("\n  This is false rejection only. Every clip is him, so nothing")
        print("  here says whether a model wrongly accepts somebody else. That")
        print("  needs impostor audio and therefore needs the move.")


if __name__ == "__main__":
    main()
