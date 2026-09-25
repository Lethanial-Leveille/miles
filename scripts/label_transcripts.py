#!/usr/bin/env python3
"""Hand labelled transcripts, so an STT change can be scored against the truth.

Every STT comparison in this repo so far has scored candidates against base.en
as if base.en were correct. CLAUDE.md already flags that as unreliable, and on
Aug 13 2026 it became the thing blocking a real decision: an initial prompt
disagreed with base.en on 16 of 40 clips and was plainly BETTER on several of
them, including turning "7 times for 5 minutes." into "Set a timer for five
minutes." A disagreement count cannot tell those apart from regressions, so the
comparison could not answer the question it was run to answer.

This produces the missing ruler. Listen to a clip, write down what was actually
said, and every future candidate gets a word error rate against that instead of
against whichever model happened to be installed first.

    python3 scripts/label_transcripts.py init          # sample the archive
    python3 scripts/label_transcripts.py               # label what is blank
    python3 scripts/label_transcripts.py --blind       # no candidate shown
    python3 scripts/label_transcripts.py --status
    python3 scripts/label_transcripts.py score         # WER per candidate

## On anchoring

Labelling is seeded with what the current model heard, because typing sixty
transcripts from scratch is how a labelling set ends up half finished. That
seeding is a real bias: a subtly wrong transcript that sounds plausible is
easier to accept than to notice.

`--blind` hides the candidate and makes you type it cold. Label a dozen that
way and compare against the anchored labels for the same clips. If they agree,
the anchoring is not costing anything. If they do not, that is worth knowing
before trusting any number this produces.

A label you are unsure of is worse than no label. Skip it.
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from config import (ARCHIVE_DIR, WHISPER_CLI, WHISPER_AUDIO_CTX,  # noqa: E402
                    WHISPER_MODEL, DB_PATH)
import tts                                                        # noqa: E402

EVAL_DIR = os.path.expanduser("~/miles/data/stt_eval")
MANIFEST = os.path.join(EVAL_DIR, "manifest.csv")
CACHE    = os.path.join(EVAL_DIR, "transcripts.json")

# Manifest versions kept behind the current one, as manifest.csv.0 upward.
BACKUPS = 5

FIELDS = ["file", "duration_s", "turn_type", "truth", "mode", "note"]

MODELS = os.path.expanduser("~/miles/whisper.cpp/models")
PROMPT = "Nova is a voice assistant running on a Raspberry Pi. Lethanial is in Gainesville."

# How much prior conversation to prime the decoder with. whisper.cpp caps the
# prompt well below this, and the tail is what matters because it is the words
# closest to the clip, so this trims from the front.
PRIOR_TURN_CHARS = 220

BEAM_5 = ["-bs", "5", "-bo", "5"]


def _prior_turns(clip_name, limit=2):
    """The last few messages saved before this clip was recorded.

    sqlite3 directly rather than database.py: this is an offline scoring script
    and it has no business running migrations against the production database or
    holding it open for writes."""
    import sqlite3
    from datetime import datetime
    try:
        stamp = datetime.strptime(clip_name.split("_")[0], "%Y%m%dT%H%M%S")
    except ValueError:
        return []
    try:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        rows = con.execute(
            """SELECT content FROM conversation_history
               WHERE created_at <= ? ORDER BY created_at DESC LIMIT ?""",
            (stamp.isoformat(), limit)).fetchall()
        con.close()
    except sqlite3.Error:
        return []
    return [r[0] for r in reversed(rows) if r[0]]


def prior_turn_prompt(row):
    """The conversation so far, as whisper's initial prompt, on follow ups only.

    Follow ups score 21.9% WER against 8.4% on initial commands, and utterances
    of four words or fewer score 18.5% against 10.9% for longer ones. Both point
    the same way: the decoder's language prior is doing the work on short input
    and it has nothing to go on. "what's the date" became "was the beat" and
    "Set a timer for" became "Seven timer four" on clean audio, so this is not a
    noise problem and a better microphone will not touch it.

    An initial command gets no prompt, which is the point. A static domain prompt
    was measured across every clip on Aug 13 2026 and landed 0.3 points worse,
    inside the noise; applying context only where context is missing is the part
    that was never tried."""
    if row.get("turn_type") != "followup":
        return None
    prior = " ".join(_prior_turns(row["file"]))
    return prior[-PRIOR_TURN_CHARS:].strip() or None


def domain_prompt_followups_only(row):
    """The Aug 13 prompt, restricted to follow ups.

    Separates two readings of that result. If the prompt genuinely buys nothing,
    this lands on the live model. If it was helping follow ups and hurting
    initials, the aggregate was hiding both and this shows it."""
    return PROMPT if row.get("turn_type") == "followup" else None

# Everything to be scored. Add a row, run `score`, and it appears. The label set
# does not change when this does, which is the entire point of having one.
# (label, model, prompt, decoder flags). A prompt may be a callable of the
# manifest row, so it can differ per clip. Decoder flags default to greedy, which
# is what every measurement in this repo was taken with and what production runs.
CANDIDATES = [
    ("base.en fp16",          f"{MODELS}/ggml-base.en.bin",      None, None),
    ("base.en q8_0 (live)",   f"{MODELS}/ggml-base.en-q8_0.bin", None, None),
    ("base.en q8_0 + prompt", f"{MODELS}/ggml-base.en-q8_0.bin", PROMPT, None),
    ("tiny.en",               f"{MODELS}/ggml-tiny.en.bin",      None, None),
    ("q8_0 + prior turn (fu)", f"{MODELS}/ggml-base.en-q8_0.bin", prior_turn_prompt, None),
    ("q8_0 + domain (fu)",     f"{MODELS}/ggml-base.en-q8_0.bin", domain_prompt_followups_only, None),
    # Beam search. whisper.cpp defaults to 5 and this pipeline has always passed
    # 1, so no comparison in the repo has ever varied it. It is the standard lever
    # for the error this set is full of: a plausible wrong word that greedy
    # commits to because it never considers the alternative.
    ("q8_0 beam 5",           f"{MODELS}/ggml-base.en-q8_0.bin", None, BEAM_5),
    ("base.en fp16 beam 5",   f"{MODELS}/ggml-base.en.bin",      None, BEAM_5),
    # Bigger, and slower in proportion. Whether the accuracy is worth the latency
    # is a separate question from whether it is more accurate, and the head start
    # in docs/AUDIO_PIPELINE.md is what decides it.
    ("small.en q8_0",         f"{MODELS}/ggml-small.en-q8_0.bin", None, None),
    ("small.en q8_0 beam 5",  f"{MODELS}/ggml-small.en-q8_0.bin", None, BEAM_5),
    # distil-whisper: near small.en accuracy at near base.en speed, in principle.
    # whisper.cpp's support is partial, since it does not implement the chunked
    # strategy the distilled models were trained to use, so measure rather than
    # trust the published numbers.
    ("distil-small.en",       f"{MODELS}/ggml-distil-small.en.bin", None, None),
    ("distil-small.en beam 5", f"{MODELS}/ggml-distil-small.en.bin", None, BEAM_5),
]

# Non speech, however whisper chose to spell it this time. These normalize to an
# empty truth so a clip of a closing door is not scored as four wrong words.
_ANNOTATION = re.compile(r"[\[(][^\])]*[\])]")


def normalize(text):
    """Lowercase, drop punctuation and annotations, collapse whitespace.

    Scoring is on words, not on how whisper punctuates. Comma placement is not
    an error worth counting and would swamp the errors that are."""
    text = _ANNOTATION.sub(" ", text or "")
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return " ".join(text.split())


def wer(truth, hypothesis):
    """Word error rate: edits to turn the hypothesis into the truth, over the
    length of the truth. Levenshtein over word lists."""
    t, h = normalize(truth).split(), normalize(hypothesis).split()
    if not t:
        # Silence labelled as silence is correct; anything heard in it is not.
        return 0.0 if not h else 1.0

    previous = list(range(len(h) + 1))
    for i, truth_word in enumerate(t, 1):
        current = [i]
        for j, hypothesis_word in enumerate(h, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (truth_word != hypothesis_word)))
        previous = current
    return previous[-1] / len(t)


def duration(path):
    with wave.open(path) as w:
        return round(w.getnframes() / w.getframerate(), 1)


def transcribe(path, model, prompt=None, extra=None):
    """Greedy unless `extra` says otherwise, because greedy is what runs live."""
    cmd = [WHISPER_CLI, "-m", model, "-f", path,
           "--no-prints", "--no-timestamps", "-ac", str(WHISPER_AUDIO_CTX)]
    cmd += list(extra) if extra else ["-bs", "1", "-bo", "1"]
    if prompt:
        cmd += ["--prompt", prompt]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def load():
    if not os.path.exists(MANIFEST):
        sys.exit(f"No manifest. Run: {sys.argv[0]} init")
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def save(rows):
    """Write the manifest, keeping the previous version.

    Labels cost an hour of listening and this file is the only copy: data/ is
    gitignored, deliberately, because these are transcripts of real speech in a
    real room. On Aug 13 2026 a bug in blind mode overwrote 49 of them with the
    single letter "a" and there was nothing to roll back to.

    Rotating backups rather than one, because the damage was only obvious after
    a second save had already run."""
    os.makedirs(EVAL_DIR, exist_ok=True)

    if os.path.exists(MANIFEST):
        for n in range(BACKUPS - 1, 0, -1):
            older, newer = f"{MANIFEST}.{n}", f"{MANIFEST}.{n - 1}"
            if os.path.exists(newer):
                shutil.copy(newer, older)
        shutil.copy(MANIFEST, f"{MANIFEST}.0")

    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=1)


def cmd_init(args):
    """Sample the archive, stratified by clip length.

    Random alone over samples the two second clips, because most turns are two
    seconds. Length is exactly what stresses a decoder differently, so the long
    ones have to be represented or the score describes short utterances only."""
    clips = sorted(f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".wav"))
    if not clips:
        sys.exit(f"No recordings in {ARCHIVE_DIR}")

    by_length = sorted(clips, key=lambda f: duration(os.path.join(ARCHIVE_DIR, f)))
    buckets = [by_length[i::3] for i in range(3)]      # short, middle, long

    random.seed(args.seed)
    per = max(1, args.n // 3)
    chosen = []
    for bucket in buckets:
        chosen += random.sample(bucket, min(per, len(bucket)))

    existing = {}
    if os.path.exists(MANIFEST):
        existing = {r["file"]: r for r in load()}

    rows = []
    for name in sorted(chosen):
        if name in existing:
            rows.append(existing[name])
            continue
        path = os.path.join(ARCHIVE_DIR, name)
        rows.append({"file": name, "duration_s": duration(path),
                     "turn_type": "followup" if "followup" in name else "initial",
                     "truth": "", "mode": "", "note": ""})

    # Anything already labelled is kept even if this sample missed it. Labels
    # are expensive and throwing one away to satisfy a sampling rule is a bad
    # trade.
    for name, row in existing.items():
        if row["truth"] and name not in {r["file"] for r in rows}:
            rows.append(row)

    save(rows)
    labelled = sum(1 for r in rows if r["truth"])
    print(f"{len(rows)} clips in {MANIFEST} ({labelled} already labelled)")


PLAYBACK_WAV = os.path.join(EVAL_DIR, "_playback.wav")
PLAYBACK_PEAK_DBFS = -6.0


def normalized_for_listening(path, target_dbfs=PLAYBACK_PEAK_DBFS):
    """Boost a clip to a consistent peak, for ears rather than for the pipeline.

    The archive is quiet and wildly uneven: measured Aug 13 2026 over twelve
    clips, peaks ran from -20 to -52 dBFS with a median of -32. No single
    speaker volume makes that set audible, because the quiet end is 30 dB below
    the loud end, so raising the system volume enough to hear the worst one
    makes the best one painful.

    Per clip normalization is the fix, and it belongs here rather than in the
    ALSA mixer: the mixer is shared with Nova's own voice, and labelling should
    not change how loud she is in the room.

    Playback only. Nothing scored ever reads this file, because changing the
    samples is exactly what would invalidate the comparison the labels exist to
    support."""
    with wave.open(path) as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    peak = np.abs(samples).max()
    if not peak:
        return path

    target = (10 ** (target_dbfs / 20)) * 32767
    boosted = np.clip(samples * (target / peak), -32768, 32767).astype(np.int16)

    os.makedirs(EVAL_DIR, exist_ok=True)
    with wave.open(PLAYBACK_WAV, "wb") as w:
        w.setparams(params)
        w.writeframes(boosted.tobytes())
    return PLAYBACK_WAV


def play(path, raw=False):
    if not raw:
        path = normalized_for_listening(path)
    subprocess.run(["aplay", "-D", tts.SPEAKER_DEVICE, path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cmd_status(args):
    rows = load()
    done = [r for r in rows if r["truth"]]
    blind = [r for r in done if r["mode"] == "blind"]
    empty = [r for r in done if not normalize(r["truth"])]
    print(f"  labelled     {len(done)}/{len(rows)}")
    print(f"    of those blind {len(blind)}")
    print(f"    non speech     {len(empty)}")
    words = sum(len(normalize(r['truth']).split()) for r in done)
    print(f"  words of truth  {words}")
    if len(done) < 30:
        print("\n  Under thirty labels. A word error rate off this is directional")
        print("  at best, and a single bad clip moves it several points.")


def cmd_label(args):
    rows = load()
    todo = [r for r in rows if args.all or not r["truth"]]
    if not todo:
        print("Nothing to label. --all to revisit.")
        return

    print(f"{len(todo)} to label."
          f"{' BLIND, no candidate shown.' if args.blind else ''}")
    print("  enter = replay   a = accept shown   n = no speech")
    print("  s = skip   q = save and quit   anything else = the true transcript\n")

    for i, row in enumerate(todo, 1):
        path = os.path.join(ARCHIVE_DIR, row["file"])
        if not os.path.exists(path):
            row["note"] = "missing from archive"
            continue

        candidate = "" if args.blind else transcribe(path, WHISPER_MODEL)
        print(f"[{i}/{len(todo)}] {row['file']}  {row['duration_s']}s  {row['turn_type']}")
        if args.blind:
            # Nothing about the existing label may be shown here. Printing it
            # was the whole reason the first blind pass measured nothing: the
            # anchor it exists to remove was right there on screen.
            if row["truth"]:
                print("    (already labelled, answer without looking it up)")
        else:
            print(f"    heard: {candidate!r}")
            if row["truth"]:
                print(f"    current label: {row['truth']!r}")

        while True:
            play(path, raw=args.raw)
            answer = input("    > ").strip()
            if answer == "":
                continue
            if answer == "q":
                save(rows)
                print("saved")
                return
            if answer == "s":
                break
            if answer == "a":
                # In blind mode there is no candidate to accept, and "a" used to
                # fall through to the branch below and be saved as the
                # transcript. That silently overwrote 49 real labels with the
                # single letter "a". A command that means nothing here has to
                # say so, not be taken literally.
                if args.blind:
                    print("    no candidate is shown in blind mode. Type what "
                          "you heard, or s to skip.")
                    continue
                row["truth"] = candidate
                row["mode"] = "anchored"
                break
            if answer == "n":
                row["truth"] = ""
                row["mode"] = "blind" if args.blind else "anchored"
                row["note"] = "no speech"
                break
            # Any other single character is a mistyped command far more often
            # than it is a real transcript. Nothing said out loud to Nova is one
            # letter long.
            if len(answer) == 1:
                print(f"    {answer!r} is one character. If you really heard "
                      f"that, type it twice to confirm; otherwise a/n/s/q.")
                if input("    > ").strip() != answer * 2:
                    continue
            row["truth"] = answer
            row["mode"] = "blind" if args.blind else "anchored"
            break
        print()

    save(rows)
    print("saved\n")
    cmd_status(args)


def cmd_score(args):
    rows = [r for r in load() if r["truth"] or r["note"] == "no speech"]
    if not rows:
        sys.exit("Nothing labelled yet.")

    cache = load_cache()
    print(f"Scoring {len(rows)} labelled clips against {len(CANDIDATES)} candidates.\n")

    results = {}
    for label, model, prompt, extra in CANDIDATES:
        if not os.path.exists(model):
            print(f"  {label:24} SKIPPED, no model at {model}")
            continue
        scores, texts = [], []
        # Progress only on a terminal. Piped to a file it is one line per
        # clip of carriage returns, which is how a 4 minute run looks like
        # a hang in a log.
        tty = sys.stdout.isatty()
        if tty:
            print(f"  {label} ...", end="", flush=True)
        for n, row in enumerate(rows, 1):
            # A callable prompt is resolved per clip. Keying the cache on the
            # resolved text rather than the candidate means two candidates that
            # happen to produce the same prompt for a clip share the result, and
            # a static prompt keys exactly as it did before this existed.
            text = prompt(row) if callable(prompt) else prompt
            # The decoder flags are appended only when there are any, so every
            # greedy candidate keys exactly as it did before beam search existed
            # and nothing already measured has to be transcribed again.
            suffix = ("|" + " ".join(extra)) if extra else ""
            key = f"{os.path.basename(model)}|{text or ''}|{row['file']}{suffix}"
            if key not in cache:
                if tty:
                    print(f"\r  {label} ... {n}/{len(rows)}", end="", flush=True)
                cache[key] = transcribe(
                    os.path.join(ARCHIVE_DIR, row["file"]), model, text, extra)
            texts.append(cache[key])
            scores.append(wer(row["truth"], cache[key]))
        results[label] = (scores, texts)
        save_cache(cache)
        if tty:
            print(f"\r  {label} ... done{' ' * 20}")

    print(f"{'candidate':26} {'WER':>7} {'exact':>8} {'clips worse than truth':>24}")
    for label, (scores, _) in results.items():
        mean = sum(scores) / len(scores)
        exact = sum(1 for s in scores if s == 0)
        wrong = sum(1 for s in scores if s > 0)
        print(f"{label:26} {100*mean:6.1f}% {exact:>4}/{len(scores)} "
              f"{wrong:>23}")

    print("\nWER is edits per true word, so lower is better and 0% is perfect.")
    print("Exact counts clips where every word matched after normalization.")

    # Split, always. Follow ups score 21.9% against 8.4% on initial commands, so
    # the aggregate above is two populations averaged, and a change that helps
    # one while hurting the other reads as no change at all. That is exactly how
    # the Aug 13 2026 prompt comparison came out a wash.
    kinds = sorted({row["turn_type"] for row in rows if row.get("turn_type")})
    print(f"\n{'candidate':26} " + " ".join(f"{k:>12}" for k in kinds))
    for label, (scores, _) in results.items():
        cells = []
        for kind in kinds:
            sel = [s for s, row in zip(scores, rows)
                   if row.get("turn_type") == kind]
            cells.append(f"{100*sum(sel)/len(sel):11.1f}%" if sel else f"{'':>12}")
        print(f"{label:26} " + " ".join(cells))

    if args.verbose:
        best = min(results, key=lambda k: sum(results[k][0]) / len(results[k][0]))
        print(f"\n--- where {best} still gets it wrong ---")
        for row, score, text in zip(rows, *results[best]):
            if score > 0:
                print(f"\n  {row['file']}  WER {100*score:.0f}%")
                print(f"    truth: {row['truth']!r}")
                print(f"    heard: {text!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    init = sub.add_parser("init", help="sample the archive into a manifest")
    init.add_argument("--n", type=int, default=60)
    init.add_argument("--seed", type=int, default=7)

    score = sub.add_parser("score", help="word error rate per candidate")
    score.add_argument("-v", "--verbose", action="store_true",
                       help="show every clip the best candidate got wrong")

    ap.add_argument("--all", action="store_true", help="revisit labelled clips")
    ap.add_argument("--blind", action="store_true", help="hide the candidate")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--raw", action="store_true",
                    help="play at the recorded level, unnormalized")
    args = ap.parse_args()

    if args.cmd == "init":
        cmd_init(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.status:
        cmd_status(args)
    else:
        cmd_label(args)


if __name__ == "__main__":
    main()
