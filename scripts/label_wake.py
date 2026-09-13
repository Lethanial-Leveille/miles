#!/usr/bin/env python3
"""Label what the wake word model actually heard, by ear.

There are two questions and they need opposite fixes, which is why a score on
its own answers neither:

    wake_misses   she did not answer. Was that a real "hey nova" she missed, or
                  the room, correctly ignored?
    wake_hits     she answered. Was that a real "hey nova", or the television?

hey_nova.onnx is dated Apr 8 2026. It predates the mic gain tuning of Aug 10
and it predates this apartment. CLAUDE.md records the old room separating
cleanly, successful wakes from 0.520 up and failures topping out at 0.365, with
an empty band between. That is not what the new room does: observed wakes run
0.43, 0.46, 0.50, 0.64 and 0.75, and the low one was genuine while the high one
was not. Overlapping in both directions means no threshold fixes it, and the
model is the thing to look at.

This builds the set that makes that measurable. Without labels, "the model is
weak" and "those were never attempts" are indistinguishable, and there is
nothing to evaluate a retrain against.

Labelling by ear, and only by ear. A transcript cannot do it: the trigger audio
is the phrase itself, not the command after it, and the score is the thing
being judged so it cannot also be the judge. label_speakers.py makes the same
argument, and for the same reason: a wrong label does not crash, it quietly
reports the model as better or worse than it is.

    python3 scripts/label_wake.py hits            # label wakes that fired
    python3 scripts/label_wake.py misses          # label wakes that did not
    python3 scripts/label_wake.py hits --min 0.3  # only the loudest cases
    python3 scripts/label_wake.py --status        # counts, no playback

Labels: y = yes, I said the wake phrase
        n = no, I did not
        u = unclear, or someone else said it
        r = replay      s = skip      q = quit

Anything left unclear stays out of the evaluation rather than being guessed.
"""

import argparse
import contextlib
import csv
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from config import (WAKE_MISS_DIR, WAKE_HIT_DIR, WAKE_THRESHOLD,  # noqa: E402
                    SPEAKER_NAME_HINT)

SETS = {"hits": WAKE_HIT_DIR, "misses": WAKE_MISS_DIR}

# Playback peak to normalize to. These clips are not listenable raw: the loudest
# near miss in the archive peaks at -40 dBFS against -10 for real close speech,
# which is thirty dB down, and through desk speakers it is silence. Judging a
# clip you cannot hear is not labelling, it is guessing.
#
# This changes what the ear receives and never what the model scored. The score
# was computed from the original samples and is printed beside the gain applied,
# so a boosted clip can never be mistaken for a loud one.
PLAYBACK_TARGET_DBFS = -3.0

# Ceiling on the boost, so a clip containing nothing does not arrive as a wall
# of amplified noise floor. An empty room sits near -70 dBFS peak, which would
# otherwise be lifted by 67 dB.
PLAYBACK_MAX_GAIN_DB = 36.0
LABELS = {"y": "wake_phrase", "n": "not_wake_phrase", "u": "unclear"}
FIELDS = ["file", "score", "label"]


def _manifest_path(directory):
    return os.path.join(directory, "labels.csv")


def _load(directory):
    """Existing labels, so a session can be stopped and resumed.

    Keyed by filename rather than by index, because the capture directories are
    pruned as they fill and an index would silently point at a different clip
    after any eviction."""
    path = _manifest_path(directory)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def _save(directory, rows):
    path = _manifest_path(directory)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name in sorted(rows):
            writer.writerow(rows[name])


def _clips(directory, minimum):
    """Filenames, worst first.

    The score leads the filename precisely so this sort is meaningful: for
    hits, the highest scoring false positive is the most damaging, and for
    misses the highest scoring failure is the closest call."""
    if not os.path.isdir(directory):
        return []
    names = [f for f in os.listdir(directory) if f.endswith(".wav")]
    scored = []
    for name in names:
        try:
            scored.append((float(name.split("_")[0]), name))
        except ValueError:
            continue
    scored.sort(reverse=True)
    return [(s, n) for s, n in scored if s >= minimum]


def _levels(path):
    """Peak and RMS in dBFS, so a clip can be described before it is played."""
    with wave.open(path, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
        params = wf.getparams()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if not samples.size:
        return None, None, params, samples
    peak = float(np.abs(samples).max())
    rms = float(np.sqrt((samples ** 2).mean()))
    to_db = lambda v: 20 * math.log10(v) if v > 0 else None      # noqa: E731
    return to_db(peak), to_db(rms), params, samples


def _normalized_copy(path, target_dbfs=PLAYBACK_TARGET_DBFS):
    """A boosted copy for listening, plus the gain applied in dB.

    Returns (path_to_play, gain_db, peak_dbfs). The copy goes to a temp file
    rather than touching the capture directory, which is a ring buffer holding
    the only record of what the model actually heard."""
    peak_db, _, params, samples = _levels(path)
    if peak_db is None:
        return path, 0.0, None

    gain_db = min(target_dbfs - peak_db, PLAYBACK_MAX_GAIN_DB)
    if gain_db <= 0.5:
        return path, 0.0, peak_db

    boosted = np.clip(samples * (10 ** (gain_db / 20.0)), -1.0, 1.0)
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    with wave.open(handle.name, 'wb') as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes((boosted * 32767).astype(np.int16).tobytes())
    return handle.name, gain_db, peak_db


def _play(path, device):
    """Play one clip, and say so when it fails.

    stderr is surfaced rather than captured. A swallowed aplay error looks
    exactly like a clip that is simply too quiet to hear, and those need
    opposite responses. That confusion is what made this tool unusable on its
    first run."""
    done = subprocess.run(["aplay", "-D", device, path],
                          capture_output=True, text=True)
    if done.returncode != 0:
        message = (done.stderr or "").strip().splitlines()
        print(f"  aplay failed on {device}: "
              f"{message[-1] if message else done.returncode}")
        return False
    return True


def _service_active(unit="miles-voice"):
    done = subprocess.run(["systemctl", "is-active", unit],
                          capture_output=True, text=True)
    return done.stdout.strip() == "active"


@contextlib.contextmanager
def _voice_service_paused(unit="miles-voice"):
    """Stop the voice loop while clips are playing, and always restart it.

    Without this the tool corrupts the dataset it is being used to label.
    Measured Sep 13 2026 on the first real session: replaying six near miss
    clips through the speakers produced three new wake_hits at 0.474, 0.586 and
    0.687 plus three new wake_misses, all of them the microphone hearing our own
    playback. Those rows then sit in the capture directories indistinguishable
    from room audio, and the next person to retrain against them is training on
    a feedback loop.

    The second reason is worse than the first. A clip that transcribes as a
    command gets executed. Nothing about these recordings is chosen to be safe
    to say aloud near an assistant that sets timers and stores memories.

    Restart is in a finally, so Ctrl+C, an exception and a normal exit all put
    the room back the way they found it."""
    if not _service_active(unit):
        yield False
        return

    print(f"Stopping {unit} so playback cannot be heard as a wake word.")
    stopped = subprocess.run(["sudo", "systemctl", "stop", unit],
                             capture_output=True, text=True).returncode == 0
    if not stopped:
        print(f"  Could not stop {unit}. Stop it yourself before labelling, or "
              f"every clip you play may be captured as a fresh sample.")
    try:
        yield stopped
    finally:
        if stopped:
            print(f"\nRestarting {unit}.")
            subprocess.run(["sudo", "systemctl", "start", unit],
                           capture_output=True)


def _speaker_device():
    """Resolved by name, since ALSA card numbers shift between boots.

    Duplicated from tts.py rather than imported, because importing tts builds
    an ElevenLabs client at module scope and labelling should not need an API
    key. enroll.py already duplicates this for the same reason; if a fourth
    caller appears, move it somewhere shared instead of copying again."""
    try:
        with open("/proc/asound/cards") as f:
            for block in f.read().split("\n"):
                if SPEAKER_NAME_HINT in block:
                    return f"plughw:{block.strip().split()[0]},0"
    except OSError:
        pass
    return "default"


def cmd_status():
    for name, directory in SETS.items():
        clips = _clips(directory, 0.0)
        rows = _load(directory)
        done = sum(1 for _, n in clips if n in rows)
        print(f"{name:8} {len(clips):>4} clips, {done:>4} labelled")
        if not clips:
            continue
        counts = {}
        for _, n in clips:
            if n in rows:
                counts[rows[n]["label"]] = counts.get(rows[n]["label"], 0) + 1
        for label, count in sorted(counts.items()):
            print(f"           {label:<18} {count}")
        bands = {"<0.20": 0, "0.20-0.40": 0, ">=0.40": 0}
        for score, _ in clips:
            key = "<0.20" if score < 0.2 else ("0.20-0.40" if score < 0.4 else ">=0.40")
            bands[key] += 1
        print(f"           by score: {bands}")


def cmd_label(which, minimum, relabel):
    directory = SETS[which]
    clips = _clips(directory, minimum)
    if not clips:
        print(f"No clips in {directory} at or above {minimum}.")
        return

    rows = _load(directory)
    todo = [(s, n) for s, n in clips if relabel or n not in rows]
    if not todo:
        print(f"All {len(clips)} clips already labelled. Use --relabel to revisit.")
        return

    device = _speaker_device()
    question = ("She ANSWERED on this. Did you say the wake phrase?"
                if which == "hits" else
                "She did NOT answer. Did you say the wake phrase?")

    print(f"{len(todo)} to label, loudest first. Threshold is {WAKE_THRESHOLD}.")
    print("  y = yes, I said it    n = no    u = unclear    r = replay")
    print("  s = skip              q = quit\n")

    with _voice_service_paused():
        _label_loop(todo, directory, rows, device, question)

    print(f"\nLabels in {_manifest_path(directory)}")


def _label_loop(todo, directory, rows, device, question):
    for index, (score, name) in enumerate(todo, 1):
        path = os.path.join(directory, name)
        print("=" * 68)
        print(f"  [{index}/{len(todo)}]  score {score:.3f}   {name}")
        print(f"  {question}")

        play_path, gain_db, peak_db = _normalized_copy(path)
        if peak_db is not None:
            detail = f"  level: peak {peak_db:.1f} dBFS"
            if gain_db:
                detail += f", boosted {gain_db:.0f} dB for playback"
            if peak_db < -35:
                detail += "   (far below close speech, which peaks near -10)"
            print(detail)

        try:
            while True:
                _play(play_path, device)
                answer = input("  > ").strip().lower()
                if answer == "r":
                    continue
                break
        finally:
            if play_path != path:
                os.unlink(play_path)

        if answer == "q":
            break
        if answer == "s" or answer not in LABELS:
            print("  skipped\n")
            continue

        rows[name] = {"file": name, "score": f"{score:.3f}",
                      "label": LABELS[answer]}
        _save(directory, rows)
        print(f"  -> {LABELS[answer]}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="?", choices=sorted(SETS),
                        help="which capture set to label")
    parser.add_argument("--min", type=float, default=0.0,
                        help="only clips scoring at or above this")
    parser.add_argument("--relabel", action="store_true",
                        help="revisit clips already labelled")
    parser.add_argument("--status", action="store_true",
                        help="counts only, no playback")
    args = parser.parse_args()

    if args.status or not args.which:
        cmd_status()
        return
    cmd_label(args.which, args.min, args.relabel)


if __name__ == "__main__":
    main()
