"""Compare whisper models or settings against real archived commands.

The point of the recording archive. Every audio decision so far has had to be
validated against enrollment recordings, because build/command.wav is
overwritten every turn and nothing kept a copy. That was workable for the
audio context change and it is not workable for a model swap, because the
cases that matter are the hard ones: low SNR, far field, the turns that
already transcribe badly.

Speed is the easy half. The half that decides is whether the transcript
changes, and on which recordings, so differences are printed in full next to
the SNR of the audio that produced them.

Usage:
  python3 compare_whisper.py --model ../whisper.cpp/models/ggml-tiny.en.bin
  python3 compare_whisper.py --audio-ctx 750
  python3 compare_whisper.py --model <path> --limit 40 --worst-snr

Nothing is changed. This only reads and reports.
"""

import argparse
import os
import sqlite3
import statistics
import subprocess
import time

from config import DB_PATH, WHISPER_CLI, WHISPER_MODEL, WHISPER_AUDIO_CTX


def fetch_recordings(limit, worst_snr):
    """Archived recordings joined to what the live pipeline made of them."""
    order = "snr_db ASC" if worst_snr else "id DESC"
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT recording_path, transcript, snr_db, embedded_duration_seconds "
        "FROM verification_log "
        "WHERE recording_path IS NOT NULL AND snr_db IS NOT NULL "
        f"ORDER BY {order} LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [r for r in rows if os.path.exists(r["recording_path"])]


def transcribe(model, audio_ctx, path, reps=2):
    """Best of `reps` wall clock, since a single run picks up scheduler noise."""
    cmd = [WHISPER_CLI, "-m", model, "-f", path, "-bs", "1", "-bo", "1",
           "--no-prints", "--no-timestamps", "-ac", str(audio_ctx)]
    best, text = None, ""
    for _ in range(reps):
        started = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = (time.monotonic() - started) * 1000
        best = elapsed if best is None else min(best, elapsed)
        text = " ".join(result.stdout.split())
    return text, best


def normalize(text):
    """Compare on words, not punctuation. Whisper's comma and period placement
    varies run to run and never changes what an action tag parses to."""
    return " ".join(
        w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")
    )


def main():
    parser = argparse.ArgumentParser(description="Compare whisper settings on real commands.")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        help="candidate model (default: the configured one)")
    parser.add_argument("--audio-ctx", type=int, default=WHISPER_AUDIO_CTX,
                        help=f"candidate audio context (default: {WHISPER_AUDIO_CTX})")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--worst-snr", action="store_true",
                        help="test the lowest SNR recordings rather than the "
                             "most recent, which is where a weaker model breaks")
    args = parser.parse_args()

    rows = fetch_recordings(args.limit, args.worst_snr)
    print()
    print("WHISPER COMPARISON ON ARCHIVED COMMANDS")
    print(f"  baseline:  {os.path.basename(WHISPER_MODEL)} at -ac {WHISPER_AUDIO_CTX}")
    print(f"  candidate: {os.path.basename(args.model)} at -ac {args.audio_ctx}")
    print(f"  recordings: {len(rows)}"
          + ("  (lowest SNR first)" if args.worst_snr else "  (most recent first)"))

    if not rows:
        print()
        print("No archived recordings yet. They accumulate as you use Nova;")
        print("come back once there are a few dozen, ideally including some")
        print("far field turns.")
        return

    same = 0
    base_times, cand_times = [], []
    differences = []

    for row in rows:
        base_text, base_ms = transcribe(WHISPER_MODEL, WHISPER_AUDIO_CTX, row["recording_path"])
        cand_text, cand_ms = transcribe(args.model, args.audio_ctx, row["recording_path"])
        base_times.append(base_ms)
        cand_times.append(cand_ms)
        if normalize(base_text) == normalize(cand_text):
            same += 1
        else:
            differences.append((row, base_text, cand_text))

    print()
    print(f"  baseline  median {statistics.median(base_times):>7.0f} ms")
    print(f"  candidate median {statistics.median(cand_times):>7.0f} ms")
    print(f"  saving           {statistics.median(base_times) - statistics.median(cand_times):>7.0f} ms per turn")
    print()
    print(f"  transcripts identical: {same}/{len(rows)} "
          f"({100 * same / len(rows):.0f}%)")

    if differences:
        print()
        print("  DIFFERENCES (the half that decides):")
        for row, base_text, cand_text in differences:
            print()
            print(f"    snr {row['snr_db']:.0f} dB, {row['embedded_duration_seconds']:.1f}s voiced")
            print(f"      baseline : {base_text}")
            print(f"      candidate: {cand_text}")
        print()
        print("  Judge these by whether the meaning survives. A dropped comma")
        print("  costs nothing; a dropped 'not' costs the whole turn.")
    print()


if __name__ == "__main__":
    main()
