"""Where does the unmeasured time in a turn go?

`timing_log` covers endpoint, transcribe, verify, claude_ttft and tts_ttfb. Those
sum to a median 325ms less than `total_perceived_ms` per turn, and the gap is
larger than several things worth optimizing, so it is worth locating before
anything else is tuned.

The gap is not mysterious, it is just outside every stopwatch. Three regions
have no instrumentation at all:

  * `_write_wav`, which runs after `note_speech_end` inside record_command
  * `archive_recording`, a copy plus a prune that walks up to ARCHIVE_MAX_FILES
  * everything in brain.py before `claude_start`, which is the whole prompt
    assembly including the hybrid memory search

This times them directly against an archived recording rather than waiting for
live turns, so it costs no API calls and needs no service restart.

    python3 scripts/profile_turn.py
    python3 scripts/profile_turn.py --runs 5
"""

import argparse
import glob
import os
import sys
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from config import ARCHIVE_DIR, TEMP_WAV  # noqa: E402


def _write_wav_equivalent(frames, rate):
    with wave.open(TEMP_WAV + ".profile", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)


def _archive_equivalent():
    """The copy plus the prune scan, without deleting anything.

    The prune is the part worth measuring: it lists and sorts the whole archive
    on every single turn, and the archive holds up to ARCHIVE_MAX_FILES."""
    import shutil
    dest = os.path.join(ARCHIVE_DIR, ".profile_probe.wav")
    shutil.copy2(TEMP_WAV + ".profile", dest)
    existing = sorted(f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".wav"))
    os.remove(dest)
    return len(existing)


def _timed(label, fn, results):
    start = time.monotonic()
    out = fn()
    results.append((label, (time.monotonic() - start) * 1000.0))
    return out


def profile(sample_wav, runs):
    import brain
    import prompts
    from database import (get_seed_memories, get_episodic_memories,
                          get_recent_messages, memory_manifest)

    with wave.open(sample_wav) as w:
        frames = w.readframes(w.getnframes())
        rate, seconds = w.getframerate(), w.getnframes() / w.getframerate()

    print(f"sample: {os.path.basename(sample_wav)}  ({seconds:.1f}s)\n")

    totals = {}
    for _ in range(runs):
        results = []

        # audio.py cannot be imported while miles-voice holds the mic lock, so
        # these two mirror what it does rather than calling it. Same syscalls on
        # the same files, which is what is being measured.
        _timed("_write_wav (equivalent)",
               lambda: _write_wav_equivalent(frames, rate), results)
        _timed("archive_recording (equivalent)",
               lambda: _archive_equivalent(), results)

        # ── brain.py, everything before claude_start ──
        seed = _timed("get_seed_memories", get_seed_memories, results)
        epi  = _timed("get_episodic_memories", get_episodic_memories, results)
        man  = _timed("memory_manifest", memory_manifest, results)
        _timed("build_enhanced_prompt",
               lambda: prompts.build_enhanced_prompt(seed, epi, "voice", man,
                                                     tier="hokage"), results)
        recent = _timed("get_recent_messages",
                        lambda: get_recent_messages(20), results)
        _timed("_trim_history", lambda: brain._trim_history(recent), results)
        # The hybrid memory search: keyword plus semantic, fused by rank.
        _timed("_with_recalled",
               lambda: brain._with_recalled(brain._trim_history(recent),
                                            "what is the weather like today"),
               results)

        for label, ms in results:
            totals.setdefault(label, []).append(ms)

    print(f"{'stage':26} {'median ms':>10}")
    print("-" * 38)
    grand = 0.0
    for label, values in totals.items():
        values.sort()
        median = values[len(values) // 2]
        grand += median
        print(f"{label:26} {median:10.1f}")
    print("-" * 38)
    print(f"{'TOTAL UNMEASURED':26} {grand:10.1f}")
    print(f"\nMeasured residual in timing_log is a median 325ms per turn.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--wav")
    args = ap.parse_args()

    sample = args.wav
    if not sample:
        found = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.wav")))
        if not found:
            print("No archived recordings to profile against.")
            raise SystemExit(1)
        sample = found[-1]

    profile(sample, args.runs)
