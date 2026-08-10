"""Read only analysis of the verification_log table.

Phase 1 diagnostic tool. Reports the score distribution, the relationship
between embedded audio duration and similarity, and the gap between what was
recorded and what actually reached the encoder. Changes no verification
logic and writes nothing back to the database.

Small sample sizes are called out inline rather than silently reported,
because a mean over four rows reads exactly like a mean over four hundred
once it is printed.

Usage:
  python3 analyze_verification.py
  python3 analyze_verification.py --after 2026-08-10T14:00 --before 2026-08-10T14:20
  python3 analyze_verification.py --after 2026-08-10T14:00 --label "far, projecting"
  python3 analyze_verification.py --include-degenerate

Timestamps are ISO 8601 and compared as strings, which sorts correctly for
this format. A date alone works as a prefix: --after 2026-08-10 covers the
whole day.
"""

import argparse
import math
import sqlite3
import statistics

from config import DB_PATH, VERIFY_THRESHOLD

# Below this many rows a summary statistic is noise dressed up as a number,
# so it gets printed with a warning rather than on its own.
MIN_ROWS_FOR_STATS = 10
MIN_ROWS_FOR_CORRELATION = 20

# Below this many seconds of voiced audio after trimming, the embedding
# describes silence rather than a speaker.
DEGENERATE_SECONDS = 0.5

DURATION_BUCKETS = [
    ("under 1s", 0.0, 1.0),
    ("1 to 2s",  1.0, 2.0),
    ("2 to 3s",  2.0, 3.0),
    ("3s and up", 3.0, float("inf")),
]


def percentile(values, p):
    """Linear interpolation between closest ranks. p is a fraction, so the
    10th percentile is p=0.10."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def pearson(xs, ys):
    """Pearson correlation coefficient. Returns None when undefined, which
    happens when either variable has zero variance."""
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    if denominator == 0:
        return None
    return numerator / denominator


def fetch_rows(after=None, before=None, include_degenerate=False):
    """Rows in the window, newest last. Degenerate rows (no voiced audio after
    trim) are excluded by default: their similarity score is an artifact of
    embedding silence and including them distorts every statistic here,
    most severely the duration correlation."""
    where = []
    params = []
    if after:
        where.append("created_at >= ?")
        params.append(after)
    if before:
        where.append("created_at <= ?")
        params.append(before)
    if not include_degenerate:
        # outcome covers rows logged after the guard landed; the duration
        # check catches the ones logged before it existed.
        where.append("outcome = 'scored'")
        where.append(f"embedded_duration_seconds >= {DEGENERATE_SECONDS}")

    sql = ("SELECT created_at, similarity, accepted, threshold_used, transcript, "
           "duration_seconds, embedded_duration_seconds, wake_confidence, turn_type, "
           "outcome, rms_dbfs, snr_db, spectral_tilt FROM verification_log")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def count_excluded(after=None, before=None):
    """How many rows in the window the default filter drops, so an empty or
    thin result never looks like 'you made no attempts'."""
    where = ["(outcome != 'scored' OR embedded_duration_seconds < ?)"]
    params = [DEGENERATE_SECONDS]
    if after:
        where.append("created_at >= ?")
        params.append(after)
    if before:
        where.append("created_at <= ?")
        params.append(before)

    conn = sqlite3.connect(DB_PATH)
    n = conn.execute(
        "SELECT COUNT(*) FROM verification_log WHERE " + " AND ".join(where), params
    ).fetchone()[0]
    conn.close()
    return n


def header(title):
    print()
    print(title)
    print("=" * len(title))


def caveat(n, minimum):
    """Inline reliability marker so every number carries its own sample size."""
    if n == 0:
        return "  (no data)"
    if n < minimum:
        return f"  [n={n}, TOO FEW TO TRUST]"
    return f"  [n={n}]"


# ── 1. Counts ──

def report_counts(rows):
    header("1. ATTEMPT COUNTS")
    total = len(rows)
    accepted = sum(1 for r in rows if r["accepted"])
    print(f"Total attempts: {total}")
    if total == 0:
        return
    print(f"Accepted: {accepted}    Rejected: {total - accepted}"
          f"    Reject rate: {(total - accepted) / total:.1%}")
    print(f"Time span: {rows[0]['created_at'][:19]} to {rows[-1]['created_at'][:19]}")

    thresholds = sorted({r["threshold_used"] for r in rows})
    if len(thresholds) > 1:
        print(f"WARNING: threshold changed during collection: {thresholds}")
        print("Accept and reject decisions across this data are not comparable.")
    else:
        print(f"Threshold in effect: {thresholds[0]}")

    print()
    print(f"{'turn_type':<12} {'accepted':>9} {'rejected':>9} {'total':>7} {'reject rate':>12}")
    for turn_type in sorted({r["turn_type"] for r in rows}):
        subset = [r for r in rows if r["turn_type"] == turn_type]
        acc = sum(1 for r in subset if r["accepted"])
        rej = len(subset) - acc
        print(f"{turn_type:<12} {acc:>9} {rej:>9} {len(subset):>7} {rej / len(subset):>11.1%}")


# ── 2. Score distribution ──

def report_distribution(rows):
    header("2. SIMILARITY DISTRIBUTION BY TURN TYPE")
    if not rows:
        print("No data.")
        return

    print(f"{'turn_type':<12} {'min':>7} {'p10':>7} {'median':>7} {'mean':>7} "
          f"{'p90':>7} {'max':>7} {'stdev':>7}")
    groups = [("ALL", rows)]
    groups += [(t, [r for r in rows if r["turn_type"] == t])
               for t in sorted({r["turn_type"] for r in rows})]

    for label, subset in groups:
        if not subset:
            continue
        scores = [r["similarity"] for r in subset]
        stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        print(f"{label:<12} {min(scores):>7.3f} {percentile(scores, 0.10):>7.3f} "
              f"{statistics.median(scores):>7.3f} {statistics.mean(scores):>7.3f} "
              f"{percentile(scores, 0.90):>7.3f} {max(scores):>7.3f} {stdev:>7.3f}"
              f"{caveat(len(subset), MIN_ROWS_FOR_STATS)}")

    print()
    print("Margin above threshold matters as much as the raw score. A genuine")
    print("speaker clustering just barely above the line is one bad clip away")
    print("from a rejection.")
    for label, subset in groups:
        if not subset:
            continue
        margins = [r["similarity"] - r["threshold_used"] for r in subset]
        near = sum(1 for m in margins if 0 <= m < 0.05)
        print(f"  {label:<10} mean margin {statistics.mean(margins):+.3f}, "
              f"{near} attempt(s) accepted by less than 0.05")


# ── 3. Duration vs similarity ──

def report_duration_buckets(rows):
    header("3. EMBEDDED DURATION VS SIMILARITY  (the decisive table)")
    if not rows:
        print("No data.")
        return

    print(f"{'bucket':<12} {'n':>4} {'mean sim':>9} {'min sim':>9} {'max sim':>9} "
          f"{'accept rate':>12}")
    for label, low, high in DURATION_BUCKETS:
        subset = [r for r in rows if low <= r["embedded_duration_seconds"] < high]
        if not subset:
            print(f"{label:<12} {0:>4} {'-':>9} {'-':>9} {'-':>9} {'-':>12}")
            continue
        scores = [r["similarity"] for r in subset]
        acc = sum(1 for r in subset if r["accepted"]) / len(subset)
        print(f"{label:<12} {len(subset):>4} {statistics.mean(scores):>9.3f} "
              f"{min(scores):>9.3f} {max(scores):>9.3f} {acc:>11.1%}"
              f"{caveat(len(subset), MIN_ROWS_FOR_STATS)}")

    durations = [r["embedded_duration_seconds"] for r in rows]
    scores = [r["similarity"] for r in rows]
    r = pearson(durations, scores)
    print()
    if r is None:
        print("Correlation undefined (no variance in one variable).")
    else:
        print(f"Pearson r (embedded duration vs similarity): {r:+.3f}"
              f"{caveat(len(rows), MIN_ROWS_FOR_CORRELATION)}")
        print("Positive r means longer audio scores higher, which is the")
        print("signature of the short utterance hypothesis.")

    below_three = sum(1 for d in durations if d < 3.0)
    print()
    print(f"Attempts embedding under 3s of voiced audio: {below_three}/{len(rows)} "
          f"({below_three / len(rows):.1%})")
    print("Resemblyzer embeddings are unstable below roughly three seconds.")


# ── 4. Trim gap ──

def report_trim_gap(rows):
    header("4. RECORDED VS EMBEDDED DURATION  (what the trim step discards)")
    if not rows:
        print("No data.")
        return

    gaps = [r["duration_seconds"] - r["embedded_duration_seconds"] for r in rows]
    ratios = [r["embedded_duration_seconds"] / r["duration_seconds"]
              for r in rows if r["duration_seconds"] > 0]

    print(f"Mean recorded:  {statistics.mean([r['duration_seconds'] for r in rows]):.2f}s")
    print(f"Mean embedded:  {statistics.mean([r['embedded_duration_seconds'] for r in rows]):.2f}s")
    print(f"Mean discarded: {statistics.mean(gaps):.2f}s"
          f"  ({1 - statistics.mean(ratios):.1%} of recorded audio on average)")

    print()
    print("Worst cases, most audio discarded:")
    print(f"{'recorded':>9} {'embedded':>9} {'discarded':>10} {'kept':>7}  transcript")
    worst = sorted(rows, key=lambda r: r["duration_seconds"] - r["embedded_duration_seconds"],
                   reverse=True)[:5]
    for r in worst:
        gap = r["duration_seconds"] - r["embedded_duration_seconds"]
        kept = r["embedded_duration_seconds"] / r["duration_seconds"] if r["duration_seconds"] else 0
        text = (r["transcript"] or "")[:44]
        print(f"{r['duration_seconds']:>8.2f}s {r['embedded_duration_seconds']:>8.2f}s "
              f"{gap:>9.2f}s {kept:>6.1%}  {text}")


# ── 5. Worst scoring attempts ──

def report_lowest(rows, count=10):
    header(f"5. {count} LOWEST SCORING ATTEMPTS")
    if not rows:
        print("No data.")
        return

    for r in sorted(rows, key=lambda r: r["similarity"])[:count]:
        verdict = "ACCEPT" if r["accepted"] else "REJECT"
        wake = f"{r['wake_confidence']:.2f}" if r["wake_confidence"] is not None else "n/a"
        print()
        print(f"  [{verdict}] similarity {r['similarity']:.3f}   "
              f"threshold {r['threshold_used']:.2f}   {r['turn_type']}")
        print(f"    recorded {r['duration_seconds']:.2f}s -> "
              f"embedded {r['embedded_duration_seconds']:.2f}s   "
              f"wake confidence {wake}   {r['created_at'][:19]}")
        print(f"    said: {r['transcript'] or '(none)'}")


# ── 6. Wake confidence ──

def report_wake_confidence(rows):
    header("6. WAKE CONFIDENCE VS SIMILARITY")
    paired = [r for r in rows if r["wake_confidence"] is not None]
    if len(paired) < 2:
        print(f"Only {len(paired)} attempt(s) carry a wake confidence value. "
              "Nothing to say.")
        print("Follow up turns have no wake word by definition, so this only "
              "ever covers initial commands.")
        return

    confidences = [r["wake_confidence"] for r in paired]
    scores = [r["similarity"] for r in paired]
    print(f"Attempts with a wake confidence: {len(paired)}")
    print(f"Wake confidence range: {min(confidences):.3f} to {max(confidences):.3f}, "
          f"mean {statistics.mean(confidences):.3f}")

    r = pearson(confidences, scores)
    if r is None:
        print("Correlation undefined (no variance in one variable).")
    elif len(paired) < MIN_ROWS_FOR_CORRELATION:
        print(f"Pearson r: {r:+.3f}{caveat(len(paired), MIN_ROWS_FOR_CORRELATION)}")
        print("Do not read anything into this yet.")
    else:
        print(f"Pearson r (wake confidence vs similarity): {r:+.3f}  [n={len(paired)}]")
        print("A positive r would suggest both degrade together, which points at")
        print("a shared upstream cause such as distance from the mic.")


# ── 7. Acoustic measures ──

ACOUSTIC_FIELDS = [
    ("snr_db", "SNR (dB)",
     "How far speech sits above the room. Simulation puts embedding cosine at "
     "0.98 for 25dB and 0.68 for 10dB, so this should dominate."),
    ("rms_dbfs", "Level (dBFS)",
     "Absolute loudness at the mic. Confounded: near and quiet looks like far "
     "and loud."),
    ("spectral_tilt", "Tilt (dB)",
     "Energy above 1kHz over below. Rises with vocal effort, barely moves with "
     "distance, so it isolates projecting from proximity."),
]


def report_acoustics(rows):
    header("7. ACOUSTIC MEASURES VS SIMILARITY")

    measured = [r for r in rows if r["snr_db"] is not None]
    if not measured:
        print("No attempts carry acoustic measures yet. These are logged only "
              "for attempts recorded after the acoustic logging landed.")
        return

    print(f"Attempts with acoustic measures: {len(measured)} of {len(rows)}")
    print()
    print(f"{'measure':<14} {'min':>8} {'mean':>8} {'max':>8} {'r vs similarity':>17}")
    for field, label, _ in ACOUSTIC_FIELDS:
        values = [r[field] for r in measured if r[field] is not None]
        if len(values) < 2:
            print(f"{label:<14} {'-':>8} {'-':>8} {'-':>8} {'insufficient':>17}")
            continue
        scores = [r["similarity"] for r in measured if r[field] is not None]
        r = pearson(values, scores)
        r_text = f"{r:+.3f}" if r is not None else "undefined"
        print(f"{label:<14} {min(values):>8.1f} {statistics.mean(values):>8.1f} "
              f"{max(values):>8.1f} {r_text:>17}"
              f"{caveat(len(values), MIN_ROWS_FOR_CORRELATION)}")

    print()
    for _, label, meaning in ACOUSTIC_FIELDS:
        print(f"  {label}: {meaning}")

    # SNR buckets, since the relationship is expected to be steeply nonlinear
    # at the low end rather than a clean straight line.
    snr_values = [r for r in measured if r["snr_db"] is not None]
    if snr_values:
        print()
        print("Similarity by SNR band:")
        print(f"{'band':<12} {'n':>4} {'mean sim':>9} {'accept rate':>12}")
        for label, low, high in [("under 10dB", 0, 10), ("10 to 15dB", 10, 15),
                                 ("15 to 20dB", 15, 20), ("20 to 25dB", 20, 25),
                                 ("25dB and up", 25, float("inf"))]:
            subset = [r for r in snr_values if low <= r["snr_db"] < high]
            if not subset:
                print(f"{label:<12} {0:>4} {'-':>9} {'-':>12}")
                continue
            sims = [r["similarity"] for r in subset]
            acc = sum(1 for r in subset if r["accepted"]) / len(subset)
            print(f"{label:<12} {len(subset):>4} {statistics.mean(sims):>9.3f} "
                  f"{acc:>11.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the M.I.L.E.S. speaker verification log.")
    parser.add_argument("--after", metavar="ISO8601",
                        help="only attempts at or after this timestamp, "
                             "e.g. 2026-08-10T14:00")
    parser.add_argument("--before", metavar="ISO8601",
                        help="only attempts at or before this timestamp")
    parser.add_argument("--label", metavar="TEXT",
                        help="name this block in the output, e.g. 'far, projecting'")
    parser.add_argument("--include-degenerate", action="store_true",
                        help="keep attempts with no voiced audio after trimming "
                             "(excluded by default)")
    args = parser.parse_args()

    rows = fetch_rows(args.after, args.before, args.include_degenerate)

    print()
    print("M.I.L.E.S. VERIFICATION LOG ANALYSIS")
    if args.label:
        print(f"Block: {args.label}")
    print(f"Database: {DB_PATH}")
    print(f"Configured threshold: {VERIFY_THRESHOLD}")
    if args.after or args.before:
        print(f"Window: {args.after or 'start'} to {args.before or 'now'}")

    if not args.include_degenerate:
        excluded = count_excluded(args.after, args.before)
        if excluded:
            print(f"Excluded {excluded} degenerate attempt(s) with under "
                  f"{DEGENERATE_SECONDS}s of voiced audio. "
                  "Rerun with --include-degenerate to see them.")

    if not rows:
        print()
        print("No attempts match. Nothing to analyze.")
        return

    report_counts(rows)
    report_distribution(rows)
    report_duration_buckets(rows)
    report_trim_gap(rows)
    report_lowest(rows)
    report_wake_confidence(rows)
    report_acoustics(rows)

    print()
    if len(rows) < MIN_ROWS_FOR_CORRELATION:
        print(f"SAMPLE SIZE WARNING: {len(rows)} attempts total. Treat every number "
              "above as directional at best.")
    print()


if __name__ == "__main__":
    main()
