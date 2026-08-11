"""Read only analysis of the timing_log table.

Reports per stage median and p90, and what fraction of perceived latency each
stage accounts for. Changes nothing and writes nothing back.

total_perceived_ms is the number that matters: speech end to first audio out
of the speaker. Everything else is decomposition, and the stage table is
ordered to make the largest contributor obvious rather than to match the
pipeline order.

Usage:
  python3 analyze_timing.py
  python3 analyze_timing.py --after 2026-08-11T09:00 --before 2026-08-11T10:00
  python3 analyze_timing.py --turn-type initial --label "morning block"
  python3 analyze_timing.py --include-actions

Action turns make a second Claude call and are excluded by default, since
they are structurally slower and smear the distribution of everything else.
"""

import argparse
import sqlite3
import statistics

from config import DB_PATH

MIN_ROWS_FOR_STATS = 10

# Ordered as the pipeline runs, so the printed table reads as a timeline.
STAGES = [
    ("speech_end_to_endpoint_ms", "Endpointing",
     "Speech end to record_command returning. Invisible to any measurement "
     "that starts after recording ends."),
    ("transcribe_ms", "Whisper",
     "whisper.cpp wall clock."),
    ("verify_ms", "Verification",
     "preprocess_wav, acoustic measures, and embed_utterance."),
    ("claude_ttft_ms", "Claude TTFT",
     "Request sent to first token. This gates first audio, not claude_total."),
    ("first_sentence_ms", "Sentence assembly",
     "First token to first complete sentence reaching the TTS queue. Nothing "
     "is spoken until a sentence boundary lands."),
    ("tts_first_audio_ms", "TTS to first audio",
     "ElevenLabs request to first chunk flushed to aplay."),
]

# Reported but not part of the critical path to first audio.
OFF_PATH = [
    ("claude_total_ms", "Claude total",
     "Full stream. Runs concurrently with TTS, so it does not add to "
     "perceived latency unless the response is short."),
    ("tts_ttfb_ms", "TTS network only",
     "Request to first chunk received. The gap to tts_first_audio is local."),
    ("action_ms", "Action branch",
     "Action dispatch plus the second Claude call. Action turns only."),
]


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def fetch_rows(after=None, before=None, turn_type=None, include_actions=False):
    where, params = [], []
    if after:
        where.append("created_at >= ?")
        params.append(after)
    if before:
        where.append("created_at <= ?")
        params.append(before)
    if turn_type:
        where.append("turn_type = ?")
        params.append(turn_type)
    if not include_actions:
        where.append("action_fired = 0")

    sql = "SELECT * FROM timing_log"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return rows


def header(title):
    print()
    print(title)
    print("=" * len(title))


def caveat(n):
    if n == 0:
        return "  (no data)"
    if n < MIN_ROWS_FOR_STATS:
        return f"  [n={n}, TOO FEW TO TRUST]"
    return ""


def report_totals(rows):
    header("1. PERCEIVED LATENCY  (speech end to first audio)")
    totals = [r["total_perceived_ms"] for r in rows if r["total_perceived_ms"] is not None]
    if not totals:
        print("No turn reached audio output. Nothing to report.")
        return None

    print(f"Turns with a complete measurement: {len(totals)} of {len(rows)}")
    print()
    print(f"  min     {min(totals):>8.0f} ms")
    print(f"  p50     {statistics.median(totals):>8.0f} ms")
    print(f"  p90     {percentile(totals, 0.90):>8.0f} ms")
    print(f"  max     {max(totals):>8.0f} ms")
    print(f"  mean    {statistics.mean(totals):>8.0f} ms{caveat(len(totals))}")
    return totals


def report_stages(rows, totals):
    header("2. STAGE BREAKDOWN  (critical path to first audio)")
    if not totals:
        print("No complete turns.")
        return

    median_total = statistics.median(totals)
    print(f"{'stage':<22} {'p50':>8} {'p90':>8} {'% of p50 total':>16}")

    accounted = 0.0
    for field, label, _ in STAGES:
        values = [r[field] for r in rows if r[field] is not None]
        if not values:
            print(f"{label:<22} {'-':>8} {'-':>8} {'-':>16}")
            continue
        p50 = statistics.median(values)
        p90 = percentile(values, 0.90)
        share = p50 / median_total * 100 if median_total else 0
        accounted += p50
        print(f"{label:<22} {p50:>8.0f} {p90:>8.0f} {share:>15.1f}%"
              f"{caveat(len(values))}")

    unaccounted = median_total - accounted
    print(f"{'unaccounted':<22} {unaccounted:>8.0f} {'-':>8} "
          f"{unaccounted / median_total * 100 if median_total else 0:>15.1f}%")
    print()
    print("Unaccounted is glue: process startup, wav writing, queue handoffs,")
    print("and the gap between stages. A large value means a stage is missing")
    print("from the instrumentation, not that the pipeline is idle.")

    print()
    print("Off the critical path:")
    print(f"{'stage':<22} {'p50':>8} {'p90':>8}")
    for field, label, _ in OFF_PATH:
        values = [r[field] for r in rows if r[field] is not None]
        if not values:
            print(f"{label:<22} {'-':>8} {'-':>8}")
            continue
        print(f"{label:<22} {statistics.median(values):>8.0f} "
              f"{percentile(values, 0.90):>8.0f}{caveat(len(values))}")


def report_by_turn_type(rows):
    header("3. BY TURN TYPE")
    types = sorted({r["turn_type"] for r in rows})
    if len(types) < 2:
        print(f"Only one turn type present ({types[0] if types else 'none'}).")
        return
    print(f"{'turn_type':<12} {'n':>5} {'p50 total':>11} {'p90 total':>11}")
    for turn_type in types:
        subset = [r["total_perceived_ms"] for r in rows
                  if r["turn_type"] == turn_type and r["total_perceived_ms"] is not None]
        if not subset:
            continue
        print(f"{turn_type:<12} {len(subset):>5} "
              f"{statistics.median(subset):>10.0f}m "
              f"{percentile(subset, 0.90):>10.0f}m{caveat(len(subset))}")


def report_by_model(rows):
    header("4. MODEL COMPARISON")

    models = sorted({r["model"] for r in rows if r["model"]})
    if not models:
        print("No turns carry a model. These are logged only for turns run "
              "after the A/B landed.")
        return
    if len(models) < 2:
        print(f"Only one model present ({models[0]}). Check MODEL_AB_TEST "
              "in config.py.")

    print("Claude TTFT is the number the A/B is about: it gates when Nova")
    print("starts talking. claude_total is reported for context but overlaps")
    print("with TTS, so it barely moves perceived latency.")
    print()
    print(f"{'model':<30} {'n':>4} {'ttft p50':>9} {'ttft p90':>9} "
          f"{'total p50':>10} {'perceived p50':>14}")

    ttft_by_model = {}
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        ttfts = [r["claude_ttft_ms"] for r in subset if r["claude_ttft_ms"] is not None]
        totals = [r["claude_total_ms"] for r in subset if r["claude_total_ms"] is not None]
        perceived = [r["total_perceived_ms"] for r in subset
                     if r["total_perceived_ms"] is not None]
        if not ttfts:
            print(f"{model:<30} {len(subset):>4} {'-':>9} {'-':>9} {'-':>10} {'-':>14}")
            continue
        ttft_by_model[model] = ttfts
        print(f"{model:<30} {len(ttfts):>4} "
              f"{statistics.median(ttfts):>9.0f} "
              f"{percentile(ttfts, 0.90):>9.0f} "
              f"{statistics.median(totals) if totals else 0:>10.0f} "
              f"{statistics.median(perceived) if perceived else 0:>14.0f}"
              f"{caveat(len(ttfts))}")

    if len(ttft_by_model) == 2:
        (m1, v1), (m2, v2) = ttft_by_model.items()
        d1, d2 = statistics.median(v1), statistics.median(v2)
        faster, slower = (m1, m2) if d1 < d2 else (m2, m1)
        delta = abs(d1 - d2)
        pct = delta / max(d1, d2) * 100
        print()
        print(f"{faster} is {delta:.0f}ms faster on median TTFT "
              f"({pct:.0f}% lower than {slower}).")

        # Spread check: a median gap smaller than the within model spread is
        # not a real difference, it is noise wearing a number.
        spread = max(percentile(v1, 0.90) - percentile(v1, 0.10),
                     percentile(v2, 0.90) - percentile(v2, 0.10))
        if delta < spread / 2:
            print(f"CAUTION: that gap is small next to the within model spread "
                  f"({spread:.0f}ms p10 to p90). Treat it as inconclusive "
                  f"until the sample grows.")


def _print_turn(r):
    label = r["model"] or "model n/a"
    print()
    print(f"  {r['total_perceived_ms']:.0f}ms total   [{label}]   {r['turn_type']}"
          f"{'   [action]' if r['action_fired'] else ''}   {r['created_at'][:19]}")
    parts = []
    for field, stage_label, _ in STAGES:
        if r[field] is not None:
            parts.append(f"{stage_label} {r[field]:.0f}")
    print(f"    {'  |  '.join(parts)}")
    print(f"    said: {r['transcript'] or '(none)'}")
    print(f"    nova: {r['response'] or '(none)'}")


def report_extremes(rows, count=5):
    scored = [r for r in rows if r["total_perceived_ms"] is not None]
    if not scored:
        header("5. SLOWEST AND FASTEST TURNS")
        print("No complete turns.")
        return

    ordered = sorted(scored, key=lambda r: r["total_perceived_ms"])

    header(f"5. {count} SLOWEST TURNS")
    for r in reversed(ordered[-count:]):
        _print_turn(r)

    header(f"6. {count} FASTEST TURNS")
    for r in ordered[:count]:
        _print_turn(r)


def report_response_length(rows):
    """Response length over time, so the prompt change is measurable rather
    than a matter of feel. Baseline before the change: median 76 words."""
    header("7. RESPONSE LENGTH")
    lengths = [r["response_words"] for r in rows if r["response_words"] is not None]
    if not lengths:
        print("No turns carry a word count. Logged only for turns after the "
              "response length work landed.")
        return
    print(f"Turns with a word count: {len(lengths)} of {len(rows)}")
    print(f"  median {statistics.median(lengths):>5.0f} words"
          f"   p90 {percentile(lengths, 0.90):>5.0f}"
          f"   max {max(lengths):>5}{caveat(len(lengths))}")
    print(f"  estimated speaking time at 150 wpm: "
          f"median {statistics.median(lengths) / 150 * 60:.0f}s, "
          f"max {max(lengths) / 150 * 60:.0f}s")
    print()
    print("Baseline before the prompt change: median 76 words, p90 117, max 120.")
    print("Offline A/B over 27 samples predicted median 56, p90 105, max 128.")


def report_cache(rows):
    header("8. PROMPT CACHE")
    measured = [r for r in rows if r["cache_read_tokens"] is not None]
    if not measured:
        print("No turns carry cache data.")
        return
    reads = [r["cache_read_tokens"] for r in measured]
    writes = [r["cache_creation_tokens"] for r in measured]
    hits = sum(1 for v in reads if v > 0)
    print(f"Turns with cache data: {len(measured)}")
    print(f"  cache hits:   {hits}/{len(measured)} ({100 * hits / len(measured):.0f}%)")
    print(f"  median read:  {statistics.median(reads):>6.0f} tokens")
    print(f"  median write: {statistics.median(writes):>6.0f} tokens")
    if hits == 0:
        print()
        print("ZERO HITS. Caching is not engaging, and the API reports this")
        print("as success. The usual cause is the cacheable prefix falling")
        print("below the model minimum (4096 tokens on Haiku 4.5), which")
        print("happens silently if the seed corpus shrinks.")

    # TTFT split by whether the turn hit the cache, which is the only direct
    # measurement of what caching is actually buying.
    hit = [r["claude_ttft_ms"] for r in measured
           if r["cache_read_tokens"] > 0 and r["claude_ttft_ms"] is not None]
    miss = [r["claude_ttft_ms"] for r in measured
            if r["cache_read_tokens"] == 0 and r["claude_ttft_ms"] is not None]
    if hit and miss:
        print()
        print(f"  TTFT on cache hit:  {statistics.median(hit):>6.0f} ms  (n={len(hit)})")
        print(f"  TTFT on cache miss: {statistics.median(miss):>6.0f} ms  (n={len(miss)})")
        print(f"  caching is worth {statistics.median(miss) - statistics.median(hit):.0f} ms per turn")


def report_quality_sample(rows, per_model=4):
    """Response text side by side, for judging answer quality by reading.

    Sampled evenly across the run rather than taken from one stretch, so a
    single odd conversation cannot stand in for a whole arm."""
    header("9. RESPONSE SAMPLE BY MODEL")
    models = sorted({r["model"] for r in rows if r["model"]})
    if not models:
        print("No turns carry a model.")
        return

    for model in models:
        subset = [r for r in rows if r["model"] == model and r["response"]]
        print()
        print(f"--- {model} ({len(subset)} turns with a response) ---")
        if not subset:
            print("  (none)")
            continue
        step = max(1, len(subset) // per_model)
        for r in subset[::step][:per_model]:
            print()
            print(f"  you:  {r['transcript'] or '(none)'}")
            print(f"  nova: {r['response']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze M.I.L.E.S. turn latency.")
    parser.add_argument("--after", metavar="ISO8601")
    parser.add_argument("--before", metavar="ISO8601")
    parser.add_argument("--turn-type", choices=["initial", "followup"])
    parser.add_argument("--label", metavar="TEXT",
                        help="name this block in the output")
    parser.add_argument("--include-actions", action="store_true",
                        help="include turns that fired an action (excluded by "
                             "default; they make a second Claude call)")
    args = parser.parse_args()

    rows = fetch_rows(args.after, args.before, args.turn_type, args.include_actions)

    print()
    print("M.I.L.E.S. TURN LATENCY ANALYSIS")
    if args.label:
        print(f"Block: {args.label}")
    print(f"Database: {DB_PATH}")
    if args.after or args.before:
        print(f"Window: {args.after or 'start'} to {args.before or 'now'}")
    if not args.include_actions:
        print("Action turns excluded (--include-actions to see them).")

    if not rows:
        print()
        print("No turns match. Nothing to analyze.")
        return

    totals = report_totals(rows)
    report_stages(rows, totals)
    report_by_turn_type(rows)
    report_by_model(rows)
    report_extremes(rows)
    report_response_length(rows)
    report_cache(rows)
    report_quality_sample(rows)

    print()
    if len(rows) < MIN_ROWS_FOR_STATS:
        print(f"SAMPLE SIZE WARNING: {len(rows)} turns. Directional at best.")
    print()


if __name__ == "__main__":
    main()
