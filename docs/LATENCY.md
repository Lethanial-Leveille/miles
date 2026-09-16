# Latency: measured, not estimated

> **Precedence rule.** This document describes the repo. The repo is the
> authority. If anything here conflicts with source, **the source wins**, and
> whoever finds the conflict fixes this file in the same session.
>
> Measurements taken Aug 10 to Aug 13 2026 and read out of `timing_log`. Re
> run `scripts/analyze_timing.py` before quoting any of it.
>
> **Live config values are declared in [CLAUDE.md](../CLAUDE.md), not here.**
> This file carries the reasoning and the measurements behind them. A number
> quoted inside a measurement is what was recorded on the date given and is
> left exactly as measured, so it may differ from what is live now. The config
> table in CLAUDE.md is the one that tracks the present.

Up: [CLAUDE.md](../CLAUDE.md) · [SESSION_START.md](SESSION_START.md) ·
[BACKEND_TODO.md](BACKEND_TODO.md) · [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md) ·
[LATENCY.md](LATENCY.md) · [VOICE_OUTPUT.md](VOICE_OUTPUT.md) ·
[BRAIN.md](BRAIN.md) · [INFRASTRUCTURE.md](INFRASTRUCTURE.md) ·
[INCIDENTS.md](INCIDENTS.md)

---

**Do not quote a latency figure that was not read out of `timing_log`.** That
rule is the reason this file exists as a single place rather than as numbers
scattered through prose.

## The turn budget, measured Sep 16 2026

From `timing_log`, voice turns since Sep 13 (eleven_v3, calendar tools, 100
turns). **The app is not timed at all**; only voice turns reach `timing_log`.

| Stage | All turns | No tool | With a tool |
|---|---|---|---|
| endpoint | 1200ms | 1200ms | 1200ms |
| transcribe | 287ms | 316ms | 266ms |
| verify | 264ms | 289ms | 246ms |
| claude_ttft | 1179ms | 1075ms | 1405ms |
| first_sentence | 230ms | 217ms | 258ms |
| tts_ttfb | 606ms | 567ms | 647ms |
| tool | | | 606ms |
| second_ttft | | | 631ms |
| action, the whole tool turn | | | 8022ms |
| **perceived** | **4251ms** (n=92) | **3857ms** (n=46) | **5258ms** (n=45) |

Tool turns were 53 of 100. The Aug 12 table below is kept as history.

Changes made against this baseline, one at a time so each shows up on its own:

- Sep 16: verification starts with the speculative transcript
  ([AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#verification-starts-with-the-speculation-sep-16-2026)).
  Expected to remove most of `verify_ms`; not yet measured live.
- Sep 16: a pre rendered bridge line when a slow tool starts on a voice turn
  ([VOICE_OUTPUT.md](VOICE_OUTPUT.md#bridges-on-slow-tool-turns-sep-16-2026)).
  **Timed separately, as `bridge_ms`** (migration 25). `total_perceived_ms`
  keeps meaning the wait for the answer, so it stays comparable with every row
  above; the first sound on a tool turn is the smaller of the two. Expected
  around 2.9s; not yet measured live.
- Sep 16: speculative transcription starts after 210ms of silence rather than
  450ms
  ([AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#speculative-transcription)).
  Expected to take most of the 287ms `transcribe_ms`; not yet measured live.

Early reading, five turns only, not a figure to quote: `verify_ms` 3 to 22ms
where the speculation held.

## The turn budget, measured Aug 12 2026

| Stage | Median |
|---|---|
| endpoint | 880ms |
| transcribe | 1166ms |
| verify | 193ms |
| claude_ttft | 1253ms |
| first_sentence | 288ms |
| tts_ttfb | 349ms |
| genuinely unaccounted | 32ms |
| **perceived** | **4325ms** |

**The tts_ttfb row above is flash_v2.** Nova moved to eleven_v3 on Sep 13 2026. Over the
first 12 live turns on v3 the median time to first byte was 647ms, nearly double. The
rest of this table predates the switch; re measure the whole budget before quoting a
new perceived total.

Re-read on the Aug 12 window, after `SILENCE_LIMIT` went to 1.2 (n=44, Haiku,
non action): endpoint 1200, transcribe 1089, verify 190, claude_ttft 1350,
first_sentence 230, tts 382, perceived p50 4324. **The sum of the stage medians
exceeds the total by 117ms and that is arithmetic, not a missing stage**: each
stage's median comes from a different turn, and medians do not add. Do not go
looking for overlap to explain a residual of this size in either direction.

**Local intent turns carried no perceived latency until Aug 13 2026.** They
never reach `tts.speak`, so nothing closed out the measurement and every one of
them logged a null total. They are also the fastest turns the system has, so
the reported median described the Claude path rather than the room, and it did
so while local intent was answering a fifth of all turns. `phrasebank.play` now
calls `timing.note_local_audio`, migration 22 adds `local_intent` to
`timing_log`, and `scripts/analyze_timing.py` reports the two paths separately in
section 1 and gives local turns their own stage table in section 2b.

A number that excludes the fast cases is worse than no number, because it looks
like a measurement. Any turn that produces audio has to close out
`total_perceived_ms`, whoever produced it.

**There is no missing time.** An earlier reading of this table left
`first_sentence_ms` out of the sum and reported a 325ms hole; counting it as its
own stage closes the residual to 32ms. `scripts/profile_turn.py` confirmed the other
suspects are noise: `_write_wav`, `archive_recording` and the entire prompt
assembly including hybrid memory search total 3.1ms.

`first_sentence_ms` at 288ms is real optimizable time: Claude's first token has
arrived, but StreamRouter is still buffering `LOOKAHEAD_CHARS = 50` and waiting
for a sentence boundary before anything can reach TTS.

> **Correction, Sep 13 2026.** `first_sentence_ms` is attributed above to
> StreamRouter buffering `LOOKAHEAD_CHARS = 50` before a sentence can flush.
> **That mechanism no longer exists.** Native tool use removed the lookahead
> entirely, and `stream_router.py` now flushes a sentence as soon as it is
> complete. The 288ms figure predates that change, so the stage is unmeasured
> at present rather than known to be optimizable. It also carries a
> second, disagreeing figure of **608ms** in
> [BACKEND_TODO.md](BACKEND_TODO.md), and neither number was taken after
> the lookahead was removed. See [BRAIN.md](BRAIN.md#streaming) for the
> mechanism and BACKEND_TODO for what to measure, in what order, before
> acting on either figure.

## History

From `timing_log`, 48 rows, Aug 10 to Aug 11 2026. Segmented by endpoint delay,
which reveals which `SILENCE_LIMIT` was live:

| Era | n | Median perceived | Median endpoint |
|---|---|---|---|
| Before `SILENCE_LIMIT = 0.9` | 20 | 8088ms | 2960ms |
| After | 28 | **4938ms** | 920ms |

Haiku, post change, non action turns: n=19, median perceived 4977ms, median
time to first token 1318ms.

Perceived latency is measured from the moment the user stopped talking to the
moment the first audio chunk is flushed to aplay. ALSA adds buffer delay after
that point, so the true figure at the speaker is slightly higher and
consistently so.

Do not quote a latency figure in this file that was not read out of `timing_log`.

## Related

- Endpointing and the speculation coupling that decides how much of transcription is hidden: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#speculative-transcription)
- The local intent path, which is the fastest class of turn and was invisible in these medians until Aug 13 2026: [BRAIN.md](BRAIN.md#local-intent)
- Time to first token and the model A/B: [BRAIN.md](BRAIN.md#model)
- Remaining latency ideas ranked by payoff: [BACKEND_TODO.md](BACKEND_TODO.md)

## Calendar tools, Sep 13 2026

Calendar questions were the slowest turns of the night: 7.8s and 8.4s perceived,
with the tool alone at 2.6s and 2.9s. Timed live, read only, on his real
calendars:

| Stage | Before |
|---|---|
| listing his calendars | 497ms, on every call |
| nine calendars, one after another | 1621ms, the slowest single one 279ms |

Changed: every calendar is fetched at the same time, each on its own client, and
the calendar list is remembered for five minutes.

| Tool, same machine, same calendars | After |
|---|---|
| upcoming events, list not yet remembered | 907ms |
| upcoming events, list remembered | 482ms |
| schedule conflicts | 446ms |

The same night, turn by turn, the rest of the budget: eleven_v3 costs about
350ms more to first audio than flash_v2 did, and the pauses between sentences
measured a median 0ms after synthesis began running ahead of playback.
