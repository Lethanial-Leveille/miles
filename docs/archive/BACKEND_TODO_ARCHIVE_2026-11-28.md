# Backend backlog archive, Nov 28 2026

Finished items moved out of [BACKEND_TODO.md](../BACKEND_TODO.md) on Nov 28 2026, text unchanged.

## Shipped and working

All live in production as of Aug 10 2026. All mic independent unless noted.

- `verification_log` table with per attempt logging: similarity, accept
  decision, threshold in effect, transcript, recorded duration, embedded
  duration, wake confidence, turn type, outcome, and acoustic measures.
  Migrations 2 through 4 in `database.py`.
- `is_noise_transcript()` in `parsing.py`. Catches bracketed and parenthesized
  sound annotations plus an exact match list of known Whisper hallucinations.
  Exact match on the whole normalized transcript, never substring, so "over."
  is noise but "Read that over for me" is speech. Twelve tests.
- Follow up loop exits on noise instead of reopening the window.
- `MAX_FOLLOWUP_TURNS = 6` hard cap regardless of what the VAD decides.
- webrtcvad mode 2 replaces the energy threshold for both onset and endpoint.
- 300ms pre roll ring buffer plus two frame onset debounce, so soft leading
  consonants are no longer clipped. This fixed transcripts like "year do I
  graduate?" losing their first word.
- `verify_voice()` returns `VERIFIED` / `REJECTED` / `NO_AUDIO`. No voiced
  audio no longer produces a meaningless similarity score or accuses the user
  of being an intruder.
- `measure_acoustics()` logs RMS dBFS, SNR dB, and spectral tilt per attempt.
- `flush_input()` drains the mic buffer after Nova speaks. See "Enclosure"
  below for why this matters more once mic and speaker share a housing.
- Mic gain checked and logged at every service start, with a loud warning and
  the restore command if it drifts.
- `scripts/analyze_verification.py`, with `--after` / `--before` / `--label` /
  `--include-degenerate`.
- `scripts/check_gain.py`, records or analyzes a clip and reports level against
  targets. **Mic dependent**: `MIC_MIXER_CARD` and `MIC_MIXER_CONTROL` in
  `config.py` will need updating for a new capsule.
- `enroll.py` rewritten (see below). **Not yet run.**

---

## Done since the Sep 14 2026 network incident

### Tailscale, installed Sep 15 2026

Was deferred here for part of a day because it had been reported as applied
when it had only been installed on the MacBook. Now real on the Pi: tailscale
1.102.4, `tailscaled` enabled and active, joined as `miles` at
`192.0.2.10` on the tailnet.

Full reasoning is in the decision log in
[engineering decision log](../ENGINEERING_HISTORY.md#decision-log). Kept here only as a pointer, because the
same item was listed as deferred and a future session searching this file for
Tailscale should not find a stale "not done".

---

### Deepgram as the online path (BUILT Sep 25 2026)

Live in `src/deepgram_stt.py`; the mechanism is in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026) and
the privacy decision is in the decision log. What stays here is the reasoning.

Measured Sep 25 2026 and it wins on accuracy, on latency and on staying quiet in
an empty room. The numbers are in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026).
Cost is not a factor: $200 of credit, then $0.0043/min prerecorded, against a
median clip of 2.9 seconds and a few minutes of audio a day.

**Whisper stays.** This is an online upgrade with a local fallback, not a
replacement. `netcheck` already diagnoses which path exists, the phrase bank
already covers what Nova can say offline, and a remote engine failing is exactly
what the local one is for. A failed or slow call must fall through to whisper
rather than producing an empty transcript, which `deepgram_nova3` in the eval
script deliberately does not do, because there an empty string is the honest
score.

Two open questions, neither of them a measurement:

**1. Where it sits relative to speculation.** Speculative transcription starts a
whisper run 210ms into the endpoint silence and has 990ms before the recording
ends. A Deepgram call on those same frames returns in a median 520ms, so it fits
inside the same window and the existing `maybe_speculate` and
`cancel_speculation` structure is the natural home. The open part is what happens
when the remote call is still in flight at the end of the recording: wait on it,
or start whisper alongside and take whichever lands first. The second is more
robust and doubles the CPU cost of a turn.

**2. It sends the room to a third party, continuously.** This is the one that is
not technical. The eval was 58 clips uploaded once, with consent asked for. Live,
**every recording goes out, including the 45 percent of wakes that are false**,
which is how the neighbours and anyone visiting end up transcribed by a company
they have never heard of. The recordings are already treated as sensitive in this
repo, as recorded in the earlier Claude guide. Options worth weighing
before wiring it up: gate the remote call behind `verify_voice` so only confirmed
speech from him leaves the Pi, which costs the speculation overlap; or accept it
and say so explicitly here.

### A level gate on the wake window (BUILT Sep 25 2026, at -42)

Shipped in `src/wake_gate.py`; the live mechanism and its three safety
properties are in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).
What stays here is why the floor is where it is, and what the hardware change
does to it.

**It is not free.**
Reject a wake whose 2.5 second window never peaks above a floor, before the
chime, before recording, before Whisper, before Claude. The full measurement,
including why peak beats RMS and why the line is not -40, is in
[AUDIO_PIPELINE.md](../AUDIO_PIPELINE.md#level-separates-where-the-score-does-not-and-far-field-set-the-line).

The trade, on 38 real wakes and 22 false, Sep 25 2026:

| reject below | real wakes lost | false wakes cut |
|---|---|---|
| -44 dBFS | 0 of 38 | 5 of 22 |
| **-42 dBFS** | **0 of 38** | **7 of 22** |
| -40 dBFS | 2 of 38 | 13 of 22 |

**Start at -42.** It is the lowest line that costs nothing measured, and it
still removes about a third of false wakes, each of which currently costs a
chime, a recording, a transcription and a Claude call before `ignore` catches
it. -40 is tempting at nearly double the yield and it is where two genuine far
field wakes actually landed.

Three things to hold onto before building it:

- **A gate that rejects is a gate that can deafen her, and silently.** It must
  log every rejection with the level it measured, or the first far field
  regression is invisible. Prefer logging the level on every wake from the
  start, gate or no gate, so the floor can be moved on evidence later.
- **It is a stopgap for the neighbours, which is the array's job.** 13 of the 22
  false wakes are quiet and the array addresses their cause; the other 9 arrive
  loud and in the room, and nothing acoustic separates those. Do not let this
  gate stand in for the work in
  [INCIDENTS.md](../INCIDENTS.md#the-near-misses-are-the-neighbours-sep-13-2026).
- **Mic dependent in the number, not the mechanism.** A new capsule and gain
  invalidate -42, and recalibrating is one afternoon of labelled clips rather
  than a retrain. That is what makes it reasonable to build before the hardware
  rather than after.

Only worth it if the level is already to hand at the decision point. It is:
`voice_main` keeps `_wake_window` as a rolling 2.5s of frames, which is what
`save_wake_hit` already writes.

### Session level verification (DONE Aug 11 2026)

Verify once per conversation session rather than once per turn. Follow ups are
short by nature ("yeah", "what about tomorrow") and will never embed reliably.

Shipped in `9c95893`, and **narrower than the proposal that sat here until
Sep 19 2026**. Session trust covers only follow ups under
`MIN_TRUSTWORTHY_SECONDS = 2.0` of embedded audio (`audio.py`). A follow up
longer than that is scored exactly as an initial command is, and a bad score
ends the conversation rather than just the turn (`voice_main.py`). A wake word
restart inside a recording is never session trusted, because it is a fresh
request and long enough to score.

**Those turns log `outcome='session_trust'` with `similarity = 0.0`**, which is
a placeholder and not a measurement. Filter on `outcome` before averaging
similarity or the zeros drag the mean down: over the 30 days to Sep 19 2026,
25 of 148 attempts were session trust.

**Security tradeoff, accepted deliberately:** the exposure is someone speaking
into the mic within ten seconds of the authenticated user, in the same room.
Current actions are weather, timers, reminders, and conversation, nothing
touching money, physical access, or outbound messages. **Revisit this if any
action with real consequence is added**, and use the `tier` column above rather
than reopening the whole decision.

**Note from the data:** the original justification, that follow up turns embed
poorly, is not what the logs showed. Follow ups kept 98.9 percent of their
audio after trimming versus 38.4 percent for initial commands, because
`listen_for_followup()` waits for speech onset while `record_command()` starts
capturing immediately. The security argument stands on its own, the embedding
quality argument does not.

### 1. SUPERSEDE: nothing can update a memory (DONE Aug 11 2026)

`memories.superseded_by INTEGER` is declared at `database.py:29` and appears
exactly once in the entire codebase. Nothing writes it. Verified Aug 11 2026:
zero rows have it set.

The consequence is that the memory store is append only in practice. A memory
recorded wrong stays wrong, and the only remedy is deleting the row by hand.

### 2. Expiry: volatile and references_date are write only (DONE Aug 11 2026)

`volatile` and `references_date` are both written by `save_memory` and read by
nothing. Fifteen seed rows are flagged volatile. Nothing expires them, nothing
filters on them, and nothing acts on a referenced date passing.

Also unranked: `get_episodic_memories` is `ORDER BY id DESC LIMIT 20`, with no
notion of importance or last use. Once more than twenty active memories exist,
older ones fall out of the prompt silently, by id, regardless of value.

Not yet urgent. There are currently zero active episodic memories, so nothing is
being displaced today. That is runway to build this properly rather than
retrofitting it under pressure, not a reason to skip it.

### 3. `remember` as a tool (DONE Aug 11 2026)

During the native tool use migration (Aug 11 2026) the question came up of
whether `[MEMORY:]` and `[MEMORY-EXPLICIT:]` should become a `remember` tool
alongside the action tags. Decision: **not yet**, and the reasoning is recorded
here so it is not relitigated from scratch.

The reason is **not** round trip cost. A `remember` tool with
`returns_to_model=False` costs zero extra Claude calls, which is the entire
purpose of that field. That argument was raised during the migration and it was
wrong.

The real reason is that **automating writes into a store that cannot be
corrected is worse than manual capture**. Manual capture produces errors a human
notices and fixes. Automated capture into an append only store produces errors
that are permanent, accumulate silently, and are only removable by hand editing
SQLite. The failure gets worse the better the tool works.

Built. `src/memory_tool.py`. The bracket tags are gone from the prompt and
`brain.py` strips any stray tag without saving it, so there is one write path.

Still open regardless: `get_episodic_memories` is `ORDER BY id DESC LIMIT 20`
with no ranking, so past twenty active episodic memories the oldest fall out of
the prompt silently, by id, regardless of value.

---
