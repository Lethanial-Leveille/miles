# Engineering history

This archive preserves observations and decisions from the accepted infrastructure
checkpoint. Dates and status labels reflect when entries were written. Some
personal or operator identifiers are represented by placeholders. For today's
workflow use [Engineering sessions](SESSION_START.md). Check current settings
in source and consult the approved [roadmap](ROADMAP.md) for sequencing.

A 2026 startup session planned against a stale handoff and got its central
premise wrong. The dated drift register and decisions below explain how that
failure happened and what was learned.

## Historical production preflight

Original operator procedure retained for context. Use current authorized procedures in [Engineering sessions](SESSION_START.md#authorized-pi-sessions).

```bash
cd ~/miles

# 1. Services actually running?
systemctl is-active miles-voice miles-server miles-tunnel

# 2. Working tree clean? Anything uncommitted from last session?
git status --short && git log --oneline -5

# 3. Test suite green, and how many tests?
cd src && python -m pytest tests/ -q | tail -3 && cd ..

# 4. Which model is live right now?
grep -n "MODEL_A\|MODEL_AB_TEST" src/config.py

# 5. Is prompt caching still engaging? A run of zeroes is the only symptom
#    of the prefix falling under the model minimum.
python3 -c "
import sqlite3
rows = sqlite3.connect('data/miles.db').execute(
  'SELECT cache_read_tokens FROM timing_log ORDER BY id DESC LIMIT 10').fetchall()
print('last 10 cache_read_tokens:', [r[0] for r in rows])"

# 6. Current measured latency, so no one quotes a stale figure
python3 -c "
import sqlite3, statistics as st
rows = sqlite3.connect('data/miles.db').execute(
  'SELECT total_perceived_ms FROM timing_log WHERE total_perceived_ms IS NOT NULL'
  ' ORDER BY id DESC LIMIT 30').fetchall()
v = [r[0] for r in rows]
print(f'perceived latency: n={len(v)} median={st.median(v):.0f}ms') if v else print('no data')"
```

Reading step 5 correctly matters, because two different things produce zeroes:

- **Scattered zeroes** among healthy reads, roughly two or three in ten, are
  normal. That is the 5 minute TTL expiring on turns spaced further apart than
  the window. About 79 percent of observed turn gaps fall inside it, so a 20 to
  30 percent miss rate is the expected steady state.
  Observed Aug 11 2026: `[4751, 0, 0, 4751, 4751, 4521, 4521, 0, 4521, 4521]`.
- **A contiguous run of zeroes**, every recent turn, means the prefix fell under
  the model minimum. **Stop and investigate before doing anything else.** This
  failure is silent: no error, no warning, no exception. That column is the only
  signal.

The read counts themselves varying (4751 against 4521) is also normal. The seed
and episodic blocks are inside the cached region and change size as memories are
added.

---

## Drift register

Claims that have drifted before. Each row has the command that settles it.
Check these when a document makes a claim about them, and add a row whenever a
new drift is caught.

| Claim | How to check | Last verified |
|---|---|---|
| Which model serves turns | `grep MODEL_A src/config.py` | Aug 11 2026 |
| Which tools Nova actually has | `python3 -c "import brain; from tools import registry; print(registry.names())"` | Sep 25 2026 (30) |
| Test count | `cd src && python -m pytest tests/ -q \| tail -1` | Oct 2 2026 (1285) |
| Perceived latency | preflight step 6 | Aug 11 2026 (4938ms median) |
| Prefix token count (never trust a written figure) | `count_tokens` on the tool schemas plus `build_enhanced_prompt` output, since tools sit inside the cached prefix, vs the 4096 floor | Sep 25 2026 (17774: tools 7103, system 10670) |
| `VERIFY_THRESHOLD` | `grep VERIFY_THRESHOLD src/config.py` | Sep 27 2026 (0.15, ecapa) |
| Speaker device resolution | `grep -n SPEAKER src/config.py src/tts.py` | Aug 11 2026 (by name, not device string) |
| Which modules exist | `ls src/*.py` | Aug 11 2026 (20 modules) |
| Schema version | `grep -c "^    (" src/database.py` around `MIGRATIONS` | Oct 30 2026 (34) |

### Why these specific ones drifted

Every one of them drifted the same way: the document recorded an **intention**
and the code recorded a **decision**, and nobody reconciled them. `stop_sequences`
was in a pipeline diagram as the plan. Sonnet was the model when CLAUDE.md was
written. The latency table was accurate the day it was measured. None of these
were careless; they were all true once.

That is the actual lesson. Docs do not drift because anyone was sloppy. They
drift because a doc records a moment and code records the present, so the fix
is not "be more careful" but "re verify on a schedule". The preflight is that
schedule.

---

## Decision log

Chronological. Newest at the bottom. Each entry: what was decided, why, and its
status. Status markers: `(DONE)`, `(NEXT)`, `(IN PROGRESS)`, `(DEFERRED)`,
`(SUPERSEDED)`. `DIVERGENCE` flags production differing from the written plan.

### Endpointing moved from amplitude to webrtcvad (Aug 10 2026) (DONE)

- `SILENCE_THRESHOLD = 200` on mean absolute amplitude was never crossed by real
  speech, because capture level was about -53 dBFS RMS.
  - Consequence: recordings ended on the timeout, not on speech, so every
    recording was 3.0 to 3.4 seconds and any command longer than that was
    truncated mid sentence. Several garbled transcripts blamed on Whisper were
    actually truncated audio.
- Replaced with webrtcvad at `VAD_MODE = 2`.
  - Reason for 2 rather than 3: Resemblyzer uses 3 for offline trimming where
    clipping a soft onset costs nothing. Live capture is less forgiving.
- `VAD_PREROLL_MS = 300` added because frames containing a soft leading consonant
  were discarded before capture started, which truncated "What year do I
  graduate?" to "year do I graduate?".

### `SILENCE_LIMIT` 3.0 to 0.9 (Aug 10 2026) (DONE)

- The 3.0 value was tuned when endpointing ran on a threshold that never fired,
  so it was adding three seconds and doing nothing else.
- Set to 0.9 against a measured worst internal pause of 0.63s.
- **Caveat recorded deliberately:** that 0.63s came from reading a scripted
  enrollment phrase. Spontaneous conversation carries longer pauses than read
  speech, so the margin is thinner in practice than it looks. `max_pause_ms` is
  logged per turn so this can be retuned from real conversation.
- Measured effect: median perceived latency 8088ms to 4938ms.

### Haiku 4.5 over Sonnet 4.5 (Aug 10 2026) (DONE)

- Measured A/B over twenty turns, strict alternation rather than randomization.
  - Reason for alternation: with a day of turns, random assignment can hand one
    arm a 60/40 split and pair it with a slow network stretch.
- Result: 614ms faster median time to first token, 31 percent lower, p=0.0007 on
  a permutation test. Distributions barely overlapped.
- Haiku also drew more follow up turns, which carry larger prompts, so the split
  worked against it and it won anyway.
- A/B harness left in place but `MODEL_AB_TEST = False`.
  - **Pin it off before measuring anything cache related.** Prompt caches are
    model scoped, so alternation makes every turn a miss.

### Prompt caching on the system prompt (Aug 10 2026) (DONE, fragile)

- System prompt is the stable prefix. Conversation history changes every turn and
  sits after the breakpoint.
- Measured effect on a cache hit: time to first token 1995ms to 639ms.
- **Fragile:** Haiku 4.5 needs a 4096 token cacheable prefix; the assembled
  prompt is about 4165. Roughly 69 tokens of margin. Falling under it disables
  caching with no error. `cache_read_tokens` is logged for exactly this reason.
- Known defect, deferred: `_episodic_block` is appended *last* in
  `build_enhanced_prompt`, inside the cached region, so every explicit memory
  save invalidates the whole prefix. `(DEFERRED)` to BACKEND_TODO.

### Response length cut by trimming history, not by instruction (Aug 10 2026) (DONE)

- Holding the prompt fixed and varying only history: 94 words with full history
  against 54 with assistant turns trimmed to 30 words.
- Nova was few shot learning her own verbosity from her own transcript, and each
  long answer made the next likelier.
- Trimming beat the alternatives. Sending no history scored worse (64 words, much
  longer tail) because an open question with no context invites a survey.
- Generalized lesson worth keeping: **the transcript is a stronger length signal
  than any instruction in the system prompt.**

### Exit phrases replaced by intent based dismiss (Aug 11 2026) (DONE)

- Twenty six exact strings could not match "alright thanks Nova" and seven of them
  ("later", "peace", "I'm good", "that's it") are ordinary mid conversation
  utterances that would have ended a session by accident.
- Replaced with `[ACTION: dismiss]`, judged on intent.

### Clock injected into the last user turn, not the system prompt (Aug 11
2026) (DONE)

- Nova had no clock and, rather than saying so, copied the date out of the
  reminder example in her own prompt. Every reminder was dated months in the
  past, so none could ever fire.
- Injected into the final user turn by `_with_current_time`, **not** the system
  prompt.
  - Reason: the system prompt is the cached prefix. A value changing every turn
    is the textbook silent cache invalidator. Caching would report success while
    never producing a hit.

### Native tool use migration (Aug 11 2026) (IN PROGRESS)

- **DIVERGENCE found:** the handoff document claimed `stop_sequences=["[ACTION:"]`
  was passed on every call, and that a tool result could therefore never return
  to the model. Neither is true in code. `ACTION_PREFIX` is consumed only by
  `StreamRouter`, and a two round trip loop for weather already exists at
  `brain.py` around the `needs_data` check.
- Migration proceeds anyway, with the **reason restated**: it is not a capability
  unlock. `needs_data` is one line and could become a set membership check.
  The reason is that adding a tool today requires consistent edits across prompt
  prose, `extract_actions`, `_parse_action_tag`, `execute_actions`, and that line,
  with nothing enforcing consistency. Four more tools is sixteen chances to miss
  one, and the failure mode is a capability the prompt does not know exists.
- Decisions taken:
  - `dismiss` becomes a tool with `returns_to_model=False` and no execution.
    Reason: it is a real state transition that exits the follow up loop, unlike
    `[calmly]` which is a rendering hint with no side effect. Making it a tool
    lets Phase 2 delete the tag parser outright rather than keeping it alive for
    one case, and every dismiss becomes a `tool_call_log` row.
  - No added acknowledgment during tool execution. Use the text block that can
    precede `tool_use` in the same response, which gives the bridge sentence free
    and in Nova's own words. Any conditional bridge behavior ships off, and gets
    decided from the `tool_ms` distribution after a week rather than from a guess
    about which tools feel slow.
  - Tool results persist to a new `tool_call_log` table, capped by age the way
    `ARCHIVE_MAX_FILES` caps recordings. Reason they do not go in
    `conversation_history`: that table feeds `get_recent_messages`, so last
    Tuesday's weather would become prompt context. Reason not `memories`:
    memories are facts about the owner, tool results are facts about the world at
    one instant.
  - Within a turn, results live in the messages array as real `tool_result`
    blocks. Across turns they do not, and `get_recent_messages` keeps returning
    plain text.
- **Gate on Phase 2, blocking:** measure the prompt with `count_tokens` at three
  points, now, after Phase 1, and against a simulated post Phase 2 prompt before
  deleting anything. Phase 2 removes roughly 30 lines of bracket instructions
  from inside the cached prefix while Phase 1 adds tool schemas to it, and those
  move in opposite directions against a 69 token margin. If the projection lands
  under 4096, stop and report rather than shipping.
  - If it does go under, the fix is to move genuinely stable content from after
    the breakpoint to before it, not to pad with filler. Padding is tokens paid
    on every cache write for nothing.

### Weather and clock verbosity (Aug 11 2026) (NEXT)

- Weather returns a finished English sentence containing four facts, so Nova
  reads the sentence aloud. Fix is the return value, not the prompt: a dict
  cannot be read as a paragraph.
- Clock verbosity is a prompt fix, not a tool. The stamp is already in every user
  turn, so a `get_time` tool would add a round trip to retrieve a value already
  in context.
  - **Sequencing hazard:** the clock guidance currently lives inside
    `ACTION_AND_MEMORY_INSTRUCTIONS`, which Phase 2 deletes. It must be lifted
    into its own section first, or Phase 2 reintroduces the reminder date bug
    fixed in `5ad97de`.

### Tool registry, Phase 1 step 1 (Aug 11 2026) (DONE)

`src/tools.py` plus `src/tests/test_tools.py`. No API wiring, no `brain.py`
changes. Test count 72 to 100.

- **The decorator returns the function unchanged**, registration is a side
  effect only.
  - Reason: every tool stays an ordinary function that can be called and tested
    directly, and registration can never alter runtime behavior. A wrapper would
    make the registry a thing that can break a working tool.
- **`ToolRegistry` is a class with a module level singleton**, `tools.registry`,
  and `tools.tool` bound to it.
  - Reason: tests build a throwaway registry, so no test can leak a tool into
    another test or into a live turn.
- **Six fields per tool**: name, description, input schema, permission,
  `returns_to_model`, function.
  - `returns_to_model` is the field that replaces the hardcoded `needs_data`
    whitelist. True means a second Claude call, false means the text produced
    alongside the call is the final answer.
  - `Permission` has four tiers: READ, WRITE, EXTERNAL_WRITE, CONTROL. Defined
    now, enforced later. Reason for defining early: retrofitting a tier across a
    dozen tools costs more than recording it at registration, and a tool added
    without one would be the tool that needed it most.
- **Validation runs at import time.** The check that earns its keep is that every
  name in `required` must exist in `properties`. The API does not catch that
  typo; it surfaces as the model omitting a parameter it was told was mandatory,
  with no error raised anywhere.
- **`api_schemas()` sorts by name.** Load bearing, not tidiness: tools render
  ahead of the system prompt in the cached prefix, so registration order leaking
  into that list would move every byte after it and silently drop every cache
  hit.
- **No `summary` field.** The capability line is the first sentence of the
  description.
  - Reason: a separate summary is one more thing that can drift, and deriving it
    puts a vague first sentence into the prompt where it gets noticed.
- **`capability_prose()` is one line per tool, not full descriptions.**
  - Reason: full descriptions already reach the model as tool schemas, so
    repeating them would pay for the same tokens twice against a 69 token cache
    margin. What the block adds is the boundary statement, that the list is
    complete, which is what separates a missing capability from a forgotten one.
- **`call()` lets argument mismatches raise `TypeError` naturally.**
  - Reason: that is a programming error between schema and signature. It should
    be loud, and wrapping it would hide which parameter was wrong.

### Prompt wiring, Phase 1 step 2 (Aug 11 2026) (DONE)

`NATIVE_TOOLS` flag, `ACTION_AND_MEMORY_INSTRUCTIONS` split three ways,
capability block generated from the registry. `brain.py` untouched. Tests 100 to
113.

- **DIVERGENCE, and the most important finding of the session.** The "69 tokens
  of cache margin" figure in `config.py` and `CLAUDE.md` was stale by roughly
  600 tokens. Measured prefix is 4836, not 4165. Real margin is +740.
  - It was load bearing: an entire phase of this migration was gated on it, and
    the gate would have been evaluated against a number that was wrong by an
    order of magnitude relative to the margin it described.
  - Corroborated independently by live `cache_read_tokens` values of 4751 and
    4521, which no one had compared against the written figure.
  - Added to the drift register. **Never quote a prefix size that was not
    measured this session.**
- **The block was split three ways, not two.** It bundled three lifetimes:
  memory tags stay indefinitely, clock guidance is permanent, action tags die in
  Phase 2. Gating the third without taking the other two required separating all
  three.
- **The clock paragraph was buried mid list inside the action instructions.**
  Phase 2 deleting that block wholesale would have taken it and reintroduced the
  bug fixed in `5ad97de`. Promoted to its own `CLOCK_INSTRUCTIONS` constant,
  included in both paths, with a regression test asserting it survives each.
- **Clock conciseness folded in here rather than deferred.** Asking for the time
  returned the time, the date, the day, and the year, because the injected stamp
  contains all four and nothing said to pick one. Two sentences, added to the
  block that was already being rewritten.
  - Deliberately not a tool. The stamp is already in every user turn, so a
    `get_time` tool would spend a full round trip retrieving a value already in
    context, on the most latency sensitive question there is.
- **One time cache invalidation accepted.** Extracting a paragraph from the
  middle of a block reorders bytes in the cached prefix. One turn pays a write.
  Contorting the structure to preserve byte identity was not worth it.
- Measured, `count_tokens` against `claude-haiku-4-5`, 4096 floor:
  - before step 2: 4757 (+661)
  - after, flag off (production): 4836 (+740)
  - after, flag on, empty registry: 4186 (+90)
- **Phase 2 gate result: PASS, but conditionally.** Projected prefix after
  deleting the action instructions is roughly 4186, about +90. That is not a
  margin. **The deletion and the tool schemas must land in the same commit.**

### Weather as the first registered tool, Phase 1 step 3 (Aug 11 2026) (DONE)

`get_weather` split three ways, forecast lookup added, geocode cached. Tests 113
to 136.

- **The verbosity fix is the return value, not the prompt.** The old function
  returned a finished English paragraph carrying four facts, so Nova read the
  paragraph aloud whatever was asked. Instructions to be brief fought the data
  she was handed. `_fetch_weather` now returns a dict, which cannot be read as a
  paragraph.
  - Generalized lesson, same shape as the history trimming finding: **when the
    model is over reporting, look at what you handed it before you look at the
    prompt.**
- **Three functions, not two.** `_fetch_weather` returns the dict, the
  registered tool returns it unchanged, and `get_weather` formats a short line
  for the legacy bracket path. The legacy formatter exists so the verbosity fix
  lands now rather than waiting on `NATIVE_TOOLS`, and it dies with the tag
  system in Phase 2.
- **Humidity and wind stay in the payload**, with the restraint written into the
  tool description rather than enforced by withholding.
  - Reason: withholding means "is it windy" hits the capability gap joke when
    the data was right there. Wrong answer, and the joke stops being funny when
    it is covering for a design choice.
- **Precipitation comes from `/data/2.5/forecast`**, same free key, three hour
  steps, four blocks of lookahead for twelve hours.
  - Honest limit recorded: this cannot say "it stops in twenty minutes". Minute
    level precipitation is One Call 3.0, a separate signup with a card on file.
    Take the coarse version until it actually annoys someone.
  - Condition id boundary is 700. Below is falling out of the sky, at or above
    is not. Getting this wrong makes Nova announce rain on a foggy morning,
    which is why there is a parametrized test walking the boundary.
  - A failed forecast returns None rather than failing the lookup. Current
    conditions are still worth answering with.
- **`precip: None` is the common case and the description says to say nothing
  at all**, rather than announcing that it will not rain. A tool that always
  returns a rain field invites rain talk on a clear day.
- **Geocode cached per process.** Coordinates for a place name do not change and
  it was being re resolved on every request. Removes one HTTP call, roughly
  paying for the forecast call added.
- Prefix after registration: 4991 with the flag on, margin +895. Weather's
  schema and capability line cost 805 tokens. The Phase 2 squeeze recorded in
  the step 2 entry resolves itself once tools exist.


### Phase 2 and 3: tool use is live (Aug 11 2026) (DONE)

Bracket action tags deleted, six tools registered, `NATIVE_TOOLS` flag removed.
Tests 136 to 166. Prefix 5942, margin +1846.

- **Deleted:** `ACTION_TAG_INSTRUCTIONS`, `parsing.extract_actions`,
  `brain._parse_action_tag`, `actions.execute_actions`, the legacy prose
  `get_weather`, `ACTION_PREFIX`, `LOOKAHEAD_CHARS`, and StreamRouter's tag
  detection. **Survived:** `strip_leading_bracket_cue` for emotion cues and
  `extract_memories`, both with a note on the function saying why.
- **Dropping the lookahead is a latency win, not just a simplification.** The
  router used to buffer 50 characters before considering anything, and that
  wait sat directly on the path to first audio. It existed only to stop a stray
  "[" being read as a tag. Tool calls arrive in their own content blocks, so a
  short first sentence now flushes immediately.
- **The follow up call is a bounded loop, not one call.** Treating it as
  guaranteed to produce speech was a real bug: it intermittently re-called
  `get_weather` instead of reading the result back, which yielded no text and
  returned an empty turn. `MAX_TOOL_ROUNDS` bounds it and the final round is
  made without `tools`, so the model has nothing left to reach for and must
  answer. A hard floor rather than a hope that it converges.
- **DIVERGENCE, caught only by a live call.** Every mocked test passed while
  Nova refused to call `get_weather` at all, asking "where?" every time. The
  cause was in the tool description: it said to use "his home location", and
  the seed memories name more than one place he lives, so the ambiguity was
  real and refusing to guess was correct behavior. Fixed by interpolating
  `DEFAULT_LOCATION` into the description so the two can never disagree.
  - Lesson worth keeping: **a tool description is prompt text and needs the
    same scrutiny.** The unit tests asserted it mentioned rain and jackets.
    They could not assert that it was unambiguous.
- **A smoke test poisoned its own next run.** The first failed turn wrote "I
  need a location" into `conversation_history`, and every following turn read
  it back as an example and repeated it. Same few shot self teaching effect as
  the verbosity finding. Smoke tests against the real database must delete
  their rows by id afterwards.
- Measured cost of an action turn: `tool_ms` 450 to 600ms for weather (two HTTP
  calls), `second_ttft_ms` roughly 620ms. About 1.1s on top of a plain turn.
  That is the figure the bridge sentence decision was deferred for; it is now
  measurable per tool rather than guessed.
- **Not built: Hevy and Google Calendar.** Blocked on scoping, credentials, and
  a decision about what a calendar write is permitted to do. Not inventing an
  API contract for a service that writes to a real calendar.


### Live testing found four defects the tests could not (Aug 11 2026) (DONE)

Every one of these passed a green suite of 166 tests and failed in the room.

- **Fire and forget tools were silent.** `final_text = " ".join(spoken_parts) or
  "Done."` assigned the fallback to the *returned* string and never spoke it.
  When the model said nothing alongside the call, which it usually does, the
  turn produced no audio at all while the database recorded a confirmation that
  was never heard. Timers, reminders and cancellations all landed this way and
  read as the tool having failed when it had worked. The fallback is now spoken.
  - Confirmed in data before fixing: `tts_first_audio_ms` and
    `total_perceived_ms` were both null on those three turns.
- **`tool_ms` was measuring the wrong thing.** It wrapped
  `asyncio.gather(tool, tts_task)`, so it reported whichever finished last. A
  weather call taking 580ms logged 6006ms, because it was really timing the
  bridge sentence playing. Now the tool is awaited and timed on its own; both
  still overlap because the TTS task is already scheduled.
  - This one mattered beyond accuracy: the bridge sentence decision was
    explicitly deferred until `tool_ms` could be read, and the column was
    unusable for that.
- **The bridge sometimes answered the question before the tool ran.** The model
  occasionally emits a full spoken answer *and* a `tool_use` in one response, so
  the owner heard a guess and then the real reading. Added `TOOL_SPEECH` to the
  prompt: never state a value before the call, say nothing or say a phrase that
  commits to nothing, and do not restate afterwards.
  - Non deterministic, which is why it survived several reruns before showing up.
- **`get_system_state` reported the core temperature in Celsius** while weather
  answered in Fahrenheit, so one conversation carried two scales. Now
  `core_temp_f`.

Also: `FOLLOWUP_TIMEOUT` is a config constant at 6 seconds, down from a bare 10
in `voice_main.py`. Ten seconds is a long time to stand in a quiet room deciding
whether you are done, and every expiry costs a full window of dead air.


### Voice tuning, and why it took so long (Aug 11 2026) (DONE)

Final: Victoria (`[voice identifier omitted]`), `eleven_flash_v2`, stability 0.80,
similarity 0.75, style 0.00, speed 1.00, `TTS_PHONEME_TAGS = True`, and
the owner as `L AE0 TH AE1 N Y AH0 L`.

The lesson is not the values. It is that **most of this hunt was aimed at the
wrong variable**, and the thing that exposed it was the owner noticing that two
repeats of the same input sounded different.

- **The measurement was broken before the candidates were.** Rankings were being
  collected one rendition per candidate. ElevenLabs produces a different
  rendition every call, and the spread between two renditions of identical input
  was as wide as the spread between candidates. Candidate 1 was rated "no" then
  "yes" on the same string. Every ranking taken before the seed was pinned was
  partly recording which generation got lucky.
  - The SDK supports `seed`. `speak()` now takes one. Production leaves it None,
    because varied delivery is wanted in conversation. **Comparisons must set
    it.**
- **`stability` was the dominant term the entire time.** 0.60 was production
  throughout the period the name sounded butchered, and rates "eh" on its own.
  Stability governs how much one rendition varies from the next, which is
  exactly why a correct phoneme string came out wrong intermittently.
  - It had been flagged as worth watching when the voice settings were first
    discussed, and then not connected to the pronunciation problem for several
    rounds of phoneme hunting. Connecting a knob to a symptom is the work;
    naming the knob is not.
- **Phonemes beat respellings, decisively.** Of twenty five candidates, the six
  that survived a first listen were all phoneme strings and not one respelling
  made the shortlist. That is what justified moving to `eleven_flash_v2`.
- **flash v2.5 does not ignore phoneme tags, it drops the word they wrap.**
  Measured: plain 0.79s, tagged 0.23s, absurd phonemes also 0.23s. Identical
  output for different phoneme strings means the content is discarded. v2 honors
  them and costs nothing: 349ms against 347ms time to first byte.
- **Stability is one dial with two failure modes.** It buys consistency by
  reducing variation, and that same variation is what makes delivery sound
  alive. 0.75 read better on ordinary sentences and occasionally missed the
  name; 0.90 held the name and read flatter. There is no setting that gives
  both, so 0.80 is a chosen point on a trade rather than a solution.

`scripts/pronounce.py` carries the method: `sweep` to bracket, `spread` for one
candidate across seeds, `stability` across settings, `demo` for real responses
at real length. **Start any future voice question with `spread`**, because
"how much does this vary" has to be answered before "which one is better" means
anything.


### Memory correction: SUPERSEDE and expiry (Aug 11 2026) (DONE)

Migration 013. `supersede_memory`, `get_memory_chain`, `expire_memories`, and
`scripts/memory.py`. Tests 233 to 251.

- **Only one new column.** `superseded_at`. `status` is free text and every
  retrieval already filtered on `'active'`, so retiring a row to `'superseded'`
  or `'expired'` removes it from the prompt with no query changes anywhere.
- **Supersede rather than edit in place.** The old row is retired and pointed at
  its replacement.
  - Reason: "the exam moved to Thursday" is different information from "the exam
    was always Thursday", and an update or a delete cannot tell them apart.
    `get_memory_chain` is what that buys, and it is the only reason to keep the
    old row at all.
- **Classification is inherited on correction unless overridden.** A correction
  is usually the same kind of fact, and re-specifying every field to fix a typo
  is how fields drift apart.
- **Expiry is enforced at read, not by a sweep.** `get_episodic_memories`
  excludes volatile rows whose `references_date` has passed.
  - Reason: no job to schedule and nothing to fall out of sync. A memory becomes
    invisible the moment its date passes whether or not any sweep has run.
    `expire_memories` only marks what the read already hides, so an expired row
    shows as expired in a listing rather than looking active and mysteriously
    absent from her answers.
- **`volatile` without a date never expires.** Volatile says a fact is
  temporary, not when it stops. Expiring without a date would be guessing, and
  fifteen seed rows are in exactly that state.
- **`remember` is now unblocked** and deliberately still not built. The reason
  it was blocked is gone; the work itself has not been done.


### remember as a tool, with memory ids in the prompt (Aug 11 2026) (DONE)

`src/memory_tool.py`, ids rendered as `(#61)` in both memory blocks, bracket tag
instructions deleted. Tests 251 to 273. Prefix 7338, margin +3242.

- **The trigger was a real duplicate.** A pending implicit memory read "Traveled
  to Singapore before summer 2026", while seed rows 61 and 99 already covered
  the Singapore internship. The duplicate guard is exact string match, so two
  different sentences about one fact both stored.
- **Ids mattered more than retrieval.** Nova already sees every memory every
  turn, so she never needed to *fetch* one to notice a duplicate. What she
  lacked was a way to *name* one. `(#61)` costs a few tokens and makes
  `supersedes` expressible.
- **The tool makes three moves possible where the tag made one.** Store,
  supersede, or do nothing. The tag could only add a row, which is why the only
  available outcome for an already known fact was a second copy.
- **`certainty` preserves the review queue.** "asked" writes active, "inferred"
  writes pending, matching exactly what explicit and implicit meant. It defaults
  to inferred, because defaulting to asked would put every guess straight into
  his permanent record, which is the expensive direction to be wrong in.
- **A bad `supersedes` id stores rather than dropping the fact.** A wrong
  reference is a worse reason to lose information than a duplicate is to keep
  it, and the model is told so it can correct itself.
- **One write path, deliberately.** `brain.py` still strips bracket tags so a
  stray one is not spoken, but no longer saves them. Leaving both live would
  double write the fact the tool just stored, which is the exact duplication
  being fixed.
- Verified live on the first attempt: "remember my exam is Friday" stored,
  "actually it moved to Thursday" superseded 103 with 104, and "remember I go to
  UF" drew "You're already on record for that" with no tool call at all.

**RAG was considered and rejected**, and the reasoning is worth keeping because
it will come up again. 100 memories is 2293 tokens against a 200,000 token
window, roughly 1 percent. RAG solves a corpus that does not fit, and this one
fits forty times over. It also does not deduplicate, since it is retrieval, and
it would make deduplication *worse*: with top k retrieval Nova sees only part of
what she knows, so she cannot reliably tell whether a fact is already stored and
would write duplicates precisely because the original was not retrieved. Full
corpus in prompt is strictly better for this. Revisit around 400 to 500 rows or
10,000 tokens of memory blocks.

What actually breaks first is not context size, it is
`get_episodic_memories` being `ORDER BY id DESC LIMIT 20` with no ranking.
Deferred by choice.

**A defect found by asking "do the ids line up".** The memory instructions used
`(#61)` and `(#42)` as examples. Nova cannot tell an example from a real row, so
she could have superseded `#42` purely because it sat beside the word supersede,
silently destroying a correct memory and replacing it with something unrelated.

That is the same failure as `5ad97de`, where she copied the reminder date out of
the example in her own prompt and dated every reminder months in the past.
**Anything in the prompt that is shaped like real data will be treated as real
data.** A second, subtler instance was in the same block: the example sentence
"his exam moved to Thursday" reads exactly like a memory. Both are now abstract,
and two tests pin it: every id appearing anywhere in the prompt must be backed by
a real row, and the instructions must contain no id shaped text at all.

### CLAUDE.md is version controlled now (Sep 13 2026) (DONE)

It was gitignored, listed under the **Personal data** block beside `data/`, from
the v0.7 module split (`1d38fb2`) until today. That block's comment justifies
`data/` and says nothing about CLAUDE.md, so the exclusion carried no recorded
reason and nobody could tell whether it was deliberate or a line appended in a
hurry.

Untracked was the wrong state for it. **Rule zero makes CLAUDE.md a document
that must be corrected in the same session as the code it describes**, and an
untracked file has no diff, so a correction leaves no trace, carries no commit
message, and cannot be reviewed. The end of session checklist below asks for
corrections in the commit message; for this one file that was impossible.

Checked before tracking it: 12 assignments in `.env` and `~/.bashrc` scanned
against the file, **no secret value appears in it**. It names environment
variables, never their contents. The only identifier it carries, the Victoria
voice id, is already in tracked `src/config.py`.

The learning constraint lost "pause and ask if I have questions" and "quiz me on
important concepts" in the same commit. Both predate the current working
rhythm and neither earned the interruption. The remaining six stand.

Because the file joins history as a new file, that trim is not visible as a
diff. It is recorded here instead, which is the point of this log.

### CLAUDE.md split into topic docs (Sep 13 2026) (DONE)

CLAUDE.md had reached 1230 lines and was loaded in full at the start of every
session. Most of it was measurement history and root cause writeups that matter
only when someone touches that specific subsystem. It is now **320 lines** and
holds only what is true on every session: status, layout, live config values,
production commands, and the rules of the road. Six topic docs carry the rest,
indexed from CLAUDE.md and linking back.

**The split rule at the time.** CLAUDE.md declared; the
topic docs explain. A live config value appears in the CLAUDE.md table and
nowhere else, because two copies of a number is how a repo stops being able to
say which is current. A number *inside* a measurement stays in the measurement
verbatim, because it is a record of what was true on a date, not a declaration
of what is true now. Those two rules look like they conflict and do not.

Everything moved byte for byte by line range rather than being retyped, and a
coverage check confirmed no substantive line was dropped. The measurements are
the most valuable content in the repo and several of them exist precisely
because a stale claim cost an earlier session real work. **Do not summarize
them.**

**The split found six drifted claims, which is the argument for having done
it.** They had been sitting in a file too long to reread:

| Claim | Source said |
|---|---|
| `WAKE_MISS_FLOOR = 0.05`, "below `WAKE_LOG_FLOOR`, deliberately" | Both are `0.15`. Equal, not below |
| `ACK_SPOKEN_CHANCE = 0.5` | `0.75`, changed Aug 12 and never propagated |
| `LOOKAHEAD_CHARS = 50` and `ACTION_PREFIX` listed as live config | Neither exists anywhere in the codebase |
| StreamRouter buffers lookahead and strips `[ACTION:...]` tags | Native tool use removed all of it; text is only ever text |
| Second call gated by `any(r["type"] == "weather" ...)` | Gated by the registry's `returns_to_model` |
| `TTS_PHONEME_TAGS` written as something to turn on later | Already `True` |
| Repo tree: "367 tests" | 500 passing, 6 skipped |

Two of those were **load bearing**. The lookahead is cited in `docs/LATENCY.md`
as the reason `first_sentence_ms` is 288ms of optimizable time, and that stage
is now unmeasured rather than known. The weather whitelist was written up as a
limitation worth fixing and had already been fixed.

Drifted claims were **not** deleted. Where the original still carries a lesson
it is kept verbatim with a dated `> **Correction**` blockquote beneath it, so
the mistake and the fix are both legible. A silently corrected doc teaches
nothing, which is rule zero applied to itself.

### Dev tools moved out of `src/` (Sep 13 2026) (DONE)

`src/` mixed the modules the three services import with one off analysis and
setup tools that nothing imports. Seven moved to `scripts/`: `analyze_timing`,
`analyze_verification`, `compare_whisper`, `check_gain`, `profile_turn`,
`seed_memories`, `setup_auth`.

**`src/` was deliberately not packaged into subfolders.** No `src/audio/`, no
`src/core/`. The runtime modules stay flat where they are, because restructuring
them breaks every import, every systemd `ExecStart`, and the test suite, for no
functional gain. The line worth drawing is runtime against operator tool, and
that line is now the directory.

**`enroll.py` stays in `src/`.** `tests/test_enrollment_audio.py` imports it as a
module, which makes it the one candidate with real runtime coupling. It is also
the counterpart to `speaker_encoder.py` and has to move in the same session as
the ECAPA swap, not before it.

**This was not a pure rename, and could not be.** Four of the seven did a bare
`from config import ...` that worked only because Python puts a script's own
directory on `sys.path`, and their directory was `src/`. Each needed the
`sys.path.insert` shim every other script in `scripts/` already carries. The
`os.path.abspath` form was chosen over the `os.path.join(..., "..")` form because
it is what seven of the eleven existing scripts use, including the newest.

**The hazard worth recording is `setup_auth.py`.** It computes
`ENV_PATH = Path(__file__).resolve().parent.parent / ".env"`, so a move to the
wrong depth would have pointed it somewhere else and the fix would have been to
**overwrite the real `.env`**, destroying the password hash and JWT secret.
`src/` and `scripts/` are both one level below the repo root, so it resolves
identically and needed no change. That was checked before the file was moved, not
after. **Check `__file__` relative paths before moving any script, especially one
that writes.**

Commands in all docs are now written to run **from the repo root**
(`python3 scripts/check_gain.py`), which is what the existing `scripts/` entries
already assumed. The `../build/` and `../whisper.cpp/` paths in the moved
docstrings were relative to `src/` and are now repo root relative.

### Permission gate enforced, and outside writes wait a turn (Sep 13 2026) (DONE)

`Permission` was recorded on every tool from the migration onward and read by
nothing; `tools.py` said "defined now, enforced later". `permits()` in the
executor enforces it now. The mechanism and its rules are in
[BRAIN.md](BRAIN.md#tools-and-the-permission-gate). This entry is why the choices
went the way they did.

**Confirmation is enforced by turn order, not by instruction.** Rejected: a
prompt rule to always ask first, which is a suggestion and the first thing a long
conversation erodes; and a confirm tool that takes the event details, which would
let the model confirm a different event than the one read back. Chosen: the
proposal is held in code, confirm takes only yes or no, and it is accepted only
on the next turn inside `CONFIRM_WINDOW_S`.

**Only EXTERNAL_WRITE is confirmed.** Timers, reminders and memories live on
this Pi and are undone by voice. A calendar write lands on a system other people
read. Confirming everything would make "yes" the word said without listening.

**Private reads are hokage through `min_tier`, not a fifth `Permission`.**
Sensitivity is per tool, and a new category would have to be threaded through
every place that switches on the four. `min_tier` can only raise, so it cannot be
used to weaken the table.

**`lower_access` is WRITE again.** It had been moved to EXTERNAL_WRITE to reach
hokage, which mislabelled a local database write and broke
`test_registered_tools`. `min_tier` gives it the floor without the wrong label.

**Oura and calendar results are shaped for the model, not passed through.**
Units in every field name, local times already formatted, busy blocks merged,
free gaps computed. Each came from a wrong answer, recorded in
[INCIDENTS.md](INCIDENTS.md#calendar-and-sleep-tools-answered-confidently-and-wrong-sep-13-2026).

### Calendar edit and delete look events up by title and day (Sep 13 2026) (DONE)

**Not by event id.** Conversation history stores what Nova said, not tool
results, so an id from a listing does not survive to the turn where he says
"delete it". Rejected: returning ids and trusting the model to carry them, which
fails silently the moment it has to guess one. Chosen: the tool takes title words
and a day, code requires exactly one match on the MILES calendar, and anything
else is answered with what is actually there.

**MILES calendar only, for now.** Shared calendars make a delete reach other
people. Widen it only after the confirmation flow has earned trust in real use.

**Single occurrences only.** The instance id is all the lookup ever holds, so a
repeating event cannot be deleted as a series by accident, and the read back says
so.

### Every recording goes to Deepgram (Sep 25 2026) (DECIDED)

Deepgram Nova 3 measured 7.6% WER against 17.5% for the best local candidate, and
faster, and better at staying quiet in an empty room. The numbers are in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026).

**The cost accepted is not technical.** Running it live sends every recording off
the Pi, and **45 percent of wakes are false**, so that includes clips of his
roommates and of the neighbours through the wall, who have not agreed to
anything. The alternative offered was gating the remote call behind
`verify_voice` so only confirmed speech from him leaves the machine, at roughly
200ms of the latency win, since verification currently runs beside transcription
rather than before it.

the owner chose to accept it on the grounds that it is his apartment and his
assistant. Recorded so a later session does not reopen it. **Revisit if anyone
else is enrolled**, because the multi speaker work in
[BACKEND_TODO.md](BACKEND_TODO.md) changes who "his voice" means, or if the
recordings are ever shared beyond this Pi.

The `verify_voice` gate stays written down as the known alternative rather than
being deleted, so the decision can be reversed without rediscovering how.

### The Oura client secret is not rotated (Sep 13 2026) (DECIDED)

It sat as a fallback default in the untracked `scripts/oura_auth.py`, was never
committed (every commit in history checked), and was moved into `.env`.
the owner chose not to rotate it. Recorded so a later session does not reopen it;
revisit only if the file or `.env` is ever exposed.

### Voice moved to eleven_v3 at stability 1.0 (Sep 13 2026) (DONE)

He described Nova as reading rather than speaking. Every suspect was tested by
rendering the same real reply side by side and listening; the table is in
[VOICE_OUTPUT.md](VOICE_OUTPUT.md#moved-to-eleven_v3-sep-13-2026). Only the model
made a clear difference. Kept Victoria over four conversational voices by his
choice.

**Rejected on evidence, not preference:** rewriting sentence mechanics in the
prompt (measured: no change in length or pauses over three samples) and changing
how replies are split for synthesis (measured about 300ms per gap, inaudible to
him).

**Costs accepted:** slower first audio, to be measured live; the 48 hand picked ack
auditions no longer apply, so every clip was re rendered on v3 after backing the
flash bank up to `data/phrases_flash_v2_backup_2026-09-13`.

**Corrections made in the same change.** VOICE_OUTPUT.md said v3 rejects
`use_speaker_boost`; the API accepts it. It said leaving flash_v2 drops his name;
that was flash v2.5, and v3 keeps the tag. Both were inherited claims that had
never been tested on v3.

**Still open:** his name on v3, which he says needs correcting, and a personality
rewrite drafted and tested but not applied, deliberately held back so the voice
change could be heard on its own.

### Nova's personality rewritten to speak, not recite (Sep 13 2026) (DONE)

He described her as reading, not speaking, and wanted a professional who is a
little sarcastic but real, talking to him like someone she cares about. Tested
three ways on real questions before choosing; the mechanism is in
[BRAIN.md](BRAIN.md#nova-speaks-she-does-not-recite).

**Rejected:** rewriting sentence mechanics. Measured over three samples per
question with tools passed, length and pauses did not move.

**Chosen:** a persona that asks for care instead of composure, and a block that
makes her interpret results. Held back until the v3 voice change had been heard
on its own, so each change could be judged separately.

**Accepted risk, guarded:** interpretation can overreach the data, and warmth
can harden into stock phrases. Both showed up in testing and both have an
explicit line in the prompt. Watch for them in real use; they are the two ways
this change fails.

**Left alone:** `ABOUT_YOURSELF` still has her call herself "the most capable
presence in whatever room I'm in", which is the old persona. Out of scope for
this change and worth raising with him.

### Calendars split by Google's accessRole, not by a list of names (Sep 13 2026) (DONE)

He does not want club events read as plans. A list of calendar names to ignore
would need editing every time he subscribes to something. Google's own
`accessRole` already draws the right line: what he owns he committed to, what he
reads he follows. So a new club calendar lands on the right side untouched.
Followed events are kept, in their own section, because he subscribes to them
precisely so there is something to go to when he wants it.

### Several changes on one turn are one batch (Sep 13 2026) (DONE)

**Rejected:** refusing a second proposal on the same turn. It closes the defect,
but it forces a separate yes per lesson, which is what he found unusable when he
wanted seven lessons on the calendar to move around afterwards.

**Chosen:** proposals on the same turn accumulate, the question covers all of
them, and one answer confirms or cancels the lot. The guarantee that confirm runs
only what was asked now holds for the batch as a whole.

**Known limit:** each tool result carries the growing question, and the model is
told to ask the one from its last result. If it asked an earlier, partial one,
the batch would still contain more than he heard. A single tool that takes every
event at once would close that, and belongs with the scheduling rework.

### Long recordings get Whisper's full window (Sep 13 2026) (DONE)

`MAX_RECORD` 18 to 60, with the audio window chosen by clip length: fast up to 15
seconds, Whisper's full window beyond, pieces past 28. The measurements are in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#long-recordings).

**Rejected:** raising the window for every clip, which would slow the nine in ten
recordings under 8.2 seconds to help three in eighty three; and scaling the window
to each clip's length, already rejected in August for corrupting transcripts.

**The threshold is 15, not 19**, because the fast window was validated only up to
14 seconds, and at 18 it dropped trailing words from a real recording.

**The pieces are cut in code at the quietest point** near each boundary, tested on
synthetic audio. Not yet tested against a real recording over 28 seconds, because
none exists yet; the first one he makes is that test.

### Memory: store what he asks, notice more, never claim a save (Sep 13 2026) (DONE)

`remember` had never been called in a month. Measured before changing anything,
then after; the numbers are in [BRAIN.md](BRAIN.md#memory-nova-actually-remembers-now).

**Chosen:** a lower bar for inferred storage, because inferred memories already
wait for review, so a reasonable guess costs one "discard" while never noticing
anything costs everything. Explicit requests store first and ask after. The claim
of a save without a call is logged rather than blocked: blocking would need
rewriting her text, and the prompt already stopped it in all fifteen samples.

**Review by voice instead of waiting for the app**, limited to ids actually in the
queue.

**Deferred:** inferred corrections skipping review, which needs a schema change;
see BACKEND_TODO.md.

### Schedule conflicts found in code, suggestions left to Nova (Sep 13 2026) (DONE)

What he meant by clearing his schedule was seeing what overlaps, not hiding or
deleting anything. Overlap is arithmetic, so code finds it; which event to skip is
judgment, so Nova suggests it, with classes and his own commitments first. The
general session planner for tutoring and study blocks is next, built for sessions
of any kind rather than for tutoring, which is temporary.

### A busy room: the wake word interrupts, the cap is 30 (Sep 13 2026) (DONE)

What happened is in [INCIDENTS.md](INCIDENTS.md#a-busy-room-held-the-microphone-sep-13-2026).

**Cap 30, not 60.** Both recordings that ran past 30 that night were guests, and
his longest real request was 18 seconds. 30 seconds is about 75 words.

**Wake word during recording, on a second model.** Sharing the main model would
have overwritten the wake audio verification reads from its buffer.

**Rejected, on measurement:**
- Ending the recording once the speculative transcript is ready. He speaks
  through pauses over 750ms in 18% of turns; it would cut him off in one in five.
- Giving each sentence its neighbours for a steadier voice. ElevenLabs refuses:
  "Providing previous_text or next_text is not yet supported with the 'eleven_v3'
  model."
- Online speech to text for speed. Whisper takes about 250ms here, so a cloud
  round trip is slower; only a streaming service that also decided the endpoint
  could be faster, and the pause data above is the same risk. Its real value is
  robustness in a loud room, which is recorded in BACKEND_TODO.md.

### Calendar questions: fetch in parallel, remember the calendar list (Sep 13 2026) (DONE)

The calendar tool was 2.6 to 2.9 seconds of the slowest turns. Measured before
changing anything: 497ms listing calendars, then 1621ms for nine calendars asked
in turn. Now fetched at once and the list remembered for five minutes: 482 to
907ms. Measurements in [LATENCY.md](LATENCY.md#calendar-tools-sep-13-2026).

**Threads, each with its own client,** because the Google client's HTTP
connection is not safe to share. **Rejected:** Google's batch endpoint, one more
API shape to get right for a saving the threads already deliver.

**The remembered list is cleared when the MILES calendar is created**, the one
change Nova can make to it. A calendar he subscribes to by hand can take up to
five minutes to appear.

### The rest of a reply as one request, sentences joined by an ellipsis (Sep 13 2026) (DONE)

The first sentence is still synthesized the moment Claude writes it, so first
audio is unchanged. Every later sentence waits for Claude to finish and goes as
one request. The listening tests and measurements are in
[VOICE_OUTPUT.md](VOICE_OUTPUT.md#one-delivery-for-the-rest-of-a-reply-with-a-breath-between-sentences).

**Rejected:** the whole reply as one request, which he liked best but which adds
Claude's full generation time before any sound; stability 0.75, which v3 appears
to ignore; context between separate requests, which v3 refuses.

**Known risk:** a very short first sentence can finish playing before Claude has
finished the rest. Each turn logs how long the rest waited; if that shows real
gaps, grouping the first two sentences is the next step.

### Sessions are planned in code, and a spoken limit counts (Sep 13 2026) (DONE)

`plan_sessions` places several sessions around his week and proposes them as one
question. Built general rather than for tutoring, which is temporary: study
blocks and workouts use it the same way. The mechanism is in
[BRAIN.md](BRAIN.md#sessions-are-placed-in-code).

**Nothing about a student is hardcoded.** Earliest start and latest end are passed
per session from what he says, with 3 PM and 8 PM only as defaults.

**One session per name per day,** the simplest rule that guarantees he never
teaches one student back to back. **Rejected:** allowing a second same day
session with a gap, which he did not ask for and which the spread already avoids.

**Two bugs caught by a dry run on his real calendar before it was used,** both now
tested: spreading by fixed positions bunched two lessons into consecutive days
when the range's last day was unusable, and a block naming today's weekday was
read as next week.

### A proposed change is spoken by code, word for word (Sep 14 2026) (DONE)

The confirmation guarantee held for what was staged and not for what was said:
Nova staged the 21st, said the 14th, and his yes created the 21st. See
[INCIDENTS.md](INCIDENTS.md#his-yes-created-a-date-he-did-not-hear-sep-14-2026).

**Chosen:** when a turn stages a proposal, `brain.py` skips the model's reply and
speaks `pending_action.staged_question()` itself, including when the proposal
follows a lookup. What he hears is what runs, and each confirmation saves a
Claude call. **Rejected:** a stronger prompt line to ask the question verbatim,
which is what had already been in place and was ignored.

**Also:** a new event is read back as a time range; an all lowercase title is
capitalized; long lengths are said in hours; today's weekday with a time still
ahead means today, while a bare weekday still means next week; "next monday at
1pm" is refused, because dateparser cannot read it.

### Club events are avoided when there is room, not ignored (Sep 14 2026) (DONE)

He wants the planner aware of club events worth going to, like a Datadog info
session. The first plan treated every followed calendar as free time and put
lessons across a mini career fair, three info sessions and a workshop.

**Chosen:** club events are soft. Each session is placed clear of them if any day
allows it, and only overlaps one when none does, and the spoken question names
every forced overlap before asking. Nova is told to put a club event she knows
he should attend into `blocked`, which is never overlapped.

**Rejected:** treating every club event as a commitment, which on his real week
leaves most weekday evenings unusable for lessons; and picking important events
by keyword in code, which is judgment about his goals and belongs to the model.

### A name is fixed everywhere at once, and a homophone is spelled (Sep 14 2026) (DONE)

What went wrong is in [INCIDENTS.md](INCIDENTS.md#fixing-one-name-took-five-read-backs-sep-14-2026).

**Chosen:** a tool that renames every MILES event containing a word, with one
question, rather than teaching Nova to call the one event tool repeatedly, which
is what produced five read backs. A rename is spelled only when Soundex says the
old and new words sound alike.

**Rejected:** spelling every rename, which adds letters to every change that could
already be heard.

**Still open:** whether adding events should skip confirmation, proposed to him the
same night and not built without his answer.

### No restarts while he is talking to Nova (Sep 14 2026) (DONE)

Three deploy restarts landed inside one of his conversations and each swallowed a
turn; see [INCIDENTS.md](INCIDENTS.md#fixing-one-name-took-five-read-backs-sep-14-2026).
Before restarting `miles-voice`, check the journal for speech in the last two
minutes (`journalctl -u miles-voice --since -120s` showing "You:", "Wake word
detected" or "Listening for follow up"). If there is any, do not restart; say so
and let him finish.

**Spelling a sound alike rename says only what changed**, "E Y instead of I E",
after he found spelling the whole name too much. A change longer than four letters
is still spelled whole.

### Additions happen at once, with undo; changes still ask (Sep 14 2026) (DONE)

**Supersedes part of "Permission gate enforced, and outside writes wait a turn"**,
which confirmed every EXTERNAL_WRITE. He said twice that scheduling had too much
confirmation.

**Chosen:** `create_calendar_event` and `plan_sessions` add at once and code reads
back exactly what was added; `undo_last_addition` removes the most recent addition
as a whole, one event or every session of a plan, for thirty minutes. Moving,
renaming and deleting still confirm on the next turn.

**Why the line is there:** an addition that is wrong costs one sentence to undo and
destroys nothing. A move or a delete overwrites what existed, and undo cannot put
back an event from someone else's invitation or a detail he typed by hand.

**What protects a misheard addition now:** the read back is spoken by code, so he
hears the real day and time, and undo is one sentence away. The permission gate is
unchanged; only hokage can add.

### A typed message is read, not spoken (Sep 15 2026) (DONE)

Every message typed in the app was spoken aloud through the room speaker by
`miles-server`, and `/chat` returned only when playback finished, which the app
showed as Nova still thinking while she was already talking. The mechanism and
the measurements are in [BRAIN.md](BRAIN.md#channels).

**The claim that this could not happen was in the repo, twice.** BRAIN.md said
channel selected the prompt fragment and gated pronunciation, and `b943ea0` said
"the text path never calls speak(), so there is no condition to get wrong". That
was structurally true when it was written and stopped being true when streaming
TTS moved into `ask_nova_async`, which speaks on every turn whatever the channel.
Both corrected in place rather than deleted.

**Chosen:** a consumer chosen by channel, `_collect_text` against
`_tts_consumer`, with the same arguments, rather than an `if` around each of the
four places that made sound. The fire and forget branch reads what the consumer
collected, so one swap keeps both channels identical downstream. The test carries
a voice control, because "nothing played" passes just as well when the fakes
stopped seeing speech at all.

**Numbers were two problems, not one.** Examples beat the rule, and the
transcript beat the system prompt. Both are the same lesson this repo has now
recorded three times: what Nova is handed outweighs what she is told. Fixed with
derived text copies of the two example carrying blocks, and a note on the final
user turn, after the cache breakpoint.

**`/chat/stream` was added rather than fixing `/ws`.** The socket handler is
`async` and calls `asyncio.run` inside the running loop, so it has never worked;
nothing uses it. SSE keeps one request with the same auth header, and the turn
runs on a worker thread so a client that goes away cannot take down a turn that
is already writing to the database.

**Not fixed, deliberately:** `WHAT_REACHES_YOU` tells Nova every message is
speech recognition output, which is false when he types. Out of scope for a
numbers fix and it changes how she treats a strange looking message, so it is in
BACKEND_TODO.md rather than in this change.

### Services bind loopback, so exposure is not the network's to decide (Sep 15 2026) (DONE)

Forced by the Sep 14 incident:
[INCIDENTS.md](INCIDENTS.md#one-deauth-cost-21-hours-offline-and-9-hours-exposed-sep-14-2026).
Ethernet recovery handed the Pi a routable public address with no NAT, and
sshd, n8n and uvicorn became internet facing without a single config change.
CGNAT had been acting as a firewall that nobody chose, configured or checked.

**Chosen:** every service that does not need to be reachable off the box binds
loopback. n8n to `127.0.0.1:5678` in the compose port mapping, uvicorn to
`localhost` in `miles-server.service`.

**A firewall was rejected as the primary control.** nftables or ufw would have
worked, and on the day it mattered `iptables -P INPUT` was `ACCEPT`. A firewall
is a second thing that has to be correct, installed and running, and it fails
open. A bind address is a property of the process itself and cannot stop being
true because a cable changed. Rate limiting and a firewall are still worth
having, but as depth, not as the thing being relied on.

**`localhost`, not `127.0.0.1`, and the difference is load bearing.** asyncio
resolves the hostname and creates one socket per `getaddrinfo` result, so
`localhost` binds both `127.0.0.1:8000` and `[::1]:8000` while the literal
address binds only the first. cloudflared routes to `http://localhost:8000` and
may present either family, so the literal address is a coin flip that happens to
be landing right. Verify both listeners after any change to that line.

**Accepted cost:** nothing off the Pi reaches the API directly any more. The
Nova iOS app must go through example.invalid; `miles.local:8000` now
refuses. That is the intent rather than a regression, and it is the part a
future session is most likely to want to undo. Do not undo it by widening the
bind.

**Tailscale closes the half of this that binding does not touch**, namely that
recovery required a publicly addressable machine at all. Installed on the Pi
Sep 15 2026 and joined as `miles` at `192.0.2.10`. It was reported as
applied several hours before it was true, because the install had been done on
the MacBook; see
[INCIDENTS.md](INCIDENTS.md#one-deauth-cost-21-hours-offline-and-9-hours-exposed-sep-14-2026).

**Tailscale SSH was deliberately not enabled.** `tailscale up` ran without
`--ssh`, so SSH remains governed by the sshd config hardened above rather than
by tailnet ACLs. Two doors with different locks is worse than one, and the
sshd side is the one with `AllowUsers` and password auth off. Turning it on
later is a decision, not a default.

### sshd hardening wins on sort order, not by editing the generated file (Sep 15 2026) (DONE)

`PasswordAuthentication` was `yes` with no rate limiting and no user allowlist
while port 22 faced the internet for nine hours. Roughly 13,000 attempts, all
failed. Root survived on `PermitRootLogin without-password`; `the owner`
survived because no botnet guessed the username, which is luck rather than
defence.

**Chosen:** `/etc/ssh/sshd_config.d/00-hardening.conf` containing
`PasswordAuthentication no`.

**Editing `50-cloud-init.conf` to say `no` was rejected.** It sets
`PasswordAuthentication yes`, and it is generated. cloud-init may rewrite it,
and the revert would be silent, at an arbitrary future date, on the setting that
matters most. Winning on sort order is durable in a way that editing a generated
file is not.

**Putting it in `sshd_config` itself was also rejected**, and this is the part
that reads backwards until the rule is known. The `Include` sits at **line 12**,
the top of the file, and **sshd takes the first value it finds and ignores every
later one**. There is no last write wins. A value in the main file would look
authoritative and be dead, beaten by the drop in directory that is read before
it.

**`AllowUsers the owner` is the exception**, and it lives in
`/etc/ssh/sshd_config` line 59. That works only because nothing in the drop in
directory sets it, so first occurrence still finds it. Someone will eventually
read those two facts side by side and think one of them is wrong. Both are true.

**Read the effective config with `sudo sshd -T`, never by reading a file**, and
validate with `sudo sshd -t` before reloading, keeping a second session open.
The full rule and the verification command are in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md#ssh-configuration).

### wlan0 recovery is an external timer, not NetworkManager's own retry (Sep 15 2026) (DONE)

NetworkManager did not fail. It read a deauth arriving mid handshake as a
rejected pre shared key, entered `no-secrets`, found no agent to ask on a
headless machine, marked the connection failed and stopped. The logic was sound
and the premise was false, and it never retried across 21 hours.

**Chosen:** `miles-wifi.timer` every two minutes, running a script that checks
carrier and a global scope IPv4 address and runs `nmcli connection up Alsander`
when either is missing.

**Relying on NetworkManager autoconnect was rejected**, because a wrong password
is not a condition that retrying fixes and NM was correct to stop. The recovery
has to come from something that does not care *why* the link is down. That is
the whole design: the watchdog never diagnoses.

**A dispatcher script was rejected** for the same reason in a different shape.
Dispatcher scripts fire on state changes, and the failure here was the absence
of any further state change. Nothing would have triggered it.

**It checks link state, not reachability, deliberately.** Adding an internet
probe would make it bounce a perfectly good link during an upstream or ISP
outage, turning someone else's problem into a dropped connection of our own.
Reachability is `netcheck`'s question and it answers it on the turn that failed.
Two components, two questions, no overlap.

**Shell rather than Python**, unlike everything else in `scripts/`, because the
boot where this matters most may be the boot where the Python environment is
itself broken. **Root rather than `the owner`**, unlike `miles-health`,
because `nmcli connection up` is polkit protected and a non root caller is
refused on the one occasion it has to work. **Silent on a healthy link**,
because systemd already writes a Starting and a Finished line per run and seven
hundred daily lines saying nothing happened would bury the ones that matter.

### Every systemd unit is versioned, and the copy is checked rather than trusted (Sep 15 2026) (DONE)

`miles-server`, `miles-voice` and `miles-tunnel` existed nowhere but `/etc`,
so a reinstall would have rebuilt three production services from memory. All
five units are now in `systemd/`.

**Symlinking `/etc/systemd/system/*.service` into the repo was rejected.**
systemd resolves symlinks and it would have kept the two identical by
construction, which is exactly the appeal. It also means a checkout, a rebase or
a permissions change silently alters production, and the repo is not root owned.
A copy that can drift and is checked beats a link that cannot drift and can be
moved by `git`.

**So the drift is accepted and made visible instead.** Change the repo copy
first, then install it, and verify with the loop in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md#systemd-services). Editing under `/etc`
and forgetting to copy back is the failure this invites, and the loop is what
catches it.

### A bare hour is a clock time, and a change reaches across the week (Sep 16 2026) (DONE)

What broke is in [INCIDENTS.md](INCIDENTS.md#moving-two-lessons-took-eight-turns-sep-15-2026).

**Chosen:** read which half of the day from the event being moved, because the
staged read back catches a wrong guess in one sentence. **Rejected:** refusing
any time without am or pm, which is how people actually talk about their week
and would have added a turn to nearly every move.

**A change uses the one matching event in the week around the named day.**
**Rejected** for delete: a delete that lands on a day he did not name destroys
something he may not have meant, while a move is proposed and read back first.

### remember returns to Nova (Sep 16 2026) (DONE)

**Supersedes** the Aug 11 reasoning that a round trip to announce a save was
latency for a sentence nobody asked for. That assumed she would already be
answering alongside the call; measured, she did so in half the turns and in none
where he asked outright. See
[INCIDENTS.md](INCIDENTS.md#his-news-was-answered-done-sep-15-2026).

**He chose a prompt only fix first**, on my recommendation. It measured worse
(1 of 8) and was dropped with his agreement. **Chosen:** the tool result carries
the instruction to answer, and a follow up after only `remember` is made
without tools. 7 of 8 through the real turn.

### Moves and deletes happen at once, with undo (Sep 16 2026) (DONE)

**Supersedes** "Additions happen at once, with undo; changes still ask" (Sep 14).
He said moving and deleting still took too much work.

**The reason for the old line was an undo that only knew how to delete.** A
move is fully reversible by patching back what it changed, and a deleted MILES
event can be re-inserted from a saved copy; only Google's id changes. So the
question before each one protected nothing undo does not, and cost a turn.

**Still asked:** one occurrence of a repeating event, which undo would bring
back as a standalone event; and renaming every event that shares a word, where
the spelled read back is what he wanted.

**Worth being honest about:** most of the Sep 15 pain was not confirmation. Of
eight turns, one was his yes; the rest were the parsing defects fixed the same
day.

### Timers are rows (Sep 16 2026) (DONE)

A timer set from the app never went off, because it slept in `miles-server`,
the same bug reminders had until Sep 6. **Chosen:** the reminders table with a
`kind` column, fired by the same poller, rather than a second table and a second
poller. **The poll interval dropped** because a timer is set to the second;
each pass is one indexed query on a small local file.

### App endpoints (Sep 16 2026) (DONE)

Memory edits supersede, so history survives an edit made with a thumb. Calendar
tap edits are MILES only, enforced by where the event is fetched from, and stay
out of Nova's undo, which is grouped by conversation turn. Typed titles are kept
as typed; `_title` exists because transcripts arrive lowercase.

**Not a people tab.** He asked how people work. The `people` table is for
tiers and birthdays, filled by hand with `scripts/people.py`; Nova cannot write
to it and no endpoint exposes it, so mentioning someone never adds them. Facts
about people, like a student's pronouns, are memories, and a temporary one can
carry an end date.

### A named weekday stays in the event's week; a listing shows what is over (Sep 16 2026) (DONE)

Found while fixing the two Andrew lessons; see
[INCIDENTS.md](INCIDENTS.md#moving-two-lessons-took-eight-turns-sep-15-2026).

**Chosen:** the nearest occurrence of the named day, because that is what a
person moving an event means. **Rejected:** always forward, which is how it
was, and always backward, which fails for "move Monday's to Friday".

**The listing still starts at now.** That rule came from Nova reading past
sessions as upcoming, and it stays. What changed is that what is over is named
separately, and only for the last day, so it cannot become a history dump.

### Noticing is its own call (Sep 16 2026) (DONE)

Measured in [BRAIN.md](BRAIN.md#a-separate-pass-notices-what-he-mentions).
**Rejected:** a stronger prompt line, which went from 4 to 7 of 24 and still
missed every fact said alongside a task. **Rejected:** Sonnet 5 for the pass,
which wrote cleaner memories and caught fewer, at twice the cost. **Chosen:**
Haiku on a background thread, pending only, never superseding. A correction he
mentions in passing therefore arrives as a new pending memory beside the old
one, not as a replacement; the schema change that would let an inferred
supersede wait for review is still in BACKEND_TODO.md.

### Latency toward two to three seconds, one change at a time (Sep 16 2026) (IN PROGRESS)

He wants two to three seconds, at least perceived. Baseline in
[LATENCY.md](LATENCY.md#the-turn-budget-measured-sep-16-2026): 3857ms on plain
turns, 5258ms on tool turns, which are half of all turns. Four changes, each
shipped and measured on its own:

1. **Verification beside the speculative transcript** (DONE, awaiting live
   numbers). Benchmarked first: running both costs whisper about 36ms.
2. **An instant spoken bridge when a tool starts** (DONE, awaiting live
   numbers). Pre rendered, played from disk, timed as its own column so the
   perceived series is not broken. Reverses the Aug 11 "no acknowledgment
   during tool execution", which was waiting on exactly this data.
3. **Ending the recording early on a finished sentence** (REJECTED). Simulated
   on 285 archived recordings it was close to safe, but the transcript it
   depends on arrives after today's cutoff, so it cannot save anything.
   **Replaced by starting the speculation sooner** (DONE, 450 to 210ms), which
   takes the transcription wait instead and never moves the end of a recording.
4. **Starting Claude during the endpoint wait** (DROPPED). The same wall: no
   transcript exists during the wait. Warming the prompt cache at the wake word
   was considered instead and dropped too: since Sep 13, cache misses were not
   slower (time to first token 850ms on 10 misses, 1121ms on 38 hits).

**Where that leaves a plain turn:** roughly 1.2s of endpoint, a few tens of ms
of transcription and verification, about 1.1s to Claude's first token, 0.2s to
a sentence, and 0.6s to eleven_v3's first audio, so about 3.1 to 3.3s. The
remaining levers are his: the voice model, and how long a pause means he is
done.

**Not changed:** flash_v2 would save about 250ms to first audio, but eleven_v3
was chosen by ear, and that is his call.

### The room speaker follows `device`, not `channel` (Sep 23 2026) (DONE)

He asked for a speaking mode in the iOS app, with Nova's real voice rather than
Apple's, so she sounds like one person everywhere. Two things had to change.

**`POST /speak`.** Every synthesis path in the repo ended at `aplay`, so the app
had no way to get audio at all. The route hands a finished reply to ElevenLabs
and streams mp3 back. It takes a finished reply rather than sentences on purpose:
eleven_v3 voices each request on its own, so a reply synthesized piecewise comes
back sounding like it changed speakers partway through, which is the exact thing
he asked to avoid. Details and the measurement are in
[VOICE_OUTPUT.md](VOICE_OUTPUT.md#the-app-asks-for-audio-of-its-own-sep-23-2026).

**Playback gated on `device`.** This is the part worth not reopening. `channel`
decided both how a reply was written *and* whether the Pi spoke it. That made
the app's two options "formatted for a screen, silent" and "formatted for speech,
played out loud in his room", and Talk Mode needs the third: formatted for
speech, silent here, synthesized on the phone. `_plays_in_the_room(channel,
device)` now answers the second question, and only `"pi"` is in the room.

Two things had already recorded that this split was intended.
`ask_nova_async`'s docstring said since Sep 14 that "the app can want spoken
output", and the Channels section of [BRAIN.md](BRAIN.md#channels) carried a
Sep 14 correction about typed messages being spoken aloud. That correction fixed
the case where the app sends `channel: "text"`. It did not fix the default:
`ChatRequest.channel` defaults to `"voice"` and the app sends no channel, so in
practice app messages were still being played in his room. The intention was
written down twice and the code still conflated the two.

**The tests caught the change, which is what they are for.**
`tests/test_text_channel.py` hardcoded `device="app"` while asserting the room
speaks, so three voice tests failed the moment playback stopped depending on
channel alone. The harness now takes a device and defaults to `"pi"`, because
most of those tests are asking what the room does. Two tests were added for the
new guarantee: the app on the voice channel is silent here, and it gets no
spoken bridge either.

**Not done here:** the app side. The Swift repo has to call `/speak` and play
what comes back, and `NOVA_APP_BACKEND.md` in that repo still says the backend
speaks every chat turn because TTS "is not gated on channel". The gating claim
was already wrong and is now wrong twice over.

### An unrecognized voice is a guest, not refused (Sep 26 2026) (DONE)

He wants friends to be able to ask Nova things, with her knowing it is not him
and gating anything personal. An unrecognized voice used to hear "I don't
recognize your voiceprint" and nothing else.

**Chosen:** a rejected verification runs the turn at genin with `GUEST_BLOCK`.
The retry band still asks him to repeat, because a near miss is usually him. A
guest sees only the conversation since its wake word (`get_messages_after`),
never his history; the guest prompt says those facts are not in front of her,
and before this change they were, in the history. Local intents run for a guest
only if their tool is allowed at genin, since they skip the permission gate.
Whose conversation it is follows the latest scored turn, and only his is
trusted on the session, so a guest's short follow ups stay a guest's. Guests
also get `NOT_ADDRESSED_TO_YOU`, because a false wake on friends talking is a
guest turn now.

**Why this needs ECAPA:** under Resemblyzer nearly everyone verified as him (EER
21.5%), so guest mode only became real with the encoder switch, done Sep 27 2026
with a fresh enrollment. Against the enrollment voiceprint, 27 labelled impostor
clips scored max 0.088 and his clips median 0.436; the threshold is 0.15.

### Prowl is relabeled on the Pi, never recomputed (Sep 25 2026) (DONE)

`prowl_tools.py` gives Nova the job search. Three choices here would otherwise
be reopened.

**Rename, do not recompute.** Prowl computes every count, filter and day
difference. The Pi renames deadline fields by `deadline_source` and turns
meaningful nulls into words, and nothing else. The alternative, passing Prowl's
fields through and telling the model in the description to read
`deadline_source`, was rejected: a description is a hint the model can miss,
and with 77 passed dates of his own and no posting dates, one miss is "you
missed 77 deadlines". A key named `your_own_target_date` cannot be misread as a
closing date. The reasoning is in
[BRAIN.md](BRAIN.md#prowl-deadlines-are-named-by-whose-they-are).

**Caps on the Pi for now.** 15 applications and 10 discoveries, cut at Prowl's
order and flagged, not counted. The right fix is a filter on Prowl's side;
deferred in [BACKEND_TODO.md](BACKEND_TODO.md#prowl-name-filter-and-limit).

**One accept per question, enforced in code.** `pending_action.propose` merges
one turn's proposals under a single yes. Rather than change that for every
caller, `accept_discovered_job` refuses when anything is already staged that
turn. Prowl's contract forbids bulk accept, and this is what makes it true.

### Duplicate memories are never merged automatically (Sep 28 2026) (DONE)

He asked for closely related memories to be grouped. Measured on real pairs,
similarity cannot tell a duplicate from two different facts about the same
person (0.79 to 0.94 against 0.76 to 0.83), and Haiku as the judge called two
in three different facts the same or an update. Wired in, it would have
overwritten "Auntie Jennifer hosts Thanksgiving" with the memory that she is
married to Uncle Marlo.

**Chosen:** he decides every merge, from the app or `memory.py similar`, with a
drafted sentence to edit and a keep both that is remembered. The only
automatic step is the memory pass skipping a fact at 0.9 or more from one
stored or waiting, where only rewordings have been seen, and never a fact he
asked her to keep. Reopen only with a judge that beats the table in
[BRAIN.md](BRAIN.md#duplicates-are-not-merged-automatically-measured-sep-28-2026).

### The XVF3800 is the mic, and the speakers stay off it (Sep 30 2026) (DONE)

The array replaced the Seiren after scoring higher than it on the same takes
(10 of 12 against 7 of 11). The voiceprint was rebuilt on it the same day,
because the two mics' voiceprints agree at only 0.738.

**Speakers were not moved to the array's jack**, though that is the documented
way to feed its AEC. Its playback is 16kHz only and sounded clearly worse by
ear. Barge in by wake word needs no AEC, and full AEC may work by sending the
array a reference copy while the AB13X plays. Reopen only if that fails and
barge in on any speech becomes worth the voice quality. Numbers in
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#the-respeaker-xvf3800-replaced-the-seiren-sep-30-2026).

### Barge in stays off after a day on the array (Sep 30 2026) (DECIDED)

Tried on both channels, with and without an echo reference, at two double talk
settings, and finally through the production path with a fixed clip: a wake
Deepgram transcribed clearly scored 0.00. Off rather than left on and unproven,
with the fixes kept. Reopen with speech detection on channel 0 first, the route
the recordings support, then a retrained wake model; the evidence is in
[the BACKEND_TODO archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#barge-in-on-the-array-tried-sep-30-2026-off).

### Speech on channel 0 cannot stop her; the wake model is retrained next (Sep 30 2026) (DECIDED)

The route the entry above chose to try first was measured and fails. Once the
echo reference was fixed to run for the whole reply, the canceller still left
her at up to minus 3 dBFS, as loud as him, and webrtcvad called half or more of
her frames speech; direction of arrival stayed on her while he talked. Built,
it would cut her off on nearly every reply. **Chosen, in his order:** retrain
`hey_nova.onnx` against an evaluation set recorded over her playback, then try a
lower speaker volume. Reopen speech detection only if a volume or enclosure
change gets her residual reliably under his speech; the numbers are in
[the BACKEND_TODO archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#barge-in-on-the-array-tried-sep-30-2026-off).

### The speakers move to the array's jack, on the 48kHz firmware (Oct 1 2026) (DECIDED)

Reverses "the speakers stay off it" above. Both of its reasons fell the same
night: the reference over a second USB device does not work, because the two
clocks slide her echo 1 to 3ms against the reference within a reply and the
canceller removed about 1 dB; on the array's own jack it reached 14 to 15 dB.
And the 16kHz playback that sounded degraded is gone on `v2.1.1_48k2ch`, which
he judged by ear to sound like the AB13X. Blocked only on resampling the mic
from 48kHz, which the firmware forces on capture too. The numbers are in
[the BACKEND_TODO archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#barge-in-on-the-array-tried-sep-30-2026-off).

### Any speech stops her, his or anyone's, at -37 dBFS on channel 1 (Oct 2 2026) (DONE)

Barge in fires on 300ms of speech louder than the floor on `BARGE_IN_CHANNEL`,
beside the wake word. **Whose speech it is is not checked**, chosen with him:
a voiceprint needs about two seconds of speech, so checking first would mean
her talking over him for two seconds, which defeats interrupting. Who it was
is decided afterward, on the whole interruption, as any turn is; a busy room
costs her the rest of a sentence, and the ignore tool keeps her from answering
it. The floor came up from -40 to -37 when people talking outside the room
stopped her 7 times; -37 kept all 29 of his takes and none of the rest, with
about 1dB of margin each way. Numbers in
[the BACKEND_TODO archive](archive/BACKEND_TODO_ARCHIVE_2026-11-28.md#barge-in-on-the-array-tried-sep-30-2026-off).
**DIVERGENCE:** the plan had `watch_for_interrupt` return why it stopped. It
runs as a thread target, whose return value is discarded, so the reason went to
the log and the audio to a carrier object instead.

### An interruption is recorded on channel 1 and never teaches the voiceprint (Oct 2 2026) (DONE)

Measured before building: sentences read over her came through channel 1 with
5.2 percent of words wrong against 64.4 on channel 0, and verified at a median
0.505; in a quiet room the two channels scored the same. So the whole
interruption, before and after she stops, is one channel. It is still kept out
of the voiceprint samples: the voiceprint was enrolled on channel 0, her voice
is under his, and the store has been poisoned once. Her history keeps only the
words he heard, marked "[You were cut off here.]".

### The wake note says what was heard, and she ignores more than she used to (Oct 1 2026) (DONE)

After she answered his phone call three times
([INCIDENTS.md](INCIDENTS.md#she-answered-his-phone-call-three-times-oct-1-2026)),
four wordings were replayed over 30 of his turns. The chosen one stays silent
on room talk 44 of 63 times (16 before) and answers 22 of 24 meant for her (24
before): a garbled reply to her ("Actually, never mind oranges") is sometimes
taken for room talk. **Chosen knowing that cost**, because missing a reply
costs him a repeat and answering room talk talks over him. A firmer wording
reached 56 of 63 and ignored "Is Trump a good president?", the Sep 26 case, and
was rejected for it. Any change to this wording is replayed first
(`scripts/replay_ignore.py`); see [BRAIN.md](BRAIN.md#the-ignore-tool).

### The wake threshold stays at 0.4 on the array (Oct 2 2026) (DECIDED)

His roommates hear the chime on false wakes. On his labels no threshold
separates his wakes from the false ones: 0.55 cuts 6 of 8 false wakes and
misses 3 of his 18, and a missed wake is worse than a stray chime. A score and
level rule is scored offline (`scripts/wake_rule.py`) and acts only once more
labelled days, his other spots included, say it costs him nothing. The real fix
is retraining the model; see
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#on-the-array-oct-2-2026-still-no-threshold).


### What a cut reply had left to say is kept (Oct 30 2026) (DONE)

`resume.py`, `_Heard.remainder`, a record on `StreamRouter`, `words_unheard`.
This was built first, kept and not spoken, and the two entries after it say it.
Tests are in `test_resume.py`.

**The record is taken at the router, not the consumer.** The consumer stops
reading the moment he cuts in and its feeder is cancelled holding whatever it
had collected, so the sentences after the cut were in nobody's hands, and Claude
is usually still writing them. The router sees every sentence in order whether
or not anything reads the queue. **Rejected:** draining the queue after the cut,
which makes the consumer wait for Claude to finish and adds a hang to a path
that used to end at once.

**Indexed by count, not by matching text.** `_Heard.handed` counts sentences
given to the speaker; the remainder is the unheard words of the cut sentence,
then `record[handed:]`. **Rejected:** subtracting the heard text from the whole
reply, which breaks on emotion cues and on the ellipses that join the rest of a
reply for synthesis. The cut point is one function (`_split_heard`), so what
history says she said and what is held as unsaid can never overlap or leave a gap.

**It is held for 120 seconds and said once.** `take()` clears it, so one "go on"
cannot repeat the speech. The window matches `CONFIRM_WINDOW_S`: long enough to
answer the door, short enough that a "go on" later is not read as finishing a
thought nobody holds.

**What lets go of it.** Another room reply that plays through. **What does not:**
a typed turn, a turn from the phone, and an ignored turn. The first two do not
answer the reply he interrupted. The third was room talk she stayed out of, and
that talk may be what cut her, so clearing on it would throw away exactly the
remainder an empty recording is about to need.

**Tool turns.** Both routers share one record. A cut during the lead in keeps
the follow up answer that was collected and never played.

**Where it starts was open, and is decided below** ("Where a resume starts").
A cut lead in is the same case: "me check. It is sunny" is a fragment before an
answer, and the rewind covers it.

### Where a resume starts (Oct 30 2026) (DONE)

`resume.speech()`, `resume.keep(text, said=...)`, `_Heard.said()`.

**Not the cut word, and not the sentence start.** The cut word is a fragment
("count alone matters little"). The sentence start can mean repeating twenty
words he already heard. The rule sits between: back up to a clause boundary
within a few words of the cut, so he hears the thought begin and nothing more.

**A boundary is** the word after a comma, a full stop, `?`, `!`, `;` or `:`, or
the words "and", "but" and "so", or the end of what she said when she was cut
just after one of those marks. **The window grows with the gap since the cut:**
4 words up to 10 seconds, 6 up to 30, 8 up to 60, 10 beyond (`REWIND_BY_GAP_S`).
A longer silence means he has lost more of the thought. With no boundary in
range it backs up the whole window. **No reordering and no model call:** the
words are hers, in order, and only the starting point is chosen.

**What is kept to make that possible.** `keep()` stores her last
`SAID_KEPT_WORDS` (10) words, the most a resume can ever back up. Nothing earlier
is needed, so nothing earlier is held. **Rejected:** keeping a flag for whether
those words are all of her reply, to treat its first word as a sentence start. The
fallback lands on the same word either way, so the flag decided nothing.

**The lead in rotates** through "Anyway,", "So,", "As I was saying," so two
resumes in a row do not open alike, and skips one that would double the first
word it introduces (no "So, so"). With nothing said before the cut there is no
lead in: she has nothing to pick back up.

**The gap is measured from when the remainder was kept**, which is when her turn
ended, seconds after the cut. That is close enough for a window measured in tens
of seconds.

### A stop that came to nothing resumes by itself (Oct 30 2026) (DONE)

`brain.speak_what_was_left()`, `voice_main.resume_after_empty_stop()`.

**"No real speech" is the two checks the loop already had, not a new one.** A
transcript `is_noise_transcript` rejects, or `verify_voice` returning `NO_AUDIO`
(too little voiced audio). Both were already "still listening" branches for a
recording that follows a stop. The second answers the worry about whisper: when
Deepgram fails, whisper can write plausible words over silence, so the
transcript looks real, but the voiced time is still tiny and `NO_AUDIO` fires.
**Rejected:** a new word count or confidence test, which would be a third
definition of nothing alongside these two.

**She speaks it through the reply consumer.** `_tts_consumer`, so the first
sentence starts at once and he can cut in on it. If he does, what is still
unsaid is held again from what that attempt got through, with a fresh `said`.
**Rejected:** `tts.speak(line)`, which plays all of it as one request (a long
wait for first audio) and cannot be interrupted, so a second tap would cost him
the rest of the answer, the exact loss this change exists to prevent.

**No Claude call.** They are her own words, and a call would cost latency and
let her rephrase instead of finish.

**Her history is amended; see "Her history after a resume" below.**

### "Go on" is a local intent (Oct 30 2026) (DONE)

`local_intent._gate_go_on`, `voice_main._run_go_on`, `resume.held()`.

**Resume only when "go on" is all he said.** The gate matches the whole
normalized transcript against a handful of phrases. "Go on and set a timer" and
"go on, what about Friday" are new requests, and a new request is answered, not
used as permission to finish the old reply. It was the
suggestion in BACKEND_TODO, and he asked for it as the rule.

**It declines when nothing is held**, checked with `resume.held()`, which does not
take the remainder: classifying must not use it up, because a gate that spends
what it checks would make the second classification of the same words fail.
Declined, the words reach Claude, who has the conversation and can ask what he
means.

**It is not executed by `execute()`.** Every other intent returns a fixed sentence
for `_run_local` to play. Finishing a reply has to go through the reply consumer
so he can cut in on it, so `run_turn` hands `go_on` to `_run_go_on`. If nothing
actually played (the window ended between classify and speaking, or synthesis
failed) it returns None and Claude answers, rather than leaving him in silence.

**History is written for this one.** "go on" and what she actually got through are
saved, for the reason `_run_local` gives: otherwise the next turn sees a "go on"
whose subject it never saw. If he cuts in again the saved text ends in the cut off
mark, as any cut reply does. This closes the history gap for a spoken "go on";
the automatic resume after a stop to nothing still leaves it open, since it has
no turn of his to attach to.

**Guests cannot use it.** Local intents skip the permission gate, and
`_local_allowed` let anything unregistered through. What she had left may be his
(she was reading his calendar when cut off), so `go_on` runs locally only when
`tier is None`, a verified him. A guest's "go on" goes to Claude at the guest tier.
The automatic resume after a stop has no such check and does not need one: it
finishes what she was already saying aloud in the room.

**Not verified on the Pi or by ear.** The embedding half is stubbed in the tests,
as in `test_local_intent.py`; only the gate and the wiring are asserted. Whether
real "go on" transcripts clear `INTENT_SIMILARITY_THRESHOLD` against the four
examples has not been measured.

### Her history after a resume (Oct 30 2026) (DONE)

`database.extend_message`, `resume.keep(message_id=...)`,
`speak_what_was_left(amend_history=True)`.

**Why it mattered.** The cut reply is saved as what he heard plus
"[You were cut off here.]", and the trimmer keeps that mark on purpose: it is why
his next message reads the way it does. After the automatic resume finished the
sentence, nothing touched that row, so the next turn's Claude believed he never
heard the end and could say it again or ask if he wanted it.

**The cut reply's own row is extended in place.** `keep()` stores the id
`save_message` returned for it; after she speaks, the mark comes off and what she
said is appended. Fully heard, the row reads as one reply with no mark, which is
true. Cut in on again, it ends in the mark again, and the next turn still sees
where. The row is amended by id, not "the latest row", so a typed or phone turn
saved in between cannot be overwritten.

**It is the literal transcript.** Replayed words and the lead in are in it
("...breeze and Anyway, and seventy..."), because that is what was said aloud.
The trimmer cuts a reply to thirty words anyway. **Rejected:** a second assistant
row (the mark stays on the first, so it still reads as cut, and two assistant
messages in a row is an odd shape); a "[resumed]" note (Claude would need teaching
what it means, where removing the mark says it directly); a cleaned "heard plus
remainder" text (claims words she did not say).

**A spoken "go on" does not amend.** It saves its own user and assistant rows, so
history already reads cut off, "go on", continuation, which is coherent.

**A failed amend is printed and swallowed.** It runs in the voice loop outside
`run_turn`'s boundary, and a locked database is not worth a restart.

**Not verified on the Pi.** The tests use a throwaway database and the real
trimmer; nothing has run against the live `miles.db`.

**Not verified by ear.** Nobody has heard it. The tests check which words come
back, not whether the phrasing sounds natural through the voice. The rotation
state and the window steps are first guesses, and the place to tune them is a
real cut on the Pi.

**Not verified on the Pi.** The tests run against fakes. The suite in the cloud
checkout is 1364 passing, 4 failing and 6 erroring, and all ten are missing
pieces of that checkout (pyaudio, and the phrase bank WAVs in gitignored `data/`),
the same ten before this change. The Pi's own count in CLAUDE.md has not been
re run, so it is left as it was.

### Push and phone health, written without a phone (Oct 30 2026) (DONE, UNVERIFIED)

The paid Apple Developer Program started today. Reminders and timers now also go
to his phone through APNs (`push.py`, migration 35), and the Nova app copies
Apple Health to the Pi (`health_tools.py`, migration 36). The mechanics are in
[INFRASTRUCTURE.md](INFRASTRUCTURE.md#push-notifications-oct-30-2026).

**Status is "written and tested against fakes", not "works".** No code ran
against Apple, and the Swift in `miles-app` was never compiled, because the
session had no Xcode. The first run on the phone is the test.

**Health sync is foreground only, deliberately.** Background delivery would wake
the app with an expired token and no way to renew it without Face ID. Building it
anyway would have produced code that looked finished and mostly could not run.
The design that fixes it is in BACKEND_TODO.md.

**The team id is not read from the Xcode project.** It holds whatever team the
app was last signed with, and enrolling in the paid program can change that. It
goes in `.env`, with the key id and the `.p8` path.

**Fields say `iphone`.** Oura already reports steps and active calories. Two
sources for one number, with nothing in the name to tell them apart, is how a
model quotes one as the other.

**Caught by the suite, both mine.** A new tool needs an entry in
`brain._STAGE_WORDS`, or the app shows nothing while it runs; the test that
requires it fired. And a test that changed `TZ` leaked New York time into the
session planner tests, because `time.tzset()` is what applies it and restoring
the variable does not call it again. Both are fixed; the second is the kind of
failure that shows in a different file from its cause.

### A device token, and the geofence in the app (Nov 2 2026) (DONE, LOGGING ONLY)

The plan for "wakes while he is away" had the phone report home or away through
an iOS Shortcuts automation. **Built in the app instead**, with CLMonitor, now
that the paid developer program is in place: no automation to set up by hand, and
one place that owns the code.

**The blocker was authentication, not location.** A crossing fires with the phone
in a pocket, long after the hour a login lasts, and renewing takes Face ID. The
same wall had already stopped background Health sync. Chosen: a 180 day token
scoped to `device`, accepted only by the background write routes. **Rejected:**
a longer login (one stolen token would then read everything for months), and an
unauthenticated presence route (anyone could tell Nova he was away).

**The scope check is the whole design,** so `get_current_user` refuses any token
carrying a scope at all, not only `device`: a scope added later cannot leak into
the full routes by accident. Mutation tested.

**Nothing acts on the data yet.** Same path the radar took: log, look, then act.
How promptly iOS reports a crossing in the background is unmeasured, and a wrong
"away" would make her deaf in his own room.

**Only home or away leaves the phone, never coordinates.** The Pi does not need
to know where he is, only whether he is home.

## Earlier project snapshot

These excerpts from the prior Claude guide preserve older observations, the legacy cloud setup and tuning rationale. Use [Headless development](DEVELOPMENT.md) for current setup and source for current settings. The full former guide, including the dated repository tree, is available with `git show 6e72b9e:CLAUDE.md`.

### Current Status

v0.7.1 shipped. Unreleased work has landed on top of it since (latency
instrumentation, webrtcvad endpointing, recording archive, dismiss action,
clock injection). All three systemd services enabled and active. Tunnel live at
example.invalid, dashboard managed.

DO NOT modify existing services unless explicitly asked. They are running in
production.
DO NOT edit ~/.cloudflared/config.yml. Tunnel config is dashboard managed.
DO NOT touch the Swift companion app (separate repo at
the Nova repository).

### What This Project Is

A personal AI voice assistant running on a Raspberry Pi 5 (8GB). The voice
personality is named Nova. M.I.L.E.S. is the system name on the resume and GitHub.

Built by the owner, CpE student at University of Florida, Class of 2029.
Primary portfolio project targeting embedded/firmware and full stack hardware to
cloud engineering roles.

The companion iOS app (Nova) is a separate repo at
the Nova repository.

### Hardware

Raspberry Pi 5 (8GB), headless Raspberry Pi OS Lite 64 bit. ReSpeaker XVF3800
USB 4 mic array (since Sep 30 2026, firmware 2.1.1_48k2ch since Oct 1; the
Razer Seiren V3 Mini before it, unplugged Oct 1), found
by name in PyAudio rather than hardcoded to a card number, since ALSA card
numbers shift on reboot. It does beamforming, noise suppression and AGC on chip
and hands over two channels; the pipeline keeps channel 0. Amazon Basics speakers
plug into the array's own 3.5mm jack (since Oct 1 2026; the QianLink AB13X USB
adapter before it), so the echo canceller hears her on the array's own clock.
Resolved by name at runtime in `tts.py` using `SPEAKER_NAME_HINT`. The array's
`PCM` playback is 54 of 60 (minus 6dB, against distortion on her loudest
syllables), saved with `alsactl store`; loudness comes from the speakers' knob. An LD2410C mmWave
presence radar on the GPIO UART (`dtparam=uart0=on`, `/dev/ttyAMA0`); wiring and
tuning in [BACKEND_TODO.md](BACKEND_TODO.md#presence-detection). Development machine
is a MacBook connected via VS Code Remote SSH at fake-calendar-10@example.com, local
network only. Remote SSH via Cloudflare Tunnel is planned and not yet configured.

Mic gain is tuned to `EXPECTED_MIC_GAIN` and checked at startup. The mixer card
and the mic itself are both resolved **by name**, never by ALSA card number,
because those shift on reboot. Why that matters, and the months of false drift
warnings it cost before Sep 6 2026, is in
[docs/AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#microphone-and-gain).

### Cloud Sessions

A cloud session (Claude Code on the web) gets a fresh clone and nothing
gitignored: no `.env`, `models/`, `data/`, `build/` or `whisper.cpp/`, and no
mic, speaker, radar or systemd. So it can read, edit, test and push code, and
it cannot run Nova. Anything that changes the Pi happens on the Pi, with
`scripts/deploy.sh`.

The environment's setup script, pasted into its settings on claude.ai, installs
everything before the session starts. This is the versioned copy; when a pin
changes in `requirements.txt`, change it here and there too. Fixed Oct 30 2026:
the earlier script died on webrtcvad and installed nothing:

```bash
#!/bin/bash
set -euo pipefail
# the clone is either the working directory or a folder named miles inside it
[ -f requirements-dev.txt ] || cd miles
# root needs no sudo; any other user does
SUDO=""; [ "$(id -u)" -eq 0 ] || SUDO="sudo"
# PyAudio compiles against PortAudio's headers
$SUDO apt-get update -qq && $SUDO apt-get install -y -qq portaudio19-dev
# allow pip on a distro managed Python; a pip too old to know it ignores it
export PIP_BREAK_SYSTEM_PACKAGES=1
# Debian's patched setuptools cannot build webrtcvad or typing (install_layout
# error), so put upstream's beside it first. wheel is Debian owned too
python3 -m pip install --ignore-installed setuptools wheel
# One pass. --ignore-installed because Debian owns some packages (httplib2) and
# pip cannot uninstall them. The CPU index is searched as well, and its 2.11.0+cpu
# torch outranks the default 2.11.0 CUDA wheel, which pulls gigabytes of nvidia
python3 -m pip install --ignore-installed \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```

Then test:

```bash
cd src && python3 -m pytest tests/ -q
```

Measured Oct 30 2026 in a clone with no `.env` and an empty home: 1447 pass,
6 skip, the same as on the Pi. The six are in `test_speculation.py` and import
`audio.py`, which opens the mic and loads `models/hey_nova.onnx`; they skip
without the model, and on the Pi while `miles-voice` holds the mic. Any
failure there is real.

### Key Config Values

**Read out of `src/config.py` on Sep 13 2026.** This is a snapshot from the former guide. Use source to verify current values.

#### Audio capture — why: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md)
- `CHUNK = 1280` (80ms frames, required by openWakeWord)
- `RATE = 16000`, `CHANNELS = 1`
- `WAKE_THRESHOLD = 0.4`
- `VERIFY_THRESHOLD = 0.15` (ECAPA; per encoder in `VERIFY_THRESHOLDS`, Resemblyzer's
  was 0.5. Below it a voice is answered as a guest, not refused)
- `VAD_MODE = 2` (webrtcvad)
- `VAD_PREROLL_MS = 300`, `VAD_ONSET_FRAMES = 2`
- `SILENCE_LIMIT = 1.2` (after he has spoken; `endpoint.py`)
- `COMMAND_ONSET_S = 3.0` (how long a command waits for him to start; was
  SILENCE_LIMIT until Oct 1 2026, and a pause after the chime lost the question)
- `MAX_RECORD = 30.0`
- `TTS_FLUSH_MARGIN_MS = 250`
- `EXPECTED_MIC_GAIN = 60` (the array's `Headset` capture control, at its maximum;
  the chip runs its own gain)
- `MIC_STALL_S = 5.0` (a mic read blocked this long ends the process so systemd
  restarts it; `mic_watchdog.py`)
- `MIC_CAPTURE_CHANNELS = 2`, `MIC_CHANNEL = 0` (the array offers only two;
  `mic_channel.py` keeps one, so every reader stays mono)
- `MIC_CAPTURE_RATE = 48000` (what the array firmware delivers: 48000 on
  2.1.1_48k2ch, live since Oct 1 2026; 16000 on 2.0.6, the rollback.
  `mic_channel.py` filters it down to `RATE`)
- `MIC_NAME_HINT = "XVF3800"`. `MIC_MIXER_CARD` is **derived** from it at import
  by `_resolve_mixer_card`, never written literally, and is `None` on a miss so
  callers report "cannot check" rather than measuring the wrong device.
- `SPEAKER_ENCODER = "ecapa"` (since Sep 27 2026; Resemblyzer voiceprint kept in
  `models/resemblyzer_backup_2026-09-26` for rollback)
- `VERIFY_MAX_SECONDS = 10.0` (verification embeds only the first ten seconds of
  trimmed speech; ECAPA on a whole long turn cost up to 2.8s of wait)
- `MIN_VOICED_SECONDS = 4.0`, `ENROLL_RECORD_SECONDS = 10`

#### Whisper — why: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#whisper)
- `WHISPER_MODEL = whisper.cpp/models/ggml-base.en-q8_0.bin` (quantized locally
  from base.en with `whisper-quantize`; derived, so regenerate after a
  whisper.cpp upgrade)
- `WHISPER_INITIAL_PROMPT = None` (a decision, not an omission)
- `WHISPER_AUDIO_CTX = 1000` (20 seconds. Do not lower without rerunning the
  validation; 750 and 900 both corrupted reference speech). Used up to 15
  seconds of audio, the length it was validated for (`audio_segments.py`)
- `WHISPER_AUDIO_CTX_LONG = 1500` (Whisper's full 30 seconds, for longer clips)
- `WHISPER_SEGMENT_SECONDS = 28.0` (longer recordings are cut at a pause and
  transcribed in pieces)
- `WHISPER_BEAM_SIZE = 5` (whisper.cpp's own default; this pipeline passed 1 until
  Sep 25 2026 and every WER figure before that date was greedy. 19.8% to 17.5%
  WER for 113ms. Set to 1 for greedy; both the speculative and full runs read it
  through `_whisper_cmd`)

#### Deepgram — why: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#deepgram-nova-3-measured-sep-25-2026)
- `DEEPGRAM_STT = True`, `DEEPGRAM_MODEL = "nova-3"`, `DEEPGRAM_TIMEOUT_S = 8.0`
- `DEEPGRAM_KEYTERMS` (since Oct 9 2026): his family's names, sent as Nova 3
  keyterm prompts so "Maj" is not heard as "Madge". Names only. Unscored: every
  clip in `data/stt_eval` had been pruned from the archive by then
- `DEEPGRAM_API_KEY` from `.env`. The key is a secret, the model name is not
- **Whisper is the floor and stays.** Every failure falls back to it, so the
  offline path does not regress. `DEEPGRAM_STT = False` returns to fully local
- Every recording leaves the Pi, including the 45 percent of wakes that are
  false. That was decided, not overlooked: see the decision log in
  [engineering decision log](#every-recording-goes-to-deepgram-sep-25-2026-decided)

#### Wake capture — why: [AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#wake-capture)
- `CAPTURE_WAKE_MISSES = True`, `WAKE_MISS_DIR = ~/miles/data/wake_misses`
- `WAKE_MISS_FLOOR = 0.15`, `WAKE_LOG_FLOOR = 0.15`
- `WAKE_MISS_MAX_FILES = 400`, `WAKE_MISS_PREROLL_MS = 2500`
- `CAPTURE_WAKE_HITS = True`, `WAKE_HIT_DIR = ~/miles/data/wake_hits`
- `WAKE_HIT_MAX_FILES = 300`
- `WAKE_LEVEL_GATE = True`, `WAKE_LEVEL_FLOOR = -42.0` (a wake whose window never
  peaks above this is dropped before the chime. The score cannot separate a real
  wake from the neighbours and level can; **do not raise the floor without far
  field clips in the sample**)
- `WAKE_RULE = True`, `WAKE_RULE_SCORE_BELOW = 0.5`, `WAKE_RULE_PEAK_ABOVE = -9.0`
  (built Oct 12 2026, live from the Oct 26 restart; a wake scoring under 0.5 must also peak louder than minus 9
  dBFS. On his 106 labelled array wakes it cut 20 of 70 false ones and lost none
  of his 36, most of them from his seat; `scripts/wake_rule.py` rescores it)
- `WAKE_GATED_DIR = ~/miles/data/wake_gated`, `WAKE_GATED_MAX_FILES = 300`
  (rejections keep their audio, because a guard that can deafen her has to be
  auditable by ear. `label_wake.py gated` reads it)

#### Speculative transcription — why:
[AUDIO_PIPELINE.md](AUDIO_PIPELINE.md#speculative-transcription)
- `SPECULATIVE_TRANSCRIBE = True`, `SPECULATIVE_SILENCE_MS = 210`,
  `SPECULATIVE_THREADS = 3`
- `SPECULATIVE_CLAUDE = True` (the first Claude call opens during the endpoint wait
  and the real turn claims it; `speculative_turn.py`)
- `WHISPER_FALLBACK_AFTER_MS = 1000` (speculative whisper starts only when Deepgram
  fails or has not answered by then)

#### Barge in — why: `config.py` comments and [BACKEND_TODO.md](BACKEND_TODO.md)
- `BARGE_IN = True` (since Oct 1 2026), `BARGE_IN_THRESHOLD = 0.30`,
  `BARGE_IN_CHANNEL = 1` (the recognition beam: the wake word and his speech are
  heard there, and an interruption is recorded there. Off
  Sep 30 while her echo crossed two USB clocks; with her on the array's jack the
  model heard him over her 27 of 29. `play()` stops mid sentence)
- `BARGE_IN_WAKE_FLOOR = -30.0`, `BARGE_IN_WAKE_WINDOW_S = 2.5` (since Oct 12 2026;
  a wake over her whose last 2.5s on channel 1 never peaks above the floor is
  ignored, its audio kept as `wakegated_*` in the stop dir. Oct 5 the model fired
  at 0.43 on near silence; `wake_gate.barge_in_too_quiet`)
- `BARGE_IN_SPEECH = True` (since Oct 1 2026), `BARGE_IN_SPEECH_FLOOR = -37.0`,
  `BARGE_IN_SPEECH_MS = 300` (any speech over her stops her, not only the wake
  word: 30ms frames over the floor that webrtcvad calls speech, that many in a
  row, on `BARGE_IN_CHANNEL`; `interrupt_speech.py`. Anyone's speech counts)
- `BARGE_IN_ECHO_CHECK = True` (since Oct 5 2026), `BARGE_IN_ECHO_SCORE = 0.8`,
  `BARGE_IN_ECHO_WINDOW_MS = 1200`, `BARGE_IN_ECHO_LAG_MS = (30, 300)` (a run whose
  levels follow what she is playing is her own echo and does not stop her; ignored
  stretches keep their audio in the stop dir as `echo_*`. Why:
  [BACKEND_TODO.md](BACKEND_TODO.md))
- `BARGE_IN_STOP_DIR = ~/miles/data/barge_in_stops`, `BARGE_IN_STOP_MAX_FILES = 100`
  (the audio behind every stop, so a false one can be heard)
- `RESUME_WINDOW_S = 120` (in `resume.py`, not `config.py`; since Oct 30 2026. How
  long the unsaid part of a cut reply is held, and the longest a resume waits. Why:
  [BACKEND_TODO.md](BACKEND_TODO.md#resuming-a-reply-he-cut-off-partly-built-oct-30-2026))

#### Model and prompt — why: [BRAIN.md](BRAIN.md)
- `MODEL_A = "claude-haiku-4-5"` — **this is the production model**
- `MODEL_B = "claude-sonnet-4-5-20250929"`
- `MODEL_AB_TEST = False`
- `PROMPT_CACHING = True`
- `MEMORY_PASS = True`, `MEMORY_PASS_MODEL = "claude-haiku-4-5"` (background
  noticing, after the reply; `MAX_PER_MESSAGE = 3` and `SAME_MESSAGE_OVERLAP =
  0.8` in `memory_pass.py`, the second skipping a piece of what the turn itself
  stored from the same message)
- `HISTORY_ASSISTANT_WORDS = 30`
- `HISTORY_GAP_NOTE_MINUTES = 30` (past this since her last reply, the newest
  message says how long ago the conversation before it ended; history has no
  times, and without it she answered from a five hour old reply)
- `MEMORY_DUPLICATE_SIMILARITY = 0.9` (the memory pass skips only a near rewording;
  lower starts dropping real facts), `MEMORY_SIMILAR_LIST = 0.75` (what
  `memory.py similar` lists for merging by hand). Why:
  [BRAIN.md](BRAIN.md#duplicates-are-not-merged-automatically-measured-sep-28-2026)

#### Conversation loop — why: [BRAIN.md](BRAIN.md#local-intent)
- `MAX_FOLLOWUP_TURNS = 6`, `FOLLOWUP_TIMEOUT = 3.0`
- `INTENT_SIMILARITY_THRESHOLD = 0.55`, `MAX_DISMISS_WORDS = 6`
- Exit is intent based, not a phrase list. Dismiss is a registered tool.

#### TTS and phrase bank — why: [VOICE_OUTPUT.md](VOICE_OUTPUT.md)
- `TTS_VOICE_ID` = Victoria (`[voice identifier omitted]`). The single place a voice
  is named; no call site references one.
- `DEFAULT_TTS_MODEL = "eleven_v3"` (chosen by ear, Sep 13 2026)
- `EXPRESSIVE_TTS_MODEL = "eleven_v3"` (HTTP only, no WebSocket)
- `TTS_OUTPUT_FORMAT = "pcm_22050"` (raw S16_LE mono, piped to aplay)
- `TTS_APP_OUTPUT_FORMAT = "mp3_44100_64"` (what `/speak` sends the app; PCM is
  free down a local pipe and wasteful down the tunnel)
- `TTS_HEDGE_AFTER_MS = 1500` (a request with no audio by then is sent again; first to
  answer plays)
- `TTS_PHONEME_TAGS = True`
- `SPEAKER_NAME_HINT = "XVF3800"` (the array's jack since Oct 1 2026; resolved by
  name at runtime in tts.py)
- `PHRASE_DIR = ~/miles/data/phrases`, gitignored. `PHRASES` in `phrasebank.py`
  is the versioned source of truth; the WAVs are derived from it.
- `ACK_SPOKEN_CHANCE = 0.75`
- `BRIDGE_DELAY_MS = 400` (a bridge line plays only if the tool is still running
  after this)
- Voice settings: `stability=1.0`, `similarity_boost=0.75`, `style=0.00`,
  `speed=1.00` (v3 ignores it), `use_speaker_boost` unset. **Changing any of
  these means re rendering the whole phrase bank.**

#### Tools and permissions — why: [BRAIN.md](BRAIN.md#tools-and-the-permission-gate)
- `PERMISSION_TIERS` (in `tools.py`): READ `genin`, CONTROL `genin`, WRITE
  `chunin`, EXTERNAL_WRITE `hokage`. A tool's `min_tier` can only raise its floor.
- Raised by `min_tier`: `remember` to `jonin`; `lower_access`, every calendar
  tool, every Oura tool, every Hevy tool, `get_health_summary`, every Prowl tool, `list_pending_memories` and
  `review_pending_memory` to `hokage`.
- `CONFIRM_WINDOW_S = 120` (in `pending_action.py`, not `config.py`)

#### Push notifications — why: [INFRASTRUCTURE.md](INFRASTRUCTURE.md#push-notifications-oct-30-2026)
- `PUSH_NOTIFICATIONS = True`, but nothing is sent until `APNS_TEAM_ID`,
  `APNS_KEY_ID` and `APNS_KEY_PATH` are in `.env`. `APNS_BUNDLE_ID` defaults to
  `example.invalid`. The `.p8` key is a secret (`*.p8` is gitignored)
- `PUSH_KINDS = ("reminder", "timer")`
- `PUSH_TTL_S = {"reminder": 6 hours, "timer": 5 minutes, "session": 3 hours}` (and
  approval, watchdog, login), `PUSH_DEFAULT_TTL_S = 3600`
  (how long Apple holds a push for a phone that is off; a late timer is noise)
- `APNS_TIMEOUT_S = 10.0`

#### Sessions — why: [SESSIONS.md](SESSIONS.md)
- `SESSION_MODELS = {"nightly_review": "claude-haiku-4-5"}` (a type missing here runs on `MODEL_A`)
- `SESSION_BLOCK_EARLIEST = (10, 0)`, `SESSION_BLOCK_LATEST = (21, 0)` (where the review
  places blocks when he gives no time)
- `CALENDAR_ROLES` (by calendar id: blocking, club, holiday; an unlisted calendar is blocking),
  `UF_BREAKS`, `UF_HOLIDAYS` (public holidays that are break days), `CLASS_TITLES` (DSA and
  the course codes, both spellings), `EXAM_WORDS`, `EXAM_EXCLUDE` (all in `config.py`)
- `MAX_ROUNDS = 6` per turn (in `sessions.py`, not `config.py`)

#### Presence radar — why: [BACKEND_TODO.md](BACKEND_TODO.md#presence-detection)
- `PRESENCE_SENSOR = True`, `PRESENCE_PORT = "/dev/ttyAMA0"`, `PRESENCE_BAUD = 256000`
- `PRESENCE_SILENT_S = 3.0` (no frame this long is logged as silent, never absent)
- `PRESENCE_ENERGY_LOG = True`, `PRESENCE_ENERGY_WINDOW_S = 60.0` (engineering mode
  on at each open; energy per gate, once a minute, to `presence_energy`)
- Logging only, to `presence_log`. Nothing Nova does reads it yet. The module's
  own thresholds and range live in its flash, set with `scripts/radar.py`; the
  table is in BACKEND_TODO.md

#### Storage and other
- `DB_PATH = ~/miles/data/miles.db`
- `ARCHIVE_RECORDINGS = True`, `ARCHIVE_DIR = ~/miles/data/recordings`,
  `ARCHIVE_MAX_FILES = 600` (~150MB, a few weeks of normal use, oldest pruned first)
- `ARCHIVE_SPOKEN = True`, `SPOKEN_DIR = ~/miles/data/spoken`, `SPOKEN_MAX_FILES = 100`
  (~45MB; labelled replies pruned last. Why: [VOICE_OUTPUT.md](VOICE_OUTPUT.md#what-she-said-is-kept-sep-28-2026))
- `DEFAULT_LOCATION = "Gainesville"`
- `WEB_SEARCH = True`, `WEB_SEARCH_TOOL_TYPE = "web_search_20250305"` (the
  variant Haiku 4.5 supports), `WEB_SEARCH_MAX_USES = 2`, located in Gainesville.
  Why: [BRAIN.md](BRAIN.md#the-rest-of-the-world-comes-from-web-search-sep-27-2026)
- `REMINDER_POLL_S = 5`, `REMINDER_LATE_S = 3600` (both in `actions.py`, not
  `config.py`)

Recordings are of a real person in a real room. Treat them accordingly.

### Tech Stack

Wake word: openWakeWord (hey_nova.onnx).
VAD: webrtcvad, mode 2.
STT: Deepgram Nova 3 online, falling back to whisper.cpp (base.en, beam search at
5, NEON ARM optimizations, capped audio context) whenever it does not answer.
LLM: Claude API, `claude-haiku-4-5`, streaming, prompt caching on the system prompt.
TTS: ElevenLabs (Victoria, eleven_v3), pcm_22050 piped to aplay.
Voice auth: SpeechBrain ECAPA (192 dim cosine similarity); an unverified voice is a guest.
Memory: SQLite WAL mode, schema migrations to version 35.
Push: Apple's push service (APNs) over HTTP/2, `httpx` with `h2`, ES256 provider token signed by python-jose.
Phone health: Apple Health (HealthKit), copied to the Pi by the Nova app when it opens.
Weather: Apple WeatherKit, with the Maps Server API for place names, both signed by apple_keys.py with the APNs key (Nov 21 2026).
Calendar: Google Calendar API v3, OAuth, dateparser for spoken times.
Health: Oura API v2, OAuth.
Workouts: Hevy API v1, static key (Hevy Pro).
Jobs: Prowl (example.invalid), its MILES surface, token auth.
Live info: Anthropic's web search tool, run server side inside the Claude call.
Backend: FastAPI + uvicorn + python-jose + passlib (bcrypt) + python-dotenv.
Tunnel: cloudflared (dashboard managed).
Process management: systemd.
Language: Python 3.13.5.

### What Is Completed

- v0.1: Wake word detection + Whisper transcription
- v0.2: Claude API integration + Fish Audio TTS
- v0.3: Resemblyzer speaker verification + energy based VAD
- v0.4: SQLite persistent memory + conversation history
- v0.5: Action tag system (weather, timers, reminders, cancellations)
- v0.6: Multi turn follow up conversation loop + exit phrases
- v0.7: Module refactor + JWT auth + FastAPI server + Cloudflare Tunnel +
  systemd services + example.invalid + Nova iOS app on physical device
- v0.7.1: ElevenLabs TTS migration + Claude streaming with StreamRouter +
  concurrent TTS queue + wake chime + Miles pronunciation normalization

Landed since v0.7.1, unreleased:
- Per turn latency instrumentation (`timing.py`, `timing_log`)
- webrtcvad endpointing, `SILENCE_LIMIT` 3.0 to 0.9
- Whisper audio context cap, Haiku migration, prompt caching
- Response length rewrite + history trimming (median 76 words to 54)
- Exit phrase list replaced by intent based `[ACTION: dismiss]`
- Command recording archive
- Clock injected into the last user turn (`_with_current_time`)
- Offline resilience: `run_turn` exception boundary in voice_main, phrase bank,
  `netcheck` cause diagnosis, spoken wake acks
- Local intent for `set_timer`, `time_of_day`, `cancel_reminder` (timers too), `dismiss`
- Speculative transcription during the endpoint wait
- Native tool use: the registry in `tools.py` and the tool loop in `brain.py`,
  replacing bracket action tags
- Permission gate enforced in the executor, one tier per turn
- Google Calendar and Oura tools; calendar additions, moves and deletes happen
  at once with a thirty minute undo
- Timers are rows fired by the reminder poller, so they survive restarts and
  fire wherever they were set
- A text turn is silent, answers in numerals, and can be streamed to the app
  over `/chat/stream`, with stage events while tools run
- App endpoints: memories (add, edit, history), reminders and timers, status
  details, and a calendar week view with tap edits on the MILES calendar
- `POST /speak` returns Nova's voice as mp3, so the app can play a reply instead
  of only showing it
- The room speaker follows `device`, not `channel`: only a turn from the Pi is
  played aloud, so the app can ask for spoken formatting and synthesize it itself
- Prowl job search tools: summary, applications, discoveries, and accepting one
  discovered job after a spoken yes
- Web search for anything current (scores, news, hours), run by Anthropic
- Hevy reads: recent workouts, one lift's history, routines and when each was
  last done, all converted back to the pounds he logged; a session plan per
  routine that says whether to add a rep or add weight, from the rep ranges
  on his Hevy routines, and flags a stall
- Automatic progression: `miles-hevy.timer` raises a lift's weight in the Hevy
  routine once a workout earns it (10 lb squat and deadlift families,
  machines and dumbbell pairs, 5 lb otherwise; dumbbells stop at the
  apartment's 50 a hand), and the session plan says what moved. A raise his
  phone's stale copy undoes at a workout save is put back

- The ReSpeaker XVF3800 on its 48kHz firmware, resampled to 16kHz at the
  stream; speakers on its own jack, so its echo canceller works
- Barge in: "hey nova" over her stops her mid sentence; so does any speech of
  his, and what he said is kept and answered as the next turn
- What a cut reply had left to say is kept for two minutes (`resume.py`), ready
  for a "go on": she restarts the sentence she was cut in, with "Anyway," or "So,"
  only after ten seconds. A stop followed by noise or too short a recording says
  it by itself, and a bare "go on" from him says it on request (a guest's does
  not). Her history is amended in place so the next turn reads one reply, said
  once. Heard on the Pi Oct 30 2026 with "Hey Nova, read the test paragraph"

- Push notifications: a due reminder or timer is also sent to his phone through
  Apple's push service, on its own thread after the room alert. `POST /push/test`
  checks the whole chain. Written from a cloud session and tested against a
  fake Apple; not yet run against the real one
- Phone health: the Nova app copies two weeks of Apple Health (steps, active
  calories, weight, workouts) to the Pi when it opens, and `get_health_summary`
  answers from the copy and says how old it is. Same caveat as push
- Home or away from the phone: a CLMonitor geofence in the app reports each
  crossing to `POST /presence/phone`, using a 180 day device token that only the
  background write routes accept. Logged only, like the radar was, until real
  days show how reliably iOS reports it

### What Is Next

- SSH via Cloudflare Tunnel: example.invalid + Cloudflare Access policy
- An evening check in that notices instead of asking: gym pace, sleep, NeetCode
  commits, Bible time; plan in [BACKEND_TODO.md](BACKEND_TODO.md#accountability-she-notices-she-does-not-ask-planned-oct-2-2026)
- v0.8+: Morning briefings (Fall 2026), Mac control, EventKit, MusicKit,
  WeatherKit migration, parallel processing
- Follow ups to push and Health, with reasons, in
  [BACKEND_TODO.md](BACKEND_TODO.md#apple-developer-program-follow-ups-oct-30-2026):
  Snooze and Done on a notification, Health sync in the background, a Lock Screen
  timer
