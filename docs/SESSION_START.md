# Session start and drift control

Read this before proposing anything. It exists because on Aug 11 2026 a full
session of design work was planned against a handoff document that had drifted
from the code, and the plan was wrong in its central premise.

---

## Rule zero: the repo wins

Documents describe the repo. The repo is the authority.

When any document here, in `CLAUDE.md`, in `docs/BACKEND_TODO.md`, or in a pasted
handoff conflicts with source, **the source wins**. Say so out loud, name the
conflicting claim, and fix the document in the same session. A silently
corrected doc teaches nothing; a loudly corrected one stops the next repeat.

This applies to me and to any assistant working here. An assistant that reads a
document claim and a source file that disagree must report the disagreement
before acting on either.

---

## Start of session preflight

Run this. It takes seconds and it is cheaper than a wrong plan.

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
| Which tools Nova actually has | `python3 -c "import brain; from tools import registry; print(registry.names())"` | Aug 11 2026 (7) |
| Test count | `cd src && python -m pytest tests/ -q \| tail -1` | Aug 11 2026 (273) |
| Perceived latency | preflight step 6 | Aug 11 2026 (4938ms median) |
| Prefix token count (never trust a written figure) | `count_tokens` on `build_enhanced_prompt` output vs the 4096 floor | Aug 11 2026 (5942, +1846) |
| `VERIFY_THRESHOLD` | `grep VERIFY_THRESHOLD src/config.py` | Aug 11 2026 (0.5) |
| Speaker device resolution | `grep -n SPEAKER src/config.py src/tts.py` | Aug 11 2026 (by name, not device string) |
| Which modules exist | `ls src/*.py` | Aug 11 2026 (20 modules) |
| Schema version | `grep -c "^    (" src/database.py` around `MIGRATIONS` | Aug 11 2026 (13) |

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

## End of session checklist

Do this before closing the tmux session. It is the other half of the preflight.

1. **Did any config constant change?** Update the Key Config Values table in
   `CLAUDE.md`. That table is the single most load bearing part of the file.
2. **Did any claim in `CLAUDE.md` become false?** Fix it now, not next session.
   Include the correction in the commit message.
3. **Did a decision get made that a future session would otherwise reopen?**
   Add it to the decision log below, with the reason. Reason matters more than
   the decision, because without it the next session relitigates it.
4. **Did production diverge from the plan?** Mark it `DIVERGENCE` in the log.
5. **Did work get deferred?** Add it to `docs/BACKEND_TODO.md` with why it was
   deferred, not just that it was.
6. **New measurement taken?** Replace the old number rather than adding a second
   one. Two latency figures in one repo means nobody trusts either.
7. **Run the test suite one more time** and record the count if it changed.

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

### Clock injected into the last user turn, not the system prompt (Aug 11 2026) (DONE)

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
    memories are facts about Lethanial, tool results are facts about the world at
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
  Lethanial heard a guess and then the real reading. Added `TOOL_SPEECH` to the
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

Final: Victoria (`qSeXEcewz7tA0Q0qk9fH`), `eleven_flash_v2`, stability 0.80,
similarity 0.75, style 0.00, speed 1.00, `TTS_PHONEME_TAGS = True`, and
Lethanial as `L AE0 TH AE1 N Y AH0 L`.

The lesson is not the values. It is that **most of this hunt was aimed at the
wrong variable**, and the thing that exposed it was Lethanial noticing that two
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
