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
| Which tools Nova actually has | `python3 -c "import brain; from tools import registry; print(registry.names())"` | Sep 14 2026 (25) |
| Test count | `cd src && python -m pytest tests/ -q \| tail -1` | Sep 16 2026 (796) |
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
   `CLAUDE.md`. That table is the single most load bearing part of the file, and
   since Sep 13 2026 it is the **only** place a live value is declared. The topic
   docs explain values and must never carry a second copy of one.
2. **Did any claim become false?** Fix it now, not next session, and include the
   correction in the commit message. Check the doc that owns the subject, not
   just `CLAUDE.md`:

   | You touched | Update |
   |---|---|
   | mic, gain, wake word, VAD, Whisper, verification | `docs/AUDIO_PIPELINE.md` |
   | prompt, model, streaming, tools, channels, local intent | `docs/BRAIN.md` |
   | TTS, voice settings, phrase bank, pronunciation | `docs/VOICE_OUTPUT.md` |
   | anything measured in `timing_log` | `docs/LATENCY.md` |
   | systemd, tunnel, API, env vars, reminders | `docs/INFRASTRUCTURE.md` |
   | something that broke in production | `docs/INCIDENTS.md`, dated |

   **A measurement never moves to a new doc, it gets replaced in the one that
   owns it.** Two copies of a number is how the repo ends up unable to say which
   is current.
3. **Did something break and get fixed?** It goes in `docs/INCIDENTS.md` with
   its date and the evidence, not in the decision log. The decision log is for
   choices made deliberately. If an incident forced a decision, write the
   incident there and the decision here, and link them.
4. **Did a decision get made that a future session would otherwise reopen?**
   Add it to the decision log below, with the reason. Reason matters more than
   the decision, because without it the next session relitigates it.
5. **Did production diverge from the plan?** Mark it `DIVERGENCE` in the log.
6. **Did work get deferred?** Add it to `docs/BACKEND_TODO.md` with why it was
   deferred, not just that it was.
7. **New measurement taken?** Replace the old number rather than adding a second
   one. Two latency figures in one repo means nobody trusts either.
8. **Run the test suite one more time** and record the count in the repo tree
   in `CLAUDE.md` if it changed. It sat at 367 in that tree for a month while the
   real count was 500, which meant the preflight's "test suite green, and how
   many tests?" check had nothing true to compare against.

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

**The split rule, so this does not get relitigated.** CLAUDE.md declares; the
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

### The Oura client secret is not rotated (Sep 13 2026) (DECIDED)

It sat as a fallback default in the untracked `scripts/oura_auth.py`, was never
committed (every commit in history checked), and was moved into `.env`.
Lethanial chose not to rotate it. Recorded so a later session does not reopen it;
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
Nova iOS app must go through miles.lethanial.com; `miles.local:8000` now
refuses. That is the intent rather than a regression, and it is the part a
future session is most likely to want to undo. Do not undo it by widening the
bind.

**Tailscale closes the half of this that binding does not touch**, namely that
recovery required a publicly addressable machine at all. Installed on the Pi
Sep 15 2026 and joined as `miles` at `100.99.248.127`. It was reported as
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
failed. Root survived on `PermitRootLogin without-password`; `theycallmelee`
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

**`AllowUsers theycallmelee` is the exception**, and it lives in
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
itself broken. **Root rather than `theycallmelee`**, unlike `miles-health`,
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
