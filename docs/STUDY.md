# Study: the iPad tutor

Design, Oct 20 2026. Nothing here is built yet. Item 4 of the accountability
plan in [BACKEND_TODO.md](BACKEND_TODO.md#accountability-she-notices-she-does-not-ask-planned-oct-2-2026)
holds the history; this doc holds the design, and is where the measurements go
once it runs.

## What it is

He hands it the material for a test or a lesson (slides as a PDF, photos of a
study guide, or just "test Thursday on eigenvalues and diagonalization"). It
teaches one concept at a time, gives him a practice problem, he solves it with
the Pencil on the iPad, and it reads his working and answers **hint first**:
where it went wrong, not the fix. The full solution is one tap away, and a
problem he needed it for comes back before the session ends.

Modelled on Solvely (problem in, steps out, follow ups on any step), with two
differences that are his calls: it teaches before it tests, and it hints before
it solves, because he is building this to learn.

**Classes, in order:** Linear Algebra, DSA, Computer Organization, then his
other CpE classes and his own projects. Linear Algebra drives the first prompt.

## Why it is not part of Nova

Nova's turn is tuned for a voice in a room: Haiku 4.5, a median reply of about
54 words, numbers spelled out, a tool registry, a permission gate, history
trimmed to 30 words a reply. A tutor needs the opposite on nearly every axis:
long written replies with LaTeX, a whole session of history kept intact, and a
model that can read handwriting. Sharing `ask_nova` would mean branching every
one of those, and each branch would be a way to break the voice loop, which is
in production.

So it is its own module, `src/study.py`, with its own prompt and its own
history. What it reuses: JWT auth, the SSE helpers in `server.py` (`_sse` and
the keepalive), the migration list in `database.py`, and `config.py` for its
values. It never touches `brain.py`, `voice_main.py` or the voice services. The
only restart it needs is `miles-server`.

## The model: Opus 5.5 for everything, effort per call

Measured Oct 20 2026 on one dense Freeform scratch page of his linear algebra,
transcribed and graded by hand against the page:

| Model | Misreads | Marked unsure `[?]` | Verdict |
|---|---|---|---|
| Haiku 4.5 (Nova's) | nearly every dense matrix | almost never | cannot check work |
| Sonnet 4.5 | the null space line, several matrices | rarely | cannot check work |
| Haiku 5.5 | 5, including minus 1 as minus 3 | rarely | cannot check work |
| Sonnet 5.5 | 2, including minus 1 as minus 5 | yes | close, not trusted |
| Opus 5.5 | none found | yes | **reads his working** |

A misread that changes a number is the failure that matters: the tutor would
tell him he is wrong when he is right. Marking `[?]` instead of guessing is
what lets it ask "is that a 0 or a 9?" instead.

**One model, two effort levels,** rather than Opus to read and a cheaper model
to talk. Opus 5.5 cannot turn thinking off, but `output_config.effort` sets how
much it thinks:
- `low` for teaching turns (explaining, next problem, a follow up question)
- `high` for checking an attempt, where a wrong read costs trust

One model keeps one prompt cache (caches belong to a model, so two models would
pay for the material twice) and one prompt to tune. If teaching turns turn out
slow or costly at `low`, Sonnet 5.5 for teaching is the fallback, measured
first.

**Cost, estimated, to be replaced by measurement.** Opus 5.5 is $4 per million
input tokens, $20 per million output, $0.20 per million for cached input. A
teaching turn rereads the material from cache and writes a few hundred words:
about 3 cents. A checked attempt is about 4,600 tokens of image in and a few
thousand out: about 8 cents. A session of 20 turns and 8 attempts is about
$1.25. Every call logs its usage, so the first real session replaces this
guess.

## A session, start to finish

1. **Start.** The app sends the course, a title, and the material: text, PDF,
   images, or any mix. The Pi saves the files under `data/study/<session id>/`
   and Opus reads everything once, writing a short plan: the concepts in
   teaching order, each with what he should be able to do after it. The plan
   goes back to the app to edit before anything is taught; a plan he never
   approves is never taught.
2. **Teach.** One concept at a time: the intuition, a worked example, then a
   question to check he has it, before moving on.
3. **Practice.** It gives a problem. He solves it on the canvas and taps
   submit. The app sends the drawing as a PNG with the problem's id.
4. **Check.** Opus reads the drawing at `high` effort and answers in one of
   four ways: correct; a hint pointing at the first wrong step; a question
   about something it could not read; or, if he asks, the full solution.
5. **Review.** Before the session ends, every problem he missed or needed the
   solution for comes back as a fresh problem of the same kind.

A session can be left and resumed. He can talk to it at any point ("I don't
get why the determinant is zero") and that is just a teaching turn.

## How it keeps track: tools, not parsing

The session's state (which concepts are taught, which problems are open, what
he missed) lives in tables, and Claude changes it by calling tools, the same
pattern as Nova's registry. Parsing its prose for "correct!" would break the
first time it phrased something differently. The tools are the study module's
own, not registered in `tools.py`, because Nova must never see them:

- `set_plan(concepts)`: the concept list, at the start
- `mark_concept(concept_id, status)`: taught, or skipped at his request
- `give_problem(concept_id, statement)`: records a problem before it is shown,
  so the app can attach his attempt to its id
- `grade_attempt(problem_id, result, first_wrong_step)`: `correct`, `wrong`,
  `unreadable`; a wrong or revealed problem is queued for review

Opus 5.5 does not allow forcing a tool call, so the prompt says when each one
is expected and the code checks: an attempt turn that ends without
`grade_attempt` is logged as ungraded rather than guessed at.

## Data

Migration 33 adds three tables:

- **`study_sessions`**: id, course, title, created_at, updated_at, status
  (`planning`, `active`, `done`), and `material` (the text he typed, plus the
  file names under `data/study/<id>/`).
- **`study_items`**: id, session_id, kind (`concept` or `problem`), position,
  parent concept for a problem, the statement, status (`pending`, `taught`,
  `correct`, `wrong`, `revealed`, `review`), attempts, created_at.
- **`study_turns`**: id, session_id, role, `content` as JSON, created_at, and
  for assistant turns the model, effort, input, output and cached tokens, and
  milliseconds. This is the conversation, and the cost and latency log.

**History is append only, and stored whole.** Opus 5.5 ties its thinking blocks
to the conversation that made them; a turn sent back edited, trimmed or
reordered is refused or loses its reasoning. So `content` is the response's
blocks exactly as returned, thinking included, and nothing is ever rewritten.
This is the opposite of Nova's history, which is deliberately trimmed, and it
is one more reason the two do not share code.

Images and PDFs are never stored in the database. A turn's JSON names the file
in `data/study/<id>/` and the image is read back from disk when the history is
sent. `data/` is already gitignored, so his notes cannot reach a commit.

**Caching.** The system prompt and the material sit first and never change
within a session, with a cache breakpoint after them, so a 20 page PDF is paid
for in full once per session and at the cached rate on every turn after.

## API

All behind the existing JWT. Bodies are JSON; images and PDFs travel base64
inside them. File uploads in FastAPI need `python-multipart`, which is not
installed, and a few megabytes of base64 is well inside what the tunnel
carries. Turns stream as SSE in the shape `/chat/stream` already uses, so the
app's existing stream reader works.

| Method | Path | What it does |
|---|---|---|
| POST | `/study/sessions` | start: course, title, text, files. Streams the plan |
| GET | `/study/sessions` | list, newest first, for resuming |
| GET | `/study/sessions/{id}` | the whole session: turns, plan, problems and their status |
| PATCH | `/study/sessions/{id}/plan` | his edits to the plan; approving it starts teaching |
| POST | `/study/sessions/{id}/turns` | a teaching turn: text. Streams the reply |
| POST | `/study/sessions/{id}/attempts` | problem id plus PNG. Streams the check |
| POST | `/study/sessions/{id}/problems/{pid}/reveal` | the full solution; queues it for review |

Events: `text` as it is written, `item` when the plan or a problem changes (so
the app updates its list without polling), `done`, `error`.

## The iPad side

Built in miles-app, in its own session; a handoff doc in the shape of
[APP_HANDOFF_DUPLICATES.md](APP_HANDOFF_DUPLICATES.md) gets written once this
API is fixed. What it needs:

- a PencilKit canvas per problem, exported as PNG at the screen's full
  resolution and cropped to what he drew. The Freeform test was at 68 percent
  zoom on a crowded page; one problem per canvas is easier to read
- math rendering: replies are Markdown with LaTeX
- a plan screen to edit and approve, a problem list with status, a reveal
  button, and the session list for resuming

## Order of work

Each step is usable before the next starts.

1. Migration 33 and the three tables, with tests.
2. `study.py`: one teaching turn and one attempt check, driven from a script
   on the Pi with a saved PNG, before any endpoint exists. This is where the
   prompt gets tuned, on his real material.
3. The endpoints in `server.py`, and a `miles-server` restart.
4. The handoff doc, then the app.

## Open questions

- **Does `low` effort teach well enough?** First thing measured in step 2.
- **Hint quality.** "Your row 2 swap is wrong" is a hint; "row 2 should be
  [0 1 3]" is the answer. Where that line sits gets tuned on his attempts,
  not decided here.
- **Voice, later.** "Nova, quiz me on what I missed yesterday" reads the same
  tables. Not in the first version.
