# MILES roadmap

> **Status, Nov 28 2026.** This file is the one place for the order of work. It was
> rewritten on Nov 28 2026 from the original brief (Nov 24 2026) and its amendments
> (Nov 24 and 25 2026); the original text is kept in
> [archive/ROADMAP_ORIGINAL_BRIEF_2026-11-24.md](archive/ROADMAP_ORIGINAL_BRIEF_2026-11-24.md).
> Phase numbers collided (the roadmap, the audio hardware work and the tutoring work all
> had a "Phase 3"), so every item below has a name, not a number. A phase is built one at a
> time: a plan and a contract first, tests, one builder at a time, one verification round, a
> separate deploy plan with a fresh backup and rollback.

## Order

| # | Item | Status |
|---|---|---|
| 1 | [Tasks](#tasks) | done and deployed Nov 27 2026 |
| 2 | [Nightly backup](#nightly-backup) | built, waiting to be merged and installed |
| 3 | [Tutoring session follow ups](#tutoring-session-follow-ups) | built, deploying |
| 4 | [Daily bug batch](#daily-bug-batch) | standing |
| 5 | [Docs cleanup](#docs-cleanup) | this branch |
| 6 | [Acceptance harness](#acceptance-harness) | queued |
| 7 | [CI](#ci) | queued |
| 8 | [Cloud coding sessions](#cloud-coding-sessions) | queued |
| 9 | [SEC5](#sec5) | queued |
| 10 | [Own the tutoring calendar](#own-the-tutoring-calendar) | queued |
| 11 | [Tutoring prep and tracking with Evallo Prepwork](#tutoring-prep-and-tracking-with-evallo-prepwork) | queued |
| 12 | [Mission Control v0](#mission-control-v0) | queued |
| 13 | [SEC4 and SEC6](#sec4-and-sec6) | queued |
| 14 | [Release control](#release-control) | queued |
| 15 | [Tutoring reply handling and Evallo commands from Nova](#tutoring-reply-handling-and-evallo-commands-from-nova) | queued |
| 16 | [Planner bugs and security cleanup](#planner-bugs-and-security-cleanup) | queued |
| 17 | [Model audit](#model-audit-local-models-for-background-jobs) | queued |
| 18 | [School](#school-assignments-and-study-plans) | queued |
| 19 | [Routine](#routine) | planned |
| 20 | [Jobs](#jobs) | planned |
| 21 | [Learning](#learning-learn-and-quiz-sessions) | queued |
| 22 | [Kitchen](#kitchen) | planned |
| 23 | [Directions](#directions) | queued |
| 24 | [Nova on iPad and Mac, with a design system](#nova-on-ipad-and-mac-with-a-design-system) | queued |
| 25 | [Full security audit](#full-security-audit) | queued, before the errand worker uses any real account or runs unattended |
| 26 | [Errand worker](#errand-worker) | queued |
| 27 | [Passive memory](#passive-memory) | queued |
| 28 | [Proactive layer](#proactive-layer) | queued |

Later, not scheduled: [Cronometer with Halo](#later), voice speed work, Halo, Environmental Intelligence.

Live already, and not in the queue: [Sessions and nightly review](#sessions-and-nightly-review-live), [Nightly review v2](#nightly-review-v2-propose-do-not-interrogate-live), the [Evallo session filter](#evallo-session-filter-live) and [Quick capture](#quick-capture-built-nov-25-2026).

### Old names to new names

| Old | New |
|---|---|
| Phase 1: session framework and nightly review | Sessions and nightly review |
| Phase 2: assignments and study plans | School |
| Phase 3: learn mode and quiz mode | Learning |
| Phase 4: directions | Directions |
| Phase 5: iPad and Mac | Nova on iPad and Mac, with a design system |
| Phase 6 and 6.5: errand worker, Nova Chrome extension | Errand worker (the extension is a part of it) |
| Phase S: security audit | Full security audit |
| Phase 7: passive memory | Passive memory |
| Tasks tab follow ups | Tasks (done) |
| Planner fixes, Evallo filter follow ups, Tailscale hostname cleanup, private data, login failsafe | Planner bugs and security cleanup |
| Stage D (reminder texts) | Tutoring reminders (live) |
| Tutoring Phase 4 (did the session happen?) | Tutoring session follow ups |
| Tutoring Phase 5 (Stage E replies) | Tutoring reply handling |
| Tutoring Phase 6 (Evallo commands from Nova) | Evallo commands from Nova |
| Tutoring Phase 7 (multiple Nova chats) | Multiple Nova chats (low priority) |
| Audio hardware "Phase 3 enrollment", "Phase 4" in BACKEND_TODO | Unrelated to this roadmap; they keep their own wording in the audio docs |

## Working rules

* Each phase: plan shown before building, tests, one builder at a time in its own worktree, one verification round, a fresh database backup before its own separate deploy, and a warning before anything touches production. If a usage limit hits, stop at a clean point, never half built.
* Who builds, what is probed first, and how review ends are in the workflow rules of [AGENTS.md](../AGENTS.md#workflow-rules). Runtime models stay configurable per session type and default to the cheapest that works (`SESSION_MODELS`, Haiku 4.5 for the nightly review). Checking handwritten work needs Opus 5.5 (measured in docs/STUDY.md).
* School and Learning use the MAS3114 Exam 3 (Monday Nov 30 2026) as their first real data: syllabus import, study plan, first learn and quiz sessions. Source material lives in `~/MILES/source_material/MAS3114/Exam3/`. Every Lesson 19 to 27 topic starts as new; determinants and null space are the prerequisites to check first.
* The full security audit runs before the errand worker uses any real account or runs unattended.

## Design rules

A "session" is a dedicated chat with its own purpose, system prompt, and tool set, separate from my main Nova conversation. It works like a support chat: it opens with a purpose, stays open until I close it, then saves a summary. Everything below is a session type.

The backend owns all logic. Clients (Nova on iPhone, iPad, and Mac) are thin and just render sessions and run client actions. This matters because I will mostly use study sessions on iPad or laptop, and the nightly review on whatever device gets the push.

### One source of truth

* MILES database owns courses, assignments, exams, tasks, and study topics. This is the truth.
* Canvas calendar feed is input only. MILES pulls it and updates its own records.
* Google Calendar is time only. MILES writes blocks to its own MILES owned calendar. Never treat calendar events as the source of an assignment or task.
* Notion is not part of this system.
* Any change I want goes through Nova, never by editing a copy.

## Evallo rules (permanent)

* Evallo's two way sync writes the whole organization's sessions onto his primary calendar and must stay on. MILES never deletes, moves or edits an event carrying `evalloEvent`, `evalloSessionId` or `orgId`, from any code, script or agent. The calendar layer refuses such writes before calling Google.
* An Evallo event counts as his only when its session id is in his verified set: the Mac agent reads his sessions from Evallo's API, keeps those whose `tutorId` equals the id in his own Evallo token, and sends only the session ids and their date range to the Pi. Every other Evallo event is ignored everywhere: busy time, week view, conflicts, suggestions, voice.
* If the set is older than 24 hours, keep using the last verified set; an Evallo event not in it counts as his only when his full name is in the tutor position of the title (never a first name alone), and it counts as busy. The review warns "Evallo sessions not verified since ...".
* The Evallo token stays on the Mac: never printed, logged or sent to the Pi.

* Exception to the stale set fallback (decided Nov 25 2026): with a fresh set, an Evallo event that is not in the set but falls outside the set's window, or has an unusable start, is still kept when his full name is in the tutor position of the title. This is wider than the rule above, which grants the fallback only for a stale set; it errs toward busy.
* A kept Evallo event still follows the ordinary availability flags: one marked transparent or declined is not busy time.
* Exception E1 (decided Nov 28 2026): follow up approvals carry the student's first name, the session start and end (Eastern) and the session id, in the payload and the push text, because the phone has to say whose session. Never a last name, email, family detail, tutor id or token. The Pi refuses a follow up unless the filter is active and the id is in his verified set.
* Every deploy plan that restarts the Pi includes an immediate Mac session push right after the restart, instead of waiting for the hourly run. After any Pi restart the filter stays off until a push is accepted.

## Sessions and nightly review (live)

**Status:** live. The original schema, session type registry and nightly review flow are in [archive/ROADMAP_ORIGINAL_BRIEF_2026-11-24.md](archive/ROADMAP_ORIGINAL_BRIEF_2026-11-24.md) and the current behavior is in [SESSIONS.md](SESSIONS.md).

## Nightly review v2: propose, do not interrogate (live)

* The opening is generated by code, not the model: today's result in one line, tomorrow's calendar one line per event, up to 3 suggested must dos with suggested times inside free gaps and a short reason each, optional tasks, then one question: "Good, or change something?"
* "Good" books everything and closes. A change shows the updated plan in the same format. No step by step questions.
* The model never states times. Plans, week views and booking results are code generated messages; the model adds at most one short line, with any clock time stripped. The final plan reads each must do's time from its real MILES calendar event, or says "not booked".
* `place_blocks` books several must dos in order inside a window, default 10:00 to 21:00, with no overlaps. `move_block` and `remove_block` touch MILES events only. A block he deleted by hand is noticed and its id cleared.
* Calendar blocks have short titles of 2 to 4 words ("Linear algebra study", "Exam 3 prep: Lessons 19 to 21"). The details go in the event description: the full task, the reason (due date or exam), and any instructions ("work practice problems, not rereading"). A block from a study plan item also lists its topics and its source_material folder path.
* Conflicts are overlapping blocking events; touching events and all day events are not conflicts.
* Backlog: a task with no `scheduled_for` is in the backlog and has an optional `due_date`. Captures without a date go there. Suggestions, at most 3: overdue or due within 3 days, then exams in the next 7 days, then conflicts to fix, then the oldest backlog items. "Push this to the backlog" works during the review.
* Calendar roles: blocking calendars are the primary and MILES, plus any calendar not yet classified. Context only calendars (UF ACM Events, UF IEEE, both NSBE calendars, Holidays in United States) show in the week view marked club or holiday but never count as busy or create conflicts, unless an event or series is marked "I attend". He attends the ACM design team. As of Nov 25 2026 no ACM design team series is on any subscribed calendar (the only recurring one, "Design Team Meeting (Members Only)", Mondays 5 PM, comes from General NSBE Member Events), so nothing is marked attended until he confirms which series it is.
* Break rule: on UF break days (`UF_BREAKS` plus holiday calendar days), recurring classes (`CLASS_TITLES`: COP3530 shown as "DSA", CDA3101, MAS3114, ENC2256, matched as they appear on his calendar; none meet Wed Nov 25 to Sat Nov 28) and recurring club events show as "likely cancelled" and are never busy, never conflicts, never suggested.

## Evallo session filter (live)

**Status:** live since Nov 26 2026. Contract: [EVALLO_FILTER_CONTRACT.md](EVALLO_FILTER_CONTRACT.md). The permanent rules are above; its open follow ups are in [Planner bugs and security cleanup](#planner-bugs-and-security-cleanup).

## Quick capture (built Nov 25 2026)

* `POST /capture` turns one line into a task or list item, authenticated by a hashed capture key that can do nothing else and reads nothing. Every capture is logged with its source. Keys are created with `scripts/capture_key.py`, written to a mode 600 file, never printed.
* Later: classify captures into task, errand (Errand worker queue) or question.

## Tasks

**Status:** done and deployed Nov 27 2026 (Tasks tab, follow ups, the intent step). Contracts: [TASKS_CONTRACT.md](TASKS_CONTRACT.md) and `docs/work/`. What is still open is in [KNOWN_LIMITS.md](KNOWN_LIMITS.md), W1 first. The old queued text of the follow ups is in the archive.

## Nightly backup

**Status:** built on `feature/nightly-backup`, waiting to be merged and installed.

Planned only, not built; it is planned after the Evallo deploy is verified, and the phase starts with a plan and contract shown to him. Today nothing regular backs up `miles.db`: only manual pre deploy copies in `data/backups/`, all on the same SD card.

* A nightly systemd timer on the Pi takes a consistent copy of `miles.db` (the SQLite backup API, then `PRAGMA integrity_check`) together with `data/calendar_roles.json` (mode 600), into `data/backups/` as one dated set, and keeps 14 days.
* The newest set is copied to the Mac over Tailscale (the Mac pulls it, so the Pi never holds Mac credentials) so a failed SD card does not take the only copy. The Mac copy lives outside any git repo, mode 600.
* A failed or missing backup is reported like any other failed job (watchdog alert and push), once a day. Restores are tested once on a scratch copy before the phase is called done.
* Never backed up: tokens and login state (`evallo_state.json`, `.env`, keys), which stay out of the backup set.

## Tutoring session follow ups

**Status:** built Nov 28 2026 on `feature/tutoring-phase4` (backend, Mac repo and Nova), deploying. Contract and deploy plan: `docs/TUTORING_PHASE4_CONTRACT.md` and `docs/TUTORING_PHASE4_DEPLOY_PLAN.md` on that branch.

About 15 minutes after one of his Evallo sessions ends, Nova pushes "How did Alex's session go?" with Completed, Canceled (which emails the family) and Skip. One tap marks it in Evallo through the Mac. Only his sessions, only with his tap, 2 hour expiry, no retries, one push per session. After a confirmed Canceled, Nova asks once whether to add a "Reschedule" task. Rescheduling inside Evallo and auto marking from Zoom attendance (with Undo) are later items.

## Daily bug batch

**Status:** standing. The bugs found in real use each day are batched and fixed together at Small tier. The first fix is W1 in [KNOWN_LIMITS.md](KNOWN_LIMITS.md) (a misclassified no_change turn can still say a false "Done").

## Docs cleanup

**Status:** this branch (`docs/cleanup-2026-11-28`). One roadmap in this order, finished BACKEND_TODO items archived, a one page [REPO_MAP.md](REPO_MAP.md), the workflow rules in AGENTS.md, and the missing systemd units in INFRASTRUCTURE.md.

Carried into this item: the Mac repo's `src/tests/test_evallo_tools.py` imports a Pi module and fails to collect on that repo's main (see [KNOWN_LIMITS.md](KNOWN_LIMITS.md)).

## Acceptance harness

**Status:** queued. The reusable real model scenario harness described in [BACKEND_TODO.md](BACKEND_TODO.md#real-model-scenario-harness-queued-nov-28-2026) and the standing rule in AGENTS.md. Reviewers find design flaws; the harness proves behavior, and every deploy plan runs it before and after.

## CI

**Status:** queued. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## Cloud coding sessions

**Status:** queued. Portable backend work in supported cloud sessions; the environment rules are in AGENTS.md (Choose the environment) and [DEV_ENVIRONMENT_CONTRACT.md](DEV_ENVIRONMENT_CONTRACT.md). Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## SEC5

**Status:** queued. Login tokens that expire and can be revoked. Security work is tracked by its SEC number here with one neutral line; details stay out of this public repo.

## Own the tutoring calendar

**Status:** queued. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## Tutoring prep and tracking with Evallo Prepwork

**Status:** queued. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## Mission Control v0

**Status:** queued. Oversight of jobs and operational state; earlier notes are in `docs/work/dev-environment.md`. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## SEC4 and SEC6

**Status:** queued. Details stay out of this public repo.

* **SEC4:** approvals signed on the phone after Face ID.
* **SEC6:** outside text treated as untrusted, so it can never trigger actions.

## Release control

**Status:** queued. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## Tutoring reply handling and Evallo commands from Nova

**Status:** queued. These were the tutoring Phases 5 and 6 in the README's tutoring automation order, renamed so the numbers stop colliding.

* **Tutoring reply handling** (was Stage E replies): act on a student's or parent's reply to a reminder, with the same tap approval, expiry and no retry rules as the other approvals.
* **Evallo commands from Nova** (was tutoring Phase 6): Evallo changes started from Nova, through the Mac, with the same rules. The Mac side already exists for status changes.

Scope for both is written when their turn comes, with a plan and a contract shown to him. Multiple Nova chats (the old tutoring Phase 7) stays lower priority.

## Planner bugs and security cleanup

**Status:** queued. One batch of the planner and calendar bugs and the security and privacy cleanups that were queued separately.

### Planner fixes

Existing planner and calendar bugs found by the Codex review of the Evallo filter. Not caused by the filter and not fixed in that phase. Planned only, not built; the phase starts with a plan and contract shown to him.

* Break day classification: a verified, opaque primary calendar lesson whose title is a UF class code (for example `DSA`) during a UF break is treated as a likely cancelled class and removed from busy time (`calendar_roles.annotate`, `is_class`). A tutoring session must never be classified as a class.
* `plan_sessions` ignores all day events on owned calendars, so it can book a session during an opaque all day event.
* Transparent (marked free) and declined events still create conflicts in `find_schedule_conflicts`, consume slots in the session planner and are listed in voice answers without an availability marker, while `busy_blocks` correctly skips them.
* Unreadable calendars: `_fetch_events` drops a calendar that fails on any page, and `busy_blocks` raises only when every calendar failed. `place_plan` and `read_events` discard the list of unreadable calendars, so a failed primary calendar can make a whole day look free and the plan can place a task over a real lesson. A failed blocking calendar must stop planning or be stated plainly.
* One answer, one read: the nightly opening and voice answers load calendar state several times (suggestions, slot placement, rendering, the past and upcoming lists), so one answer can mix different decisions or clocks. Read once per answer.

### Evallo filter follow ups

Found by the Codex adversarial reviews of the Evallo session id filter and deliberately not fixed in that phase (he closed the review rounds). Planned only, not built.

* Mac pin flow (High): `evallo_pin.py` can print a response's `tutorName` verbatim, and the confirm step writes it into the pin file. If a token, cookie value or the supplied tutor id appears in that name it would be displayed and persisted. Refuse a name that contains the saved login's token, any cookie value or the supplied id, before displaying or writing it, and test it.
* Name rule: the slash form (`<student> / Lethanial Leveille`) ignores everything after a `|`, so `Algebra / Lethanial Leveille | Taylor <> Jordan Price` matches his name. End the match after his name, as the documented form does.
* Same second snapshots: the Pi truncates `fetched_at` before converting an offset form to UTC, so two snapshots in the same UTC second can replace one another. Convert to UTC first, then truncate; accept only `Z` or reject offsets.
* Mac pin file: opening a FIFO at the pin path blocks the hourly job and the load check. Open non blocking, then check the type.
* Mac timeouts: a response that keeps delivering chunks can hold one request past the 60 second budget (httpx timeouts are per wait, not total), and pin requests lost their default timeout. Enforce a total deadline and restore the pin timeout. Elapsed time should be captured when the last response arrives, not after validation.
* Retry button: the Pi's Retry for the hourly Evallo job is marked delivered at the Mac's check in, which now discards it silently. Make the Mac run the push job when told to retry, or remove the button for this job.
* Watchdog status for the hourly job: a success followed by a failure the same day leaves `agent_jobs.ok = 1`, and a success after midnight leaves the previous night's alert open. The report `detail` is also stored and sent in a push body as free text; cap and sanitize it.
* Count mismatch: an `unreadable` state from a count mismatch loses the known `verified_at`.
* Survivors report: sub second precision of Google `updated` is lost at the cutoff, and its header text about stale runs is wrong.
* Lows: a `Date` header with year 0026 is accepted, and the `--id-file` path check and `getpass` fall back are loose (details in the review output).

### Login failsafe and the Evallo token

Planned only, not built; there was no separate login failsafe item, so this one holds the Canvas login friction rules above (needs_login, one login per weekly batch, the keepalive) together with the Evallo token move below. The order against the other phases is set with him when it is planned.

* The Pi stops holding the Evallo token. Today it holds a copy, `~/miles/data/evallo_state.json` (copied Nov 23 2026, mode 600), read by the voice tool `set_session_status`, the nightly canary (`miles-evallo-canary.timer`) and the morning login check (`miles-evallo-logincheck.timer`). That copy is a known temporary exception to the Evallo token rule, kept for now so nothing breaks.
* Steps, in this order: (1) a voice session status change is queued on the Pi (an approval style request, nothing saved but the student, the time and the status) and carried out by the Mac agent, which already holds the login and the Evallo runner; (2) the nightly canary and the morning login check move to the Mac as launchd jobs; (3) once nothing on the Pi reads the file, the Pi copy is deleted and the Pi's readers of `evallo_state.json` are removed from the code; (4) the login expired message, README and `scripts/evallo_login.py` never mention copying the file (already fixed in the Evallo filter phase).
* Done means: no file on the Pi holds an Evallo token or cookies, with a test that fails if code on the Pi reads `evallo_state.json`.

### Tailscale hostname cleanup

Small cleanup, planned only. The Pi's raw Tailscale address is written in scripts and docs (`docs/INFRASTRUCTURE.md`, the login and watchdog scripts, `docs/BACKEND_TODO.md`, `docs/INCIDENTS.md`, `docs/SESSION_START.md` and the Mac agent's helper scripts). Switch them to the Tailscale MagicDNS hostname of the Pi instead of the address, so the address is not repeated in the public repo and a changed address breaks nothing. The address already in public history stays as is (decided Nov 26 2026). Check each script still reaches the Pi by hostname from the Mac and from the phone's network before removing the address, and keep the address out of any new commit.

### Private data out of the repo

* Done in the history scrub of Nov 26 2026: the real calendar ids moved out of `src/config.py` into the untracked `data/calendar_roles.json`; tests and docs use fake names, emails and ids.
* Queued: a pre push scan script (`scripts/pre_push_scan.py`, run by a git pre push hook) that checks the commits being pushed for names from the Evallo student list, emails, calendar ids, tutor ids, tokens and keys, and refuses the push on any hit. Also decide whether to rewrite the public history that still holds one student's first name (committed before Nov 26 2026); that needs a force push and his approval.

## Model audit: local models for background jobs

**Status:** queued.

Planned only, not built. The phase starts with a plan and contract shown to him, as every phase does.

* Audit first: list every Claude API call in MILES and classify each one by (1) whether it needs low latency, (2) whether it uses tools, and (3) how much answer quality matters. The result is a table in the contract, one row per call site, with the model it uses today and its logged cost and latency.
* Stays on Claude: the voice loop and anything with heavy tool use.
* Candidates for Ollama on the Pi: memory candidate extraction, the Gmail recruiting watcher's email sorting, session summaries and chat auto titles, plus anything else the audit shows is background, tool free and tolerant of an occasional miss.
* Pick a local model that fits the Pi's memory alongside the services already running (measure free memory with the services up before choosing). Local jobs run at night, under the same CPU and memory limits as the errand worker, so they never compete with the voice loop.
* Every local result is validated (structured output checked against a schema, plus a sanity check per job). A failed validation falls back to Claude for that item, and the fallback is logged.
* Benchmark before switching anything: run the local model and Claude on the same real logged inputs, and show him accuracy, time and cost side by side. A job moves only after he approves its numbers, one job at a time.
* Folds in the open Haiku 5.5 versus Haiku 4.5 comparison for the voice loop, from the logged cost and latency of real turns. That comparison covers the voice loop only; nothing else changes model without its own numbers.
* The audit also defines the provider interface (one call shape for text, structured output and tools, with the model, timeout, validation and fallback chosen per call site) that later fallback providers would plug into. The phase defines the interface; it does not add other providers.

## School: assignments and study plans

**Status:** queued.

### Canvas feed

* Store my Canvas calendar feed URL as a secret, never in the repo.
* Poll it daily. Upsert assignments by the event UID so moved due dates update cleanly. Log what changed.
* Parse the course from the event data. Keep a `courses` table: id, code, name, semester.
* `assignments`: id, course_id, title, type (homework, quiz, exam, project, other), due_at, weight (nullable), canvas_uid (unique), source, updated_at.

### Syllabus import (once per semester)

* A CLI command that takes a syllabus PDF, uses Claude to extract grade weights and exam dates, shows me the result, and only writes to the database after I confirm.

### Study plan generator

* For each exam, I provide or confirm the list of topics it covers.
* `topics`: id, course_id (nullable so non course topics like interview prep work), name, stage (new, learning, reviewing, solid), confidence, next_review_at, ease, source_material (path to PDFs or notes).
* `topic_prereqs`: topic_id, prereq_id.
* Order topics by prerequisites (topological sort).
* Schedule backward from the exam date: learn sessions for new topics first, then spaced reviews roughly 7, 4, 2, and 1 days before, scaled by exam weight and how many days are left.
* Final 1 to 2 days: a mixed, timed practice test across all topics.
* Plan items become rows in `tasks` with source study_plan. The nightly review shows them and reschedules anything I skip, so the plan stays realistic instead of letting work pile up.

## Routine

**Status:** planned Nov 26 2026.

Planned only, not built. The phase starts with a plan and contract shown to him. When planning it, ask him his workout split, his target bedtime and his usual wake time.

* Workouts: MILES schedules 4 workout blocks a week on the MILES calendar, each labeled with that day's split. After each block it checks Hevy for a logged workout. If there is none, it removes the block and reschedules later that day or later in the week, keeping the split order and 4 per week.
* Hevy: MILES may create and update Hevy routines, including progressive overload targets and the ab wheel. Logged workouts are read only unless he asks to fix a specific one and confirms. Needs Hevy Pro and an API key stored like the other secrets (a mode 600 file, never printed).
* Sleep: a wind down block before his target bedtime; the nightly review plans around his real wake time from Oura; later, morning blocks shift when Oura shows he woke late.
* Nova Action Button (replaces the iPhone capture Shortcut; the Mac hotkey keeps using capture):
  * A Nova App Intent for the iPhone Action Button that starts voice right away.
  * The result shows in a small card at the top of the screen without opening the app, and a cyan indicator shows in the Dynamic Island while it works.
  * A Control Center control with Nova's icon runs the same intent.
  * From the lock screen without Face ID it runs at the lowest clearance: add tasks and list items, and ask what is next. Anything higher requires Face ID.

## Jobs

**Status:** planned Nov 26 2026.

Planned only, not built.

* First a Gmail recruiting watcher that reads only recruiting emails. It proposes tasks with deadlines and prep blocks for online assessments and interviews, through his approval checkmarks, and logs every application and online assessment to PROWL.
* Then the `job_apply` form filler moves here from the errand worker, starting with Greenhouse and Lever, Workday later, and always stopping before submit.

## Learning: learn and quiz sessions

**Status:** queued.

### learn_session (new material)

1. Ground everything in my actual course material (the topic's source_material), not generic explanations.
2. Open with one quick "take a guess" question before teaching.
3. Check prerequisites with 2 quick questions. If I miss them, review the prerequisite first.
4. Teach in small chunks: explain one idea, show one worked example, then give me a similar problem. Reduce help each round until I solve one unaided.
5. Ask me to explain the idea back in my own words and point out gaps.
6. On close: generate 3 to 5 recall questions, save them to the topic, set stage to learning with low confidence, and schedule the first review.

### quiz_session (review)

* Pull topics whose next_review_at is due.
* Ask recall questions. I answer before seeing anything. Grade each answer 0 to 5 with a short explanation.
* Update ease, confidence, next_review_at, and stage using a simplified SM2 algorithm. Log every review in `review_log`.
* Missed topics come back sooner. Topics that stay strong move to solid.

## Kitchen

**Status:** planned Nov 26 2026.

Planned only, not built.

* A hands free voice cooking mode on the Pi: one step at a time, "next", and timers.
* Meal planning for his bulk from his Cronometer totals; ingredients are added to his grocery list; a weekly meal prep block on the calendar.

## Directions

**Status:** queued.

* Add a client action type: the backend can return an action object that Nova executes on the device, since only the phone knows my location.
* `open_maps(destination, mode)`: Nova opens `maps://?daddr=<destination>&dirflg=<w|d|r>` (walking, driving, transit).
* `get_eta(destination, mode)`: Nova uses MapKit `MKDirections` with my current location and speaks or shows the distance and time.
* `saved_places` table so short names work (for example Marston, home, RecWell).

## Nova on iPad and Mac, with a design system

**Status:** queued. The design system is a new part of this item: shared components and tokens for Nova across iPhone, iPad and Mac. Its scope is written when its turn comes.

* Add the iPad destination to the Nova target and adapt layouts for large screens (sidebar with sessions, main area for the active session).
* Check whether Nova runs on my Mac as an iPad app on Apple Silicon. Tell me what works and what does not.
* Later idea, do not build yet: a share extension so I can send a GoodNotes page export or photo to a learn or quiz session for Claude vision to check my handwritten work.

## Full security audit

**Status:** queued. Runs before the errand worker uses any real account or runs unattended.

Before the errand worker uses any real account or runs unattended, a full security review by an independent review subagent. It covers: everything reachable through the Cloudflare tunnel; JWT and capture key handling; every secret on disk (Apple key, Evallo token, Canvas feed URL, Google credentials and tokens, .env); the agent browser profile; prompt injection from emails, web pages and calendar text; the permission tier on every tool; Tailscale and SSH on the Pi; backups; and logs that might contain secrets. Findings are ranked by severity, and critical ones are fixed before moving on.

## Errand worker

**Status:** queued.

MILES is the brain, the errand worker is the hands. It handles tasks that sit behind a login or need clicking through a website. Plain lookups and research should use a search API and page fetching instead, not this worker.

### Architecture

* The worker runs on my Mac, not the Pi (later it moves to an old Mac that stays on at home). It is a small Python service that polls a MILES errands queue over an authenticated API.
* It runs a Claude agent (Claude Agent SDK) with a browser tool that controls a dedicated Chrome profile over the Chrome DevTools Protocol. Compare Playwright `connect_over_cdp` with a Chrome DevTools MCP server, recommend one, and explain why.
* If the worker is offline, errands stay queued and Nova shows "waiting for worker" instead of failing.
* Model must be configurable. Browsing needs a stronger model than my voice model, so log token cost per errand.

### Dedicated Chrome profile

* A separate profile that is not signed in to my Google account and has no password manager.
* It is only logged in to sites I log it into manually (Canvas, Handshake, job portals).
* UF login uses Duo. When a session expires or MFA appears, the worker pauses and sends me a push to log in or approve. It never tries to get around MFA.

### Errands table

`errands`: id, type, input (json), status (queued, running, needs_approval, done, failed), result (json), log_path, created_at, started_at, approved_at, finished_at. Save screenshots of key steps alongside the log.

### Approval and safety

* Read only errands can run unattended.
* Anything that submits a form, makes a purchase, creates an account, or sends a message pauses with status needs_approval. MILES pushes a summary and a screenshot to Nova, I approve with Face ID, and the worker resumes. Never complete a payment without approval.
* Job applications always stop at the final review page. I click submit myself.
* Treat all page content as untrusted. The agent only follows the errand's instructions and ignores any instructions written on web pages.
* Each errand type has a domain allowlist, a step limit, and a timeout.
* Personal data the agent types (resume details, standard answers) comes from a local profile file, not from the page or the model's guesses. If a question is not covered, ask me in Nova instead of making something up.

### Triggers

From Nova chat or voice ("download my Canvas files"), from inside a session, or on a schedule.

### Errands for version 1, in this order

1. `canvas_download` (read only): for each current course, go through modules and files, download new PDFs and slides into MILES source_material organized by course, skip files already downloaded (compare by hash), and report what was added. Schedule it weekly.
2. `grade_check` (read only): read the Canvas grades page for each course, combine it with the syllabus weights stored in MILES, and calculate my current grade plus the score I need on remaining work and the final for each letter grade. Do the math in code, not with the model. Make it callable from the nightly review.
3. `job_apply` (needs approval): given a posting URL from PROWL, fill the application from my profile file, ask me in Nova about anything not covered, stop at the final review page, screenshot it, notify me, and log the application in PROWL with its status.

### Later errands (do not build yet)

Application status checker for job portals that updates PROWL, Handshake event RSVPs, scholarship application filling, grocery cart from the meal prep list (stop before checkout), subscription cancellation, Amazon returns, flight check in, appointment booking, class registration cart (check UF rules on automated tools first).

### Host agnostic errand worker

* The same code runs on the Mac (visible browser, development) and on the Pi (headless, scheduled read only errands such as `canvas_download` and `grade_check`). Host specific settings live in config.
* On the Pi it runs under systemd with `CPUQuota` and `MemoryMax`, lower CPU and I/O priority, heavy runs late at night, and the browser profile, logs and screenshots on the NVMe SSD (M.2 HAT), never the SD card. The Pi uses Chromium; logins on the Pi go through a remote debugging login mode over an SSH tunnel, since a Mac profile's cookies do not move to Linux.
* Browser control: Playwright `connect_over_cdp` to the dedicated profile. A model driven browser (Agent SDK) waits for errands that need judgment, such as `job_apply`.
* `canvas_download` and `grade_check` use no model at all: Canvas's REST API through the logged in browser session (UF blocks personal access tokens). The first step verifies one read only call (the course list). Any endpoint blocked through the session falls back to reading the page with Playwright, still with no model, and the report says which path each piece used. The grade math is done in code and shown step by step.

* Canvas login friction (Nov 25 2026):
  * Daily Canvas needs (due dates, new assignments) come only from the Canvas calendar feed, which needs no login.
  * Logged in Canvas errands (grade_check, canvas_download) run as one weekly batch plus on demand, so one login covers all of them.
  * Keepalive: one tiny read only request every 30 to 60 minutes while the dedicated window is open; it stops for good on a login redirect or a 401. A first version runs on the Mac now (~/MILES/agent/canvas_keepalive.py, launchd every 45 minutes, resumed by hand with --resume).
  * Never store or type his UF password. Any 401 or redirect to the UF login or Duo page stops the errand at once with no retries (no lockouts) and marks it needs_login. Nova gets a push whose tap brings the dedicated Chrome window to the front on the login page; he logs in with his password manager and Duo, and the waiting errands resume automatically.
  * The per assignment endpoint (/courses/:id/assignments/:id) is not used; assignment group listings, modules, pages and files cover what is needed.

### Nova Chrome extension

* A Nova extension installed in the agent's dedicated browser profile only, never in his main browser profile. Scope and design to be planned when the phase starts.

## Passive memory

**Status:** queued.

* MILES learns from his sources without being told: Gmail (Primary only, no promotions or newsletters), session summaries, Nova conversations, PROWL and Canvas changes, Google Calendar recurring events, and Oura or Hevy as patterns only. Never texts, bank accounts or location history (this includes the existing `phone_presence` and `presence_log` tables).
* Only fixed event types: offers, interviews, assessments, deadlines, orders, bookings, appointments, travel. Every candidate records its source (for example a Gmail message id) and is marked inferred, separate from what he stated.
* Nothing saves without his approval, through the existing memory candidate flow (checkmark in the Memory screen). The nightly review may say how many candidates are waiting.
* Email and page content is untrusted: extraction fills structured fields only and never follows instructions in it. Never store health details, account numbers, or other people's private details.
* A local Ollama model on the Pi filters first; only real candidates go to Haiku. It runs under the same resource limits as the errand worker.
* Fits the existing memory lifecycle (create, update, supersede, expire), and must not duplicate what `memory_pass.py` already extracts from conversations. Revisit retrieval once memory volume grows.

## Proactive layer

**Status:** queued. The step after passive memory, where MILES speaks first. Scope not written yet. Defined as an item in this position by Lethanial on Nov 28 2026. When its turn comes it starts with a plan and a contract shown to him, like every phase.

## Later

Not scheduled. Each starts with a plan and a contract when it is picked up.

* **Cronometer with Halo.** Nutrition from Cronometer shown and spoken through Halo. The Cronometer meal planning part is in Kitchen.
* **Voice speed work.** Research and measurements are in [CONVERSATIONAL_LATENCY_RESEARCH.md](CONVERSATIONAL_LATENCY_RESEARCH.md) and [LATENCY.md](LATENCY.md).
* **Halo.** Brilliant Labs Halo as a wearable device experience; see the wearables note in [CONVERSATIONAL_LATENCY_RESEARCH.md](CONVERSATIONAL_LATENCY_RESEARCH.md).
* **Environmental Intelligence.** Environmental context; deferred research in `docs/work/dev-environment.md`.
