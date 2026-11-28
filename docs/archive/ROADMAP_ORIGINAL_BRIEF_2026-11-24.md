# MILES roadmap, original brief and superseded text (archived Nov 28 2026)


Kept as written. [ROADMAP.md](../ROADMAP.md) is the current order and wins wherever they differ.


## Context

You are working in my MILES repo (Python 3.13 backend on a Raspberry Pi 5, FastAPI with JWT, SQLite in WAL mode, a native Anthropic tool registry with permission tiers checked against a clearance rank system) and my Nova iOS app (SwiftUI, Keychain JWT, Face ID, talks to miles.lethanial.com over REST and WebSocket, already receives APNs push notifications).

Before writing any code:

1. Read the repo and summarize how conversations, tools, permission tiers, push notifications, and the Google Calendar integration currently work.
2. Propose a plan for Phase 1 only, including schema changes and files you will touch, and wait for my approval.
3. Follow the existing conventions (tool registry, tiers, logging, timing_log, service layout). Do not introduce new frameworks unless you ask first.

Working rules:

* Build one phase at a time. Stop after each phase so I can use it for real for a few days before we continue.
* Do not overengineer. Build the simplest version that works, and only generalize where this prompt says to.
* After each piece, explain what you built and why in plain English so I can explain every part myself in an interview.
* Write tests for the scheduling, spaced repetition, and sync logic.
* Model choice must be configurable per session type. I am cost conscious, so default to the cheapest model that does the job well and tell me where a stronger model is worth it.

## Core idea

A "session" is a dedicated chat with its own purpose, system prompt, and tool set, separate from my main Nova conversation. It works like a support chat: it opens with a purpose, stays open until I close it, then saves a summary. Everything below is a session type.

The backend owns all logic. Clients (Nova on iPhone, iPad, and Mac) are thin and just render sessions and run client actions. This matters because I will mostly use study sessions on iPad or laptop, and the nightly review on whatever device gets the push.

## One source of truth

* MILES database owns courses, assignments, exams, tasks, and study topics. This is the truth.
* Canvas calendar feed is input only. MILES pulls it and updates its own records.
* Google Calendar is time only. MILES writes blocks to its own MILES owned calendar. Never treat calendar events as the source of an assignment or task.
* Notion is not part of this system.
* Any change I want goes through Nova, never by editing a copy.

## Phase 1: session framework and nightly review

### Schema (adapt to existing tables where they already exist)

* `sessions`: id, type, status (open, closed, missed), created_at, closed_at, summary, push_sent_at, reminder_sent_at
* Session messages: reuse the existing conversation storage, keyed by session_id
* `tasks`: id, title, notes, scheduled_for (date), status (pending, done, skipped, moved), must_do (bool), source (manual, canvas, study_plan, meal_prep), assignment_id (nullable), topic_id (nullable), calendar_event_id (nullable), created_at, updated_at

### Session type registry

Each session type defines: system prompt, allowed tools, an opening message builder (what it says first, using current data), and close behavior (what it saves when I close it). Register `nightly_review` now. Design the registry so `learn_session`, `quiz_session`, `sunday_meal_prep`, and `weekly_review` can be added later by adding an entry, not by changing the framework.

### Nightly review flow

1. A systemd timer on the Pi fires at 21:30 America/New_York, creates a `nightly_review` session, and sends an APNs push whose payload includes session_id and type.
2. If I have not opened it by 22:30, send one reminder push. If it is still unanswered at 03:00, mark it missed.
3. Tapping the push deep links Nova into that session (for example `nova://session/<id>`), not the main chat.
4. Conversation flow:
   * Show today's tasks and ask what got done. Update each status.
   * Show what is coming up in the next 7 days (assignments, exams, study plan items).
   * Ask me to pick 2 or 3 must dos for tomorrow, plus optional tasks. Cap must dos at 3.
   * Schedule must dos as blocks on the MILES owned Google Calendar, avoiding conflicts with my existing events.
   * Confirm the plan, then close when I say I am done.
5. On close, save a short summary (planned vs done counts, what moved, anything I said that matters for tomorrow).

### Tools for nightly_review

`get_tasks_for_date`, `update_task_status`, `get_upcoming_items`, `create_tasks`, `schedule_block` (MILES owned calendar only), `close_session`. Assign permission tiers following existing conventions.

### Nova changes

* Handle the push tap and deep link into the session.
* A sessions list showing open sessions, with a Close button on each.
* A session view that reuses the existing chat UI but shows the session type and purpose at the top.

## Resume bullets

Do not edit my resume. At the end of each phase, write a short changelog entry focused on measurable impact (what it does, what it improved, numbers where possible) so my job tracker can propose bullet updates for me to approve.

## Start

Begin by reading the repo and giving me the summary and Phase 1 plan. Do not write code until I approve the plan.

## Superseded amendment text


### Old order and working rules

* Order: nightly review v2, then the Evallo session id filter, then the nightly backup (see Nightly backup), then the Tasks tab fix (see Tasks tab follow ups), then Phases 2, 3, 4, 5 and 6 in that order, then Phase 6.5, then Phase S before any errand runs unattended, then Phase 7.
* Each phase: plan shown before building, tests, one builder subagent at a time, an independent review at the end, a fresh database backup before its own separate deploy, and a warning before anything touches production. If a usage limit hits, stop at the end of the current phase, never half built.
* Phase 2 and 3 use the MAS3114 Exam 3 (Monday Nov 30 2026) as their first real data: syllabus import, study plan, first learn and quiz sessions. Source material lives in `~/MILES/source_material/MAS3114/Exam3/`. Every Lesson 19 to 27 topic starts as new; determinants and null space are the prerequisites to check first.

### Old model plan

* Main session and builder subagents: Sonnet. Read only search and exploration subagents: Haiku.
* Opus only for independent reviews and design decisions, and only after asking.
* Runtime models stay configurable per session type and default to the cheapest that works (`SESSION_MODELS`, Haiku 4.5 for the nightly review). Checking handwritten work needs Opus 5.5 (measured in docs/STUDY.md).

### Tasks tab follow ups (queued Nov 25 2026, now done)

One contract change to docs/TASKS_CONTRACT.md, planned, reviewed and deployed as one phase. Plan and contract first, Codex contract review, no building until approved. Not built.

* Found by the Codex contract review and checked against the live code:
  * Real bug, low severity: `POST /tasks/{id}/complete` has no day parameter. A tap at 4:01 a.m. on an overview loaded before 4:00 ticks the new day's occurrence of a daily task, and the previous day's tick is lost. Fix: Nova sends the overview's `today`; the Pi answers 409 with the current day on a mismatch and Nova refreshes.
  * Contract wording only: the reopen route says it removes "today's completion", but for a one time task the code removes all of its completion rows, and a daily task removes today's. Fix the wording to match the code.
  * Checked and not bugs: moving a task to the backlog after Google already deleted the block (the retry finishes the move), and reopening a one time task that was completed on an earlier day.
* Start date for daily tasks (found Nov 25 2026: "add reading as a daily task starting Monday" drops the date today, because the voice tool, the create route and `tasks_for_day` have no start date, and the tool then confirms the task as if it were saved in full):
  * A nullable `start_date` (YYYY-MM-DD) on daily tasks, by migration. `tasks_for_day` includes a daily task only on days on or after its `start_date`; no start date means every day, as now.
  * The voice tool `task_add` takes an optional `start_date` for a daily task and rejects one on a one time task with a plain message. Dates are the 4 a.m. days.
  * `POST /tasks` can create a daily task with an optional `start_date` (today or later). The Task object gains `start_date`. Nova's add sheet gets a daily option with a start date picker.
  * To settle in the contract: what the complete and reopen routes answer for a daily task before its start date, how the voice reads of today's tasks and the nightly review treat one, and what `GET /tasks/overview` shows for the days before the start.
* Guard for dropped details, for the voice tools and the create route (found the same day): when a request includes something a tool cannot store, the tool result must say so plainly ("Saved X. NOT saved: Y. Tell him this."), so the model tells him what was dropped and never confirms it. Because the model can only pass fields the tool defines, each task tool gets an optional text field for the part of the request it could not fit into any field, and its description tells the model to fill it; a non empty value always produces the "NOT saved" line. Start with the task and list tools, then audit the other write tools for the same gap. Tests: a request with an unstorable detail returns the NOT saved line; a fully stored request returns none.
