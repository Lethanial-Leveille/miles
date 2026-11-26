# AGENTS.md

Rules for coding agents (Codex, Claude Code and their subagents) that apply to this repo only. The general workflow (plans, contracts, Codex reviews, builders, deploys) is global: `~/.claude/CLAUDE.md` for Claude and `~/.codex/AGENTS.md` for Codex. This file adds MILES rules on top and wins where they differ. Read CLAUDE.md first; it holds the full repo conventions and production warnings.

## Production database and services
- The Pi services, the Cloudflare tunnel and `~/miles/data/miles.db` are production. Do not modify the running services, edit `~/.cloudflared/config.yml`, or write to the production database unless Lethanial explicitly asks.
- Before any migration or deploy, take a fresh backup into `~/miles/data/backups/` named `miles-before-<what>-<YYYYMMDD-HHMMSS>.db`, and tell him before anything touches production.

## Secrets
- Never print a secret (API keys, JWTs, OAuth tokens, capture keys, `.env` values) in command output, logs or replies. Write it to a mode 600 file (on the Pi under `~/.miles-keys/`) and report only the path and key id. Verify by hash or length, never by value. If one is printed, say so and rotate it.
- Never commit secrets or private data; the list is under Things to Never Commit in CLAUDE.md. Run `git status` before `git add`.

## Private data stays out of the repo
- This repo is PUBLIC. Never put real names of students, clients or other people, real email addresses, or real calendar ids in code, tests, docs, comments or commit messages. Use obvious fakes: first names like Alex and Jordan, addresses like `primary.fake@example.com`, ids like `fake-calendar-01@example.com`.
- The real calendar ids live in `data/calendar_roles.json` (untracked, mode 600), loaded by `src/config.py` through `CALENDAR_ROLES_FILE` (env `MILES_CALENDAR_ROLES_FILE`). The placeholder ids in `config.py` are fakes. Tests set the variable to a missing file so they never read the real ids.
- Before any push to GitHub, scan the commits being pushed (added lines, commit messages, file paths and branch names) for names, emails, calendar ids, tutor ids, tokens and keys, and say so in the deploy plan. Pushing is never done on a guess.

## UF password
- Never store, type or ask for his UF password, and never try to get around Duo or MFA.
- Any 401, `invalid_grant`, or redirect to the UF login or Duo page stops the job at once with no retries (no lockouts) and marks it as needing login. He logs in himself.

## Evallo
- Evallo's two way sync writes the whole org's sessions onto his primary Google Calendar and must stay on.
- NEVER delete, move or edit any calendar event carrying Evallo's fields (`evalloEvent`, `evalloSessionId`, `orgId`) from any code, script or agent, not even a one off script. The calendar layer must refuse such writes, with a test.
- An Evallo event counts as his only if its session id is in the verified set from the Mac. If the set is stale, only his full name in the tutor position of the title counts, never "Lethanial" alone. Unknown means busy, never free.
- The Evallo token stays on the Mac: never printed, logged or sent to the Pi. Only session ids go to the Pi.
- Known temporary exception: the Pi currently holds a copy of the Evallo login, `~/miles/data/evallo_state.json` (copied Nov 23 2026, mode 600), used by the voice tool `set_session_status`, the nightly canary and the morning login check. Do not copy it again, do not add readers of it, and never print it. The plan to remove it is the roadmap item "Login failsafe and the Evallo token" in docs/ROADMAP.md.

## Contracts
- The shapes the backend and the app share live in `docs/*_CONTRACT.md` (`SESSIONS_CONTRACT.md`, `TASKS_CONTRACT.md`). Change them only with his approval.
- Contract reviews also check for conflicts with the amendments in `docs/ROADMAP.md`.

## Roadmap
- `docs/ROADMAP.md` holds the phase plan and its amendments. One phase at a time; stop after each for real use.
- Before any Phase 6 errand runs unattended, the full security review in `docs/ROADMAP.md` must be done.
