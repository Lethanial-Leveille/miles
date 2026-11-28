# Shared engineering instructions

This file is the shared engineering authority for Claude and Codex. [CLAUDE.md](CLAUDE.md) covers Claude coordination under these rules. Owner instructions, approved contracts and phase gates continue to apply.

## Start with the assignment

Check the task's scope, authorization, environment, branch, HEAD and working tree before editing. Use the [repository map](docs/REPOSITORY_MAP.md) to open only the relevant contract, implementation and tests. Use [isolated development commands](docs/DEVELOPMENT.md) and the [session workflow](docs/SESSION_START.md) as needed. Source and tests describe current behavior. An approved contract governs intended behavior. Surface any relevant conflict before acting and correct stale descriptive docs within scope.

## Tiers

Pick the highest tier that applies and reassess if the scope changes.

| Tier | When it applies | What to do |
|---|---|---|
| Small | A UI fix or a clear bug | Add a test. Review your own change. |
| Medium | A new tool, screen or job | Write a five line plan. Get one Codex review. |
| Large | Migrations, texting people, changing Evallo, money, auth or secrets | Write a short contract and get Lethanial's approval. Get a Codex review and one focused verification. |

## The loop

1. Build with tests.
2. Codex reviews and rates each finding Blocker, Should fix or Note.
3. Fix every Blocker and every Should fix that has one obvious answer.
4. Codex verifies only those fixes. Then stop.

Run it with `scripts/loop` (see `scripts/loop --help`). It does steps 2 to 4 and prints the final report.

- Two review rounds at most. Anything still open goes in [docs/KNOWN_LIMITS.md](docs/KNOWN_LIMITS.md), unless it is a Blocker.
- Codex reads only the diff, the files it touches and the contract. It runs read only, never deploys and never touches production. Before a run, check usage with `python3 ~/.codex/usage.py` and tell Lethanial if it is over its threshold.
- Use Astra High for the review. Use Opus only with his approval.

## What counts as a Blocker

A finding is a Blocker only if it can realistically happen in his real life and would do one of these:

- lose data
- contact a real person wrongly
- change Evallo wrongly
- spend money
- say something false

Everything else is Should fix or Note.

## When to ask Lethanial

Ask only when:

- two valid fixes change behavior he would notice
- the scope grows past the plan
- a Blocker survives round two

Otherwise decide, record the reason and keep going.

## Limits

- At most two feature branches are open at once.
- Do not add new workflow documents unless he asks.
- The final report has five parts: built, found, fixed, deferred, questions for him.

## Switching agents

Either agent can pick up an approved assignment. Use the [continuation procedure](docs/SESSION_START.md) and the existing contract, branch, handoff and test evidence. Keep one active writer per worktree. A change of agent carries forward all approved scope, holds and permission limits.

## Choose the environment

- Prefer a supported cloud session for portable backend work, research, planning and synthetic tests when the correct revision and isolated tooling are available.
- Prefer Mac CLI for Xcode, unpublished local work and authorized private offline diagnostics. For private diagnostics, have the owner run local scripts on approved data outside the repository and cloud sync. Give the agent only owner reviewed, sanitized results.
- Live Pi diagnostics and production deployment each require separate owner authorization. Diagnostic authority is bounded by the approved time, actions and resources. Use existing observations where possible. Service, hardware, gain or configuration changes and acquisition of occupied devices require explicit authority.
- If the current environment is unsuitable, explain the specific limitation, recommend the appropriate environment and complete any safe independent work before handing off.

## Production and scope

- The Pi services, Cloudflare tunnel and `~/miles/data/miles.db` are production. Keep ordinary work isolated. Do not touch running services, live data, or production settings without explicit owner authority and role permission. The tunnel is dashboard managed. Leave `~/.cloudflared/config.yml` alone.
- Before any deployment or migration against real data, warn the owner, take a fresh backup at `~/miles/data/backups/miles-before-<what>-<YYYYMMDD-HHMMSS>.db` and prepare rollback. Never restart the voice service during a conversation. Follow the [permanent Evallo restart procedure](docs/ROADMAP.md#evallo-rules-permanent) in every deploy plan that restarts the Pi.
- Codex never deploys, accesses production or starts another Codex run. Push only with explicit authority. Use normal Git hooks and guards. Report a guard failure and follow the authorized workflow.
- The owner grants standing authorization for routine local Git branch creation, staging and commits within approved development assignments. Check the branch and working tree, and stage only intended paths. If the sandbox blocks an authorized operation, request approval for that exact Git command. Keep Git hooks and guards active. This authorization is limited to the approved development branch and excludes pushes, changes to `main`, history rewrites, and broader filesystem or network access.
- Work on the authorized branch or worktree. Keep changes within scope. Nova is a separate repository and requires separate scope. Follow the [approved roadmap](docs/ROADMAP.md) and amendments one phase at a time, with a real use checkpoint between phases. Complete the roadmap security review before permitting any Phase 6 errand to run unattended. Leave the owner's resume untouched.

## Secrets and private data

- Never print, echo, log or display secrets in command output, logs, replies, files or Git. This includes tokens, API keys, passwords, JWTs, cookies, private keys, environment values, Keychain items and browser session state. Debug with names of headers or keys, booleans, counts, lengths or short hashes. Never write probes that print values or circumvent a secret guard.
- Pass secrets through stdin, a pipe or a mode 600 file. Keep secrets out of arguments and shell history. Where a Pi key file is explicitly authorized, use `~/.miles-keys/` and report only its path and key ID. If a secret is exposed, immediately identify the destination without repeating the value and recommend rotation.
- This repository is public. Never commit secrets, real people’s names, emails, calendar or tutor IDs, private datasets, real databases, recordings, voiceprints, enrollment data, model weights, wake models, compiled Whisper binaries, build artifacts, tunnel credentials or Google and Oura OAuth files. Use obvious fakes such as Alex, Jordan and `primary.fake@example.com`. Keep real mappings, credentials, raw diagnostic evidence and browser state out of Git, agent handoffs and automatic export.
- Real calendar roles live in untracked, mode 600 `data/calendar_roles.json`, selected through `MILES_CALENDAR_ROLES_FILE` in `src/config.py`. Tests select a missing file to avoid real identifiers. Check `git status` before staging and stage specific paths. Before an authorized push, check visibility and scan added lines, messages, paths and branch names for private identifiers and credentials. Report the scan in the deploy plan when applicable.

## UF and Evallo

- Never store, type or ask for the owner's UF password or bypass Duo or MFA. A 401, `invalid_grant` or a redirect to UF login or Duo stops the job immediately without retries. Mark it as needing login and let the owner log in.
- Keep Evallo two way sync on. Never delete, move or edit an event with `evalloEvent`, `evalloSessionId` or `orgId` through any code, script or agent. The calendar write boundary must refuse it before calling Google and have a regression test.
- Use the Mac verified session set and the [permanent Evallo rules](docs/ROADMAP.md#evallo-rules-permanent) for ownership, stale and out of window fallback, and availability flags. Keep the last verified set when stale. The fallback requires the owner's full name in the tutor position. A first name alone is insufficient. Keep unresolved availability conservative and respect the rules for transparent or declined events.
- The Evallo token stays on the Mac. Only verified session IDs and their date range go to the Pi. The existing temporary mode 600 `~/miles/data/evallo_state.json` on the Pi supports `set_session_status`, the nightly canary and the morning login check. Never print it, copy it again or add readers. See the [retirement plan](docs/ROADMAP.md#login-failsafe-and-the-evallo-token-planned-nov-25-2026).

## Shared contracts and verification

The backend owns shared API contracts. Obtain owner approval for changes and check the relevant roadmap amendments. Keep Nova copies paired with the intended backend revision using the [contract sync instructions](docs/REPOSITORY_MAP.md#shared-api-contracts).

Explain the purpose of code changes and new concepts briefly before building. Keep changes small and logical, favor flat Python, explain the reasons behind code decisions in comments, and keep refactoring within scope. Avoid introducing APScheduler without revisiting the decision. For feature behavior that needs human or hardware validation, use [the testing ladder](docs/TESTING.md) for examples, scripted checks and owner validation. Spell numbers as words in Nova's spoken responses. Use the owner's preferred name by default and reserve the nickname for rare moments of genuine success. Use hyphen free owner facing prose.

Keep relevant topic docs in sync with affected behavior. Update the public README in the same commit when a feature, stack, hardware, endpoint or headline measurement changes.
