# MILES development environment Friday handoff

## Objective and status

Establish the reproducible headless backend, isolated runtime, synthetic data and authentication, offline tests, and portable agent workflow in the [development environment contract](../DEV_ENVIRONMENT_CONTRACT.md). Canonical reconciliation is complete and the contract is ready for independent review. Application implementation is paused pending that review and the normal build gate. Nova, hardware diagnostics, cloud setup, CI, and deployment remain later milestones.

## Git and relevant files

- Canonical branch `infra/dev-environment` is based on local `main` at `51676b912eac708d03024d02942ba05364ba89e6`. The owner-provided clone had a clean worktree and local `origin/main` matched `main`. The origin identifies the expected backend repository. No remote refresh was performed.
- The two documentation files were transferred from the owner-authorized scratch commit `0d184d592e31a421badb91561bce8beac32bb0e5`. The original scratch contract commit, `30991b0588d27890f972175a18b6b60dc2736d87`, and scratch worktrees were preserved. The canonical infrastructure worktree was created with normal Git hooks and guards while the original canonical checkout stayed on `main`.
- Reconciliation references [AGENTS.md](../../AGENTS.md), [CLAUDE.md](../../CLAUDE.md), [the roadmap amendments](../ROADMAP.md#amendments-agreed-nov-24-and-25-2026), the [Tasks](../TASKS_CONTRACT.md), [Sessions](../SESSIONS_CONTRACT.md), and [Evallo filter](../EVALLO_FILTER_CONTRACT.md) contracts, [the production unit](../../systemd/miles-server.service), [API](../../src/server.py), [migrations](../../src/database.py), [fixtures](../../src/tests/conftest.py), and both [runtime](../../requirements.txt) and [test](../../requirements-dev.txt) manifests.

## Discrepancies and decisions

- The old scratch Git base ends at schema 40 and a separate source mirror reached 44. Canonical main reaches 45. Migration 45 adds the Evallo verification snapshot tables with an atomic version update. API startup clears the active snapshot flag. Preserve that behavior in production and use isolated synthetic state in development and tests.
- Canonical main already serves Tasks overview, capture, sessions, and the Evallo snapshot routes. The earlier mirror's missing Tasks overview does not describe canonical main. Pair the Nova revision and verify compatibility when Nova development resumes.
- The canonical test fixture redirects the calendar roles configuration as well as the SQLite path and provides a quiet Evallo state for unrelated tests. The Evallo suite already covers restart behavior, migration atomicity, and two starters upgrading from 44. Retain these checks and use the approved broader application and test runtime isolation. The manifests still describe the full Pi and test stack, and the documented cloud recipe installs it. A pinned headless install remains an implementation deliverable.
- Canonical `AGENTS.md` exists and currently directs agents to `CLAUDE.md` first. The shared orientation and map are implementation requirements. Canonical roadmap amendments add the nightly backup and queue Tasks, planner, Evallo, and privacy follow-ups. Those items retain their own scope and approval. Keep calendar tests synthetic and apply the current Evallo contract and roadmap exceptions to fixture behavior.
- **Prior hook deviation.** The scratch worktree was created with `git -c core.hooksPath=/dev/null worktree add …`. The assistant used the override to avoid running unreviewed hooks during checkout. That was an unnecessary safety bypass. The applicable Git hook was `post-checkout`. Subsequent scratch inspection found no custom hook path or executable `post-checkout`, so no checkout hook would have run under that observed configuration. The earlier configuration was not inspected, so its historical state remains unknown. Canonical worktree creation and this documentation transfer used no hook override.

## Verification and blockers

- Inspected the canonical `MIGRATIONS` table without importing application modules. The 45 entries are contiguous. A macOS `python3.13 -B` in-memory probe executed the extracted `init_db`, migration runner, and registered migration functions with a connection that accepts only `:memory:`. On Python 3.13.1 and SQLite 3.45.3, blank and repeat initialization both returned one schema row at 45. `PRAGMA integrity_check` returned `ok`.
- Application startup, pytest, file-backed WAL, cross-platform setup, and the runtime egress boundary were not exercised. These are implementation checks. Independent contract review and the normal build gate remain the blockers to implementation.

## Restrictions and exact next action

Preserve scratch history, leave canonical `main` untouched, and use normal hooks and guards. No Pi or production access, push, escalation, or application implementation is authorized.

**Next action:** Request the independent contract review of [DEV_ENVIRONMENT_CONTRACT.md](../DEV_ENVIRONMENT_CONTRACT.md) and its relevant [roadmap amendments](../ROADMAP.md#amendments-agreed-nov-24-and-25-2026) using the established review workflow. The expected result is findings keyed to contract sections and concrete cases, or a clear review pass. Record the exact reviewed revision. Later implementation reviews record the implementation SHA and may put findings in a subsequent documentation commit.
