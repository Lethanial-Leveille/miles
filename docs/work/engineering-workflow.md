# Engineering workflow cleanup handoff

**Branch** `docs/engineering-workflow`

**Accepted base** `6e72b9e` on `infra/dev-environment`

**Objective** Reduce routine planning, review and context costs while retaining owner approval and safety boundaries. Acceptance requires a shared entry point, proportionate task gates, portable handoffs, preserved history, a documented Sessions discrepancy and a local documentation commit.

## Decisions and progress

[AGENTS.md](../../AGENTS.md) owns shared rules. [CLAUDE.md](../../CLAUDE.md) owns Claude coordination. [Engineering sessions](../SESSION_START.md) owns the handoff workflow and links to the [historical decision log](../ENGINEERING_HISTORY.md). The [backend backlog](../BACKEND_TODO.md) points to the approved roadmap for phase order and retains earlier research. The [repository map](../REPOSITORY_MAP.md#shared-api-contracts) records contract ownership and pairing. The [README](../../README.md#long-term-direction) records the product direction.

Treat this as Medium documentation work since it changes coordination across multiple entry points. The accepted infrastructure contract and roadmap remain intact.

## Verification

On macOS, `git diff --check` and `git diff --cached --check` passed on the documentation tree. A read only Python check verified the added and relocated relative links and anchors and confirmed preservation of all 72 decision headings and 74 historical backlog subsections. The comparison used the accepted backend base `6e72b9e` and the local Nova `origin/main` snapshot `d9e8c9f`. The Sessions contract difference was a single insertion describing migration 44 and nightly review v2. Use `git log -1 --format='%h %s' docs/work/engineering-workflow.md` to identify the documentation commit containing this handoff. Application tests were outside this documentation change.

## Restrictions and next action

The Nova copy remains pending separately authorized reconciliation against the selected backend revision. CI, cloud setup, Nova development configuration, backend refactoring, production work and new product features remain deferred. The accepted SQLite residual risk remains recorded in the [infrastructure handoff](dev-environment.md).

Next, inspect the local checkpoint with `git log --oneline 6e72b9e..docs/engineering-workflow` and `git diff --stat 6e72b9e..docs/engineering-workflow` before any integration decision.


## November 27 continuation

`docs/agent-operation-modes` starts at `6e6cf04`. The owner requested collaborative, Claude only and Codex only operation under the same shared rules. The owner intends to reserve at least half of the next Codex allowance for collaboration with Claude tonight. This is a session intention.

The nightly backup checkpoint is `feature/nightly-backup` at `f4e4fab`. Read its contract and handoff with `git show f4e4fab:docs/NIGHTLY_BACKUP_CONTRACT.md` and `git show f4e4fab:docs/work/nightly-backup.md`. The existing authorization covers synthetic local backup development subject to phase gates. The recorded November 27 hold until 20:00 Eastern governs when implementation may start. No explicit instruction lifting that hold was found. If the earlier authorization was meant to allow a local build before 20:00, the owner needs to confirm that exception. The full database push routing identifier decision and applicable contract gate remain pending. Any real backup, transfer, timer installation or deployment requires its own authorization, and the rollout decisions on FileVault and the privileged installer remain open.

Verification for this update uses a staged whitespace check, relative link checks and a comparison of the classification, production and secrets rules against the base. No application tests are needed for these documentation changes.
