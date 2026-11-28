# Engineering sessions

Start from [AGENTS.md](../AGENTS.md), the authorized revision and the relevant area of the [repository map](REPO_MAP.md). The project workflow works in Mac CLI and supported cloud sessions.

| Request | What to do |
|---|---|
| “Start work on …” | Confirm the objective, authorized base, worktree and task class. Open the relevant files through the map. Create a short handoff in [docs/work](work/) for work that spans sessions or has a phase contract. |
| “Work with Claude and Codex” or “Continue with Claude only” or “Codex, take it from here” | Use the requested operating mode and keep the assignment scope and gates. In collaboration, identify who owns the next step. |
| “Continue …” or “Resume …” | Find the named branch and its handoff. Match the branch, HEAD and working tree to the recorded state. Resolve discrepancies before editing and continue from the documented next action within scope. |
| “Wrap up” or “Hand this off” | Stop at a coherent checkpoint. Run proportionate verification, record results and outstanding work, and stage only intended files for an authorized local commit using normal hooks. Push and deployment require their own approval. |
| “Get an independent review” | Follow the owner's review scope. For a contract, supply only that contract and the named roadmap section. For implementation, supply the exact reviewed SHA, base or diff range, diff, touched files and contract. Record findings against the reviewed revision. |

## Continue after an interruption

1. The outgoing agent, if available, records a coherent checkpoint in the existing handoff. Include the commit and dirty worktree status, completed checks, pending approvals or review gates, and the next action. Preserve partial work in place if it is unsafe to commit.
2. At the owner's direction, the available agent reads the handoff, contract and recorded test results. Confirm the repository, branch, base, HEAD and working tree. If the outgoing agent could not finish the handoff, reconstruct the checkpoint from Git and available test evidence and label any gaps.
3. Resume the next authorized step. Run targeted checks if changes or missing evidence warrant them. When an independent review is due, prepare the scoped packet for a fresh reviewer session. If the review is unavailable, record the gate and exact revision and stop there.

## Handoff record

Record the branch, base and latest implementation or reviewed commit, plus a paired client or backend SHA when relevant. Include the objective, scope and acceptance criteria, decisions and reasons, progress and affected paths, exact verification commands, platform, tested revision and outcomes, unresolved findings or permission limits, and one specific next action. Use relative links and sanitized evidence. A handoff commit can record the prior implementation SHA without needing its own SHA inside the file. [Development environment handoff](work/dev-environment.md)

## Keep current docs accurate

When you encounter relevant drift, report it and update the owning doc within scope. Check current settings against the source files named in the map. Keep measurements in the relevant topic doc with conditions and dates. Use the [incident log](INCIDENTS.md) for production failures, the [decision log](ENGINEERING_HISTORY.md#decision-log) for choices and rationale, and the [backend backlog](BACKEND_TODO.md) for deferred work. Check only the relevant claims and run the verification appropriate to the change.

## Authorized Pi sessions

Live observation requires separately scoped owner authorization. Use the relevant [infrastructure procedures](INFRASTRUCTURE.md) and approved read only evidence to check service state, checkout, model, cache or latency as needed. Routine development uses the [isolated launcher](DEVELOPMENT.md). For an authorized remote Claude operator session, `tmux new -s miles` and `tmux attach -t miles` preserve the session across SSH disconnects.

The [engineering history](ENGINEERING_HISTORY.md) preserves dated decisions, drift evidence and the original preflight for reference.
