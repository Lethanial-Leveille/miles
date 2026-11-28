# Claude Code coordination

Follow the shared instructions in [AGENTS.md](AGENTS.md). Start from the task, its handoff and the relevant entry in the [repository map](docs/REPOSITORY_MAP.md).

- Follow the tiers and the loop in AGENTS.md. Run the loop with `scripts/loop`.
- For Large work, show the owner a short contract before building and wait for approval. Use one Sonnet builder at a time in its own Git worktree. Reviews run through `scripts/loop`, with the reviewer AGENTS.md picks by tier (Claude for Small and Medium, Codex for Large) and only the files AGENTS.md allows. Use a rescue only after two Sonnet attempts.
- When the owner requests collaboration, a solo assignment or a switch to Codex, follow the shared operating modes and continuation procedure.
- Choose ordinary reasoning for routine coordination and use existing search results before delegating exploration. Use Opus for difficult design or material review work only with the owner's approval.
- At a phase boundary, record outcomes and measurements for the owner, address material review findings and prepare the deployment plan with backup and rollback steps for separate approval. Use the [handoff workflow](docs/SESSION_START.md) when pausing or switching sessions.
