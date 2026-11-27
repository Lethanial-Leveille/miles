# Claude Code coordination

Follow the shared instructions in [AGENTS.md](AGENTS.md). Start from the task, its handoff and the relevant entry in the [repository map](docs/REPOSITORY_MAP.md).

- For Small work, implement directly and self review. For Medium work, share a concise plan and use proportionate review.
- For a Large roadmap phase, show the owner the plan and contract before building. Follow the owner's approval and independent review gates. Use one Sonnet builder at a time in its own Git worktree. Route the contract and implementation reviews through the configured Codex workflow with only the files the owner permits. Use a rescue only after two Sonnet attempts and overflow only under the owner's existing contract, test and usage conditions.
- Choose ordinary reasoning for routine coordination and use existing search results before delegating exploration. Use Opus for difficult design or material review work only with the owner's approval.
- At a phase boundary, record outcomes and measurements for the owner, address material review findings and prepare the deployment plan with backup and rollback steps for separate approval. Use the [handoff workflow](docs/SESSION_START.md) when pausing or switching sessions.
