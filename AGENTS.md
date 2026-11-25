# AGENTS.md

Instructions for coding agents (Codex, Claude Code and their subagents) working in this repo.
The repo's own rules, conventions and production warnings are in CLAUDE.md; read it first. The
phase plan and its amendments are in docs/ROADMAP.md.

## Codex and Claude roles

Set by Lethanial on Nov 25 2026.

- **Plan and contract.** Opus writes the plan and the contract at the start of every phase. This is
  standard and does not need his approval each time. Any other use of Opus still needs his approval
  first.
- **Contract review before building.** Before any builder starts, Codex on GPT 6.2 Astra at High
  effort reviews the contract for gaps, missing error cases, and conflicts with the roadmap
  amendments. Its findings are shown to him together with the plan.
- **Orchestration.** Claude Code on Sonnet orchestrates and runs builder subagents one at a time, each
  in its own worktree.
- **End of phase review.** Codex replaces the Opus review at the end of every phase, using the Codex
  adversarial review on Astra at High effort.
- **Stubborn bugs.** If a bug fails twice, it is delegated with the Codex rescue command on Astra at
  High.
- **Usage limits.** If Claude hits a usage limit, Codex on GPT 6.2 Sol at Medium may finish the current
  builder task only if a written contract and tests already exist in that worktree. Otherwise work
  stops at a clean point.
- **Codex limits.** Codex never deploys, never touches production, and runs in the workspace write
  sandbox only.
- **Review gate.** The automatic review gate stays off.
- **Main session model.** If the model menu offers an Opus Plan option (Opus in plan mode, Sonnet for
  execution), use it for the main session.
