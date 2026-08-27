# Contributing to ICE-Cal

This repository serves the ICE-Cal research line. Changes should preserve a clear chain from the
human-confirmed design to the active Contract, implementation owner, and executable evidence.

## Before changing semantics

1. Start at [note/README.md](note/README.md) and inspect [note/governance.json](note/governance.json).
2. Resolve any mismatch among the Concept Figure, Design Inspector, active Contract, and code.
3. Obtain explicit approval before activating a new Contract or starting training/live evaluation.
4. Add validation at the closest owner boundary; do not hide research rules in `scripts/`.

## Engineering rules

- Always use `uv run`; do not invoke Python directly.
- Preserve env/backend/config/async invariants in [AGENTS.md](AGENTS.md).
- Read the [engineering documentation](docs/README.md) and the applicable runbook before operating
  a server-side training or playback workflow.
- Keep backend-specific behavior behind the `SimBackend` contract.
- Keep asset and XML metadata off hot paths.
- Preserve unrelated dirty-worktree changes.
- Use English for code comments and public API docstrings.

## Validation

Run focused tests nearest the changed owner first. Before a PR, follow the PR gate in
[AGENTS.md](AGENTS.md), including `make test-all`. Documentation-only changes must at minimum pass
the Architecture structural checks when those pages are affected and `git diff --check`.

Server-side foreground training must follow the resource envelope in
[docs/runbooks/server-training.md](docs/runbooks/server-training.md); task-specific Hydra profiles
remain the authority for algorithm and method semantics.

Use Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, and
`chore:`. Do not stage, commit, push, or open a PR without the corresponding user authorization.
