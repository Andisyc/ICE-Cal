# ICE-Cal

ICE-Cal is a research repository for **In-Context Execution Calibration**: adapting a frozen
planner/execution stack from observable Support trajectories without online parameter updates.
The runtime is derived from UniLab, but this repository documents only the ICE-Cal research
question, its contracts, implementation evidence, and current design transition.

## Current status

`TRANSITION-BLOCKED`. The human-confirmed Design Inspector describes a per-forward,
per-Support-window correction `Δzᵢ`. The current Concept Figure, active FADA contracts, and
implemented training lineage still describe one condition-level fixed `Δz`. Until those semantic
artifacts are synchronized and approved, this repository does **not** authorize new engineering,
training, or evaluation under the per-window design.

## Read first

1. [Documentation registry](note/README.md)
2. [Concept Figure](note/architecture/08_in_context_execution_calibration.html)
3. [Design Inspector](note/architecture/09_in_context_execution_calibration_design_inspector.html)
4. [FADA research registry](note/fada/README.md)
5. [Machine-readable governance state](note/governance.json)
6. [Engineering documentation and runbooks](docs/README.md)

The two Architecture HTML pages are self-contained for direct local opening. For their optional
loopback server and provenance, see [the Architecture README](note/architecture/README.md).

## Engineering boundary

- Use `uv run` for Python commands.
- Preserve the UniLab-derived env, backend, runner, and configuration contracts described in
  [AGENTS.md](AGENTS.md).
- Treat `scripts/` as orchestration, not as the owner of long-lived research semantics.
- Do not infer policy quality from smoke tests, gradient checks, or lifecycle evidence.
- Training, live simulation, Git publication, and semantic contract activation require explicit
  authorization.
- Repeatable server operations belong to [engineering runbooks](docs/runbooks/), while research
  authority, dated plans, and evidence remain under [`note/`](note/).

For the upstream general-purpose RL framework and its documentation, use
[UniLab](https://github.com/unilabsim/UniLab). They are intentionally not mirrored here.
