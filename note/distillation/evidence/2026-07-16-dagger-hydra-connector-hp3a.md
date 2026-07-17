# DAgger Hydra Connector HP-3a Evidence

Date: 2026-07-16

Scope: Hydra and entrypoint routing only. This evidence does not claim that the
production persistent runtime factory, real G1 envs, or shared weights exist.

## Core Parameter Path

```text
training.workflow.execution_mode
-> run_single_entry_workflow parse and mixed-state guards
-> persistent runtime factory inputs
-> run_multirole_dagger_workflow collector arguments
-> result execution_mode
-> service close in finally
```

## Red Evidence

The first focused run failed at the intended boundaries:

- Hydra struct rejected the missing `execution_mode` field.
- legacy workflow calls did not forward `execution_mode`.
- `run_single_entry_workflow` did not accept the persistent factory.

## Green Evidence

Focused OFF/ON connector: `4 passed, 316 deselected in 0.44s`.

Affected distill config/script group: `70 passed, 250 deselected in 3.55s`.

Ruff: `All checks passed!`.

The ON semantic fake receives the original global config, both role owner
configs, role specs, and scenario specs. Workflow receives only
`scenario_collector`; the legacy callback is `None`. The service is closed once
after workflow return. OFF mode forwards only the old callback route and does
not construct a service.

## Stale Search

Search found the flag only in its Hydra owner, script/workflow consumers,
tests, architecture/governance notes, and evidence. There is no hidden second
default or alternate ON selector.

## Decision

HP-3a passes its config/entrypoint boundary. Production ON remains deliberately
fail-closed without a runtime factory. HP-3b must provide that owner before any
real persistent command is authorized.
