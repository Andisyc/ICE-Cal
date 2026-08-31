# FADA shared privileged environment repair

## Accepted behavior

`walk`, `static_stand`, and `walk_to_stand` collection use the same
`G1WalkFlat` environment and therefore the same 303-D privileged Oracle
observation contract. Scenario behavior differs only through the existing
`commands` mutation owned by the collection transaction.

## Preserved behavior

- Keep the v022 20+1 Oracle lineage, Planner-IDM architecture, losses,
  checkpoint schema, scenario ratios, cold-start windows, and progress logs.
- Keep one persistent environment lifecycle and close it once on normal or
  exceptional teardown.
- Do not run training, simulation, deployment, Git publication, or alter policy
  quality semantics.

## Steps and proof

1. Add a failing worker regression proving an enabled standing curriculum
   constructs one environment and aliases the standing scenario to that
   `G1WalkFlat` owner.
2. Remove the dedicated standing-owner materialization from the persistent
   runtime and route static collection to the resident walking environment.
3. Run the focused worker and FADA collection tests, then Ruff, formatting,
   compile, and diff checks for the touched boundary.

