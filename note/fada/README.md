# FADA Workflow

Default recall order:

1. `contracts/README.md`
2. active method and training contracts
3. `../architecture/concept/04_fada_method_discussion.data.json`
4. `../architecture/concept/06_fada_design_detail_discussion.data.json`
5. current plan/checklist only when continuing active work

The active Context authority is `FADA-CONTEXT-METHOD-v005` with
`FADA-CONTEXT-TRAIN-v004`. It defines the fixed-left-knee-`0.7` multi-sliding-window Support-Query
route accepted in Architecture 09 and Design Inspector 10. The former differentiable-dynamics route
and the implemented single-anchor route remain history. The single-anchor checkpoint worsened all
seven healthy-trajectory distance metrics with both stored and online Support.

The current visual authorities are
`architecture/architecture/09_trajectory_conditioned_execution_alignment.data.json` for the overall
method and `architecture/concept/10_in_context_execution_calibration_design_inspector.data.json` for
Context training. The accepted design uses one independent fault Support to produce a fixed
`delta_z`; a complete no-Context fault Label Query supplies multiple causal first-action windows;
and a separate calibrated re-execution stage tests closed-loop trajectory quality.

The Phase-1 privileged-teacher runners and checkpoints are historical negative evidence and no
longer define the active Context route. Pair-window data, collection, masked first-action loss,
schema-v2 persistence, schema-v3 checkpoints, and evaluation compatibility are implemented. The
bounded no-optimizer MuJoCo preflight passed; formal multi-window training has not started.
