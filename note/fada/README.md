# FADA Workflow

Default recall order:

1. `contracts/README.md`
2. active method and training contracts
3. `../architecture/concept/04_fada_method_discussion.data.json`
4. `../architecture/concept/06_fada_design_detail_discussion.data.json`
5. current plan/checklist only when continuing active work

The active Context authority is `FADA-CONTEXT-METHOD-v004` with
`FADA-CONTEXT-TRAIN-v003`. It implements the fixed-left-knee-`0.7` Support-Query action-supervision
route accepted in Architecture 09 and Design Inspector 10. The former differentiable-dynamics route
(`FADA-CONTEXT-METHOD-v003` / `FADA-CONTEXT-TRAIN-v002`) remains history after 10/10 real-MuJoCo
gates rejected its Context candidates.

The current visual authorities are
`architecture/architecture/09_trajectory_conditioned_execution_alignment.data.json` for the overall
method and `architecture/concept/10_in_context_execution_calibration_design_inspector.data.json` for
Context training. The implemented route uses one independent fault Support to produce a fixed
`delta_z`, a no-Context fault Label Query for first-action supervision, and a separate post-training
calibrated re-execution stage.

The Phase-1 privileged-teacher runners and checkpoints are historical negative evidence and no
longer define the active Context route. The Context model, dataset, collector, preflight, trainer,
and checkpoint owners are implemented. The bounded fixed-`0.7` MuJoCo preflight passed; formal
Context training and post-training closed-loop evaluation have not started.
