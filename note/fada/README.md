# FADA Workflow

Default recall order:

1. `contracts/README.md`
2. active method and training contracts
3. `../architecture/concept/04_fada_method_discussion.data.json`
4. `../architecture/concept/06_fada_design_detail_discussion.data.json`
5. current plan/checklist only when continuing active work

Current Context method authority: `FADA-CONTEXT-METHOD-v003`, with design-only training authority
`FADA-CONTEXT-TRAIN-v002`. Context remains `z_repaired = z + delta_z`; Tracker Encoder and Decoder
stay frozen, and only Context Encoder trains. A first faulty rollout is Context input; a paired
second trajectory is compared with the healthy reference through a learned differentiable fault
dynamics ensemble. Actions and optimized `delta_z` are not Context labels.

Current implementation status: design-only. Historical v006 restored full-horizon walking but failed
formal paired quality because lateral displacement and yaw drift worsened. It remains negative
evidence and is not a Context label source. Context implementation is blocked on an exact `E/D`
checkpoint, paired probe/reference lifecycle, fault-transition schema, differentiable-model gates,
trajectory loss, and model-to-MuJoCo validation protocol.

The Phase-1 privileged-teacher runners and checkpoints are historical negative evidence and no
longer define the active Context route. The completed Planner-IDM prerequisite continues to use
`persistent_async`; no Context runtime owner has yet been accepted.
