# FADA v012 Engineering Checklist

## Current confirmed design

- [x] ICE-Cal owns privileged Oracle training.
- [x] Oracle is generic privileged SAC on one G1WalkFlat/MuJoCo task.
- [x] Oracle actor directly consumes the typed privilege bundle.
- [x] Gait/feet-phase Reward is forbidden; non-zero aliases fail closed.
- [x] One run produces 20 intermediate checkpoints and final iteration 5000.
- [x] Planner input is state66 + previous-action29 history, command3 separate.
- [x] Planner future is action-free K×66.
- [x] Intermediate checkpoints are IDM-only coverage; final Oracle owns labels.
- [x] IDM-before-Planner, frozen-IDM gradient path, first-action supervision, and receding horizon are preserved.

## Unit A implementation gate

- [ ] Explicit production-code authorization received.
- [ ] RED contract tests retained.
- [ ] Typed privilege owner implemented and shape-probed.
- [ ] No gait-reward preflight implemented with negative test.
- [ ] DR families implemented at env/backend owner layers.
- [ ] 20+1 lineage manifest implemented and mixed-lineage rejection tested.
- [ ] Focused tests pass.
- [ ] Formal runtime audit admitted.
- [ ] Long Oracle training separately authorized and completed.
- [ ] Final Oracle policy-quality audit admitted.

## Unit B implementation gate

- [ ] Unit A lineage and quality admission exist.
- [ ] Explicit Unit B production-code authorization received.
- [ ] 98→66/29/3 split and 95-dimensional Planner history implemented.
- [ ] Action-free future and leakage regression test pass.
- [ ] Single-task collection roles and 1:2 sampling implemented.
- [ ] Optimizer ownership and gradient tests pass.
- [ ] New checkpoint schema rejects v011 artifacts.
- [ ] Focused and repository gates pass.
- [ ] Formal runtime audit admitted.
- [ ] Planner–IDM training separately authorized.
