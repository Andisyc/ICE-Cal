# FADA v013 Module Test Cards

Status: HISTORICAL under v014. The command-conditioned no-gait Reward cases remain regression
evidence, but v013 distribution and formal/policy-quality admissions cannot authorize v014.

## MTC-A — command-conditioned Reward admission

- **Input:** resolved `G1WalkFlat/MuJoCo` Oracle config.
- **Output:** one 98-D Actor route with zero-command stand Reward and nonzero-command walk Reward.
- **Invariant:** command is the only mode authority; static and recovery stand rows share the support
  term set; stand and walk terms do not leak across branches.
- **Negative cases:** disabled mode, missing support geometry, tracking in stand, standing terms in
  walk, `stand_action_l2`, or `stand_still`.
- **Evidence owner:** Hydra composition tests, preflight negative tests, and Reward dispatch tests.

## MTC-B — no-gait boundary

- **Input:** Reward scales and gait-constraint config before environment construction.
- **Output:** admitted phase-free Reward structure.
- **Invariant:** phase/footfall scales are zero or absent; gait constraint is disabled with zero
  penalty scale.
- **Negative cases:** nonzero alias, enabled constraint, or nonzero constraint scale.
- **Evidence owner:** Oracle preflight tests, then formal runtime audit.

## MTC-C — unchanged Oracle lineage and Planner–IDM boundary

- **Input:** typed privileged observation and one 20+1 Oracle lineage.
- **Output:** final-Oracle labels plus intermediate IDM coverage for the v012 tensor contract.
- **Invariant:** state66/previous-action29/command3 split, action-free future, serial optimizer
  ownership, and checkpoint lineage remain unchanged.
- **Evidence owner:** existing Oracle/Planner-IDM module evidence plus future v013 runtime and quality
  admission.
