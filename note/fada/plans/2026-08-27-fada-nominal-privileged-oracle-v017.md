# FADA v017 Nominal Privileged Oracle — Documentation Transition Plan

> Status: SUPERSEDED / HISTORICAL. v022 restores a live-privileged grouped-DR teacher with an
> iteration curriculum.

## Objective

Remove downstream calibration faults from Oracle ownership. Train the perfect privileged Oracle only
under nominal dynamics; inject left-knee Gain and every other failure only after the source Oracle
and Planner–Tracker have been trained and frozen.

## Semantic migration

| Object | v017 owner | Retired v016 interpretation | Engineering proof destination |
|---|---|---|---|
| Oracle training distribution | nominal `G1WalkFlat/MuJoCo` task | nominal plus left-knee Gain `[0.8,1.0]` | composed-config absence and runtime preflight |
| Privileged observation | Critic state interface only | carrier for randomized Gain scales | Actor/Critic provenance tests |
| 20+1 lineage | one nominal privileged run | one gain-randomized privileged run | strict checkpoint-lineage tests |
| Left-knee Gain | downstream failed-rollout collection | Oracle domain randomization | collection identity and isolation tests |

## Required future engineering unit

1. Disable actuator-strength and every physical/domain randomization in the privileged Oracle owner
   configuration.
2. Make preflight reject any Oracle-side execution fault, including left-knee Gain.
3. Preserve the typed Critic interface, 98-D Actor, single Reward, zero gait-phase placeholders, SAC
   owners, and 20+1 checkpoint schema.
4. Prove the nominal Oracle route before any server training; separately prove that failed-rollout
   collection can inject Gain without mutating the frozen source lineage.

## Invalidated evidence

All v016 module, formal, command, checkpoint, and policy-quality evidence is historical because its
effective configuration contains Oracle-side Gain randomization. The unsuccessful
`G1WalkFlat_v016/model_5000.pt` must not label Planner–IDM data or authorize v017.

## Stop conditions

- Any failure parameter is sampled during Oracle training.
- A v016 checkpoint or receipt is reused as v017 evidence.
- Gain is visible to or changes the source Oracle lineage.
- Code modification, simulation, training, server operation, or Git mutation begins without a new
  engineering authorization.
