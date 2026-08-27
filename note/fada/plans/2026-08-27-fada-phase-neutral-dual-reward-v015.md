# FADA v015 Phase-Neutral Dual-Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove gait-clock authority while preserving the 98-D/66-D Planner–IDM tensor contract,
and provide one nominal dual-Reward SAC profile that can be validated before privileged/Gain work.

**Architecture:** `G1WalkEnvCfg` owns one `gait_phase_enabled` switch. Disabled tasks retain two
constant-zero compatibility slots through reset, observation, and step. A new nominal Hydra profile
owns the dual-Reward task; the privileged Oracle profile inherits it and adds only its existing
runtime and Gain distribution.

**Tech Stack:** Hydra/OmegaConf, NumPy/MuJoCo environment owner, PyTorch SAC configuration, pytest.

**Spec:** `note/fada/contracts/active/method/FADA-METHOD-v015.md`

## Global Constraints

- Keep Actor input 98-D and Planner state 66-D; the final two state slots are constant zeros.
- Preserve the existing gait-enabled base task and its checkpoints.
- Do not add a new runner, network, adapter, checkpoint schema, or backend API.
- Do not run simulation, training, server operations, Git mutation, or policy-quality evaluation.

## Affected Module Set

| Module | Relation | Obligation | Proof |
|---|---|---|---|
| G1 walk config/phase lifecycle | direct state owner | add disabled lifecycle without changing enabled behavior | MTC-A tests |
| nominal SAC Hydra profile | direct config owner | compose phase-neutral dual Reward with no privilege/DR | MTC-B tests |
| privileged Oracle profile/preflight | config consumer | inherit nominal owner and reject an enabled clock | MTC-C tests |
| FADA input split | transitive tensor consumer | verify 98→66/29/3 unchanged | existing + focused regression |
| checkpoint/Planner/IDM owners | preserved | no code/schema/update change | existing focused regressions |

### Task 1: Freeze RED semantic cases

**Files:**
- Modify: `tests/envs/locomotion/g1/test_gait_constraint.py`
- Modify: `tests/algos/test_fada_privileged_oracle_v012.py`

- [ ] Add reset, observation, and step cases requiring two constant-zero slots when disabled.
- [ ] Add nominal and privileged composition cases, including rejection of an enabled clock.
- [ ] Run the narrow cases and record failure against v014 behavior.

### Task 2: Implement the G1 owner-level switch

**Files:**
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`

- [ ] Add `G1WalkEnvCfg.gait_phase_enabled: bool = true` for compatibility.
- [ ] Return zeros at disabled reset, expose zeros in observation, and preserve zeros during step.
- [ ] Keep all enabled behavior unchanged and avoid backend/private access.

### Task 3: Establish the config inheritance boundary

**Files:**
- Create: `conf/offpolicy/task/sac/g1_walk_flat/mujoco_no_gait_dual_reward.yaml`
- Modify: `conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_privileged_oracle.yaml`
- Modify: `src/unilab/algos/torch/distill/fada_privileged_oracle_sac.py`

- [ ] Move the shared dual-Reward/command/no-phase facts into the nominal profile.
- [ ] Make the privileged profile inherit it and retain only privileged/Gain-specific facts.
- [ ] Fail closed when privileged preflight sees `gait_phase_enabled=true`.

### Task 4: Prove GREEN and close locally

**Files:**
- Create: v015 module evidence, one-shot execution unit, governance and review receipts.

- [ ] Run MTC-A/B/C pseudo-samples and controlled counterexamples.
- [ ] Run affected G1, Oracle, input-contract, config, lint, and Inspector consistency checks.
- [ ] Complete R2 final-gate review and stop before formal runtime or training.
