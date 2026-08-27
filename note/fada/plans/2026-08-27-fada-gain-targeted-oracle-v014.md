# FADA v014 Gain-Targeted Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed full-domain-randomization Oracle source distribution with one
learnable nominal-plus-left-knee-gain distribution while preserving the v013 Reward and
Planner–IDM contracts.

**Architecture:** Hydra remains the single owner of the Oracle distribution. The existing
`G1ActuatorStrengthConfig` reset path samples left-knee actuator effectiveness, and the existing
FADA privileged observation exposes the resulting Kp/Kd scale only to the Critic. The Oracle SAC
preflight fail-closes any unrelated randomization or drift from the sealed gain profile.

**Tech Stack:** Hydra/OmegaConf, NumPy/MuJoCo reset randomization, PyTorch SAC, pytest.

---

## Engineering boundary

- Requested behavior: train one Oracle on nominal rows plus left-knee gain attenuation rows.
- Preserved behavior: single `G1WalkFlat/MuJoCo` task, 98-D deployable Actor observation,
  command-conditioned no-gait Reward, 20+1 checkpoint lineage, and all Planner–IDM tensor,
  supervision, freeze, and receding-horizon semantics.
- Distribution: actuator index `3`; `sampling_mode=single_candidate`; gain multiplier
  `g ~ U(0.8, 1.0)` on non-nominal rows; nominal probability `0.3`.
- Unrelated DR disabled: friction, gravity, COM, mass, body mass, armature, independent Kp/Kd,
  joint-position bias, torque RFI, control delay, and pushes.
- Existing observation noise is preserved: joint angle `0.01`, joint velocity `0.1`, all other
  configured noise scales zero.
- The Actor receives no explicit gain value. The existing typed privileged Kp/Kd scale tail lets
  only the Critic observe the applied effectiveness.
- Non-scope: simulator execution, training, checkpoint creation, policy-quality claims, changing
  the later calibration sweep `g in [0.8, 1.2]`, or generalizing beyond left-knee attenuation.

### Task 1: Activate v014 semantics

**Files:**
- Create: `note/fada/contracts/active/method/FADA-METHOD-v014.md`
- Create: `note/fada/contracts/active/training/FADA-TRAIN-v014.md`
- Move to history: v013 method/training Contracts
- Modify: `note/fada/contracts/README.md`, `note/fada/README.md`
- Modify: Design Inspector data and regenerate its standalone HTML

- [ ] State the exact gain distribution, privileged/deployable information boundary, preserved
  Reward semantics, and evidence limitations.
- [ ] Mark v013 source Contracts historical and make v014 the only active base source pair.
- [ ] Update the Frozen Backbone Inspector card without changing calibration-basis DR semantics.

### Task 2: Freeze RED admission cases

**Files:**
- Modify: `tests/algos/test_fada_privileged_oracle_v012.py`
- Modify: `note/fada/testing/v014_module_test_cards.md`

- [ ] Assert that the resolved Oracle profile contains only the approved actuator-strength
  distribution and preserves the existing observation noise.
- [ ] Parameterize one-field invalid profiles covering unrelated DR, wrong actuator identity,
  wrong range/probability, and direct strength leakage into the extra Critic tail.
- [ ] Run the focused tests and record failure against the still-v013 production profile.

### Task 3: Implement the minimum owner-level change

**Files:**
- Modify: `conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_privileged_oracle.yaml`
- Modify: `src/unilab/algos/torch/distill/fada_privileged_oracle_sac.py`

- [ ] Replace full DR with the existing `actuator_strength` profile.
- [ ] Add one resolved-config validator at Oracle preflight; do not modify scripts, backend APIs,
  environment reset logic, network schemas, or checkpoint format.
- [ ] Keep checkpoint identity fail-closed through the existing resolved config hashes.

### Task 4: Prove GREEN and close the local unit

**Files:**
- Create: v014 module evidence, execution receipt, governance receipt, plan/final review receipts.

- [ ] Run the narrow Oracle profile and preflight tests.
- [ ] Run actuator-strength owner tests, Oracle formal-route offline tests, config checks, Ruff,
  and Inspector standalone checks.
- [ ] Review responsibility, dependency direction, tensor provenance, state/checkpoint lifecycle,
  and evidence limits.
- [ ] Stop before simulator or training. Formal runtime audit and policy-quality evaluation remain
  separate gates.

