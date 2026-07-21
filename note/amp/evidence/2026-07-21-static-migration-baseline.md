# AMP Async Migration Static Baseline

Date: 2026-07-21

Scope: read-only Architecture/code/config inspection. No training, rollout,
checkpoint load, throughput benchmark, or policy-quality evaluation was run.

## Evidence

- `E1`: AMP_mjlab Architecture Atlas:
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/note/architecture/architecture/amp_mjlab_repository.data.json`
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/note/architecture/runtime/amp_training_runtime.data.json`
- `E2`: AMP implementation owners:
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/rsl_rl/runners/amp_on_policy_runner.py`
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/rsl_rl/algorithms/amp_ppo.py`
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/rsl_rl/modules/discriminator.py`
  - `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/rsl_rl/utils/motion_loader.py`
- `E3`: UniLab async owner code:
  - `src/unilab/ipc/async_runner.py`
  - `src/unilab/ipc/rollout_ring_buffer.py`
  - `src/unilab/algos/torch/appo/{runner,worker,learner,staging}.py`
  - `src/unilab/algos/torch/appo/runtime.py`
- `E4`: UniLab G1/backend owner code:
  - `src/unilab/envs/locomotion/g1/joystick.py`
  - `src/unilab/base/backend/base.py`
  - `src/unilab/base/backend/{mujoco,motrix}/backend.py`
- `E5`: current owner configs:
  - `conf/appo/config.yaml`
  - `conf/appo/task/g1_walk_flat/mujoco.yaml`
- `E6`: current Architecture and contract state:
  - `note/architecture/concept/03_g1_multiteacher_distillation_method.data.json`
  - `note/architecture/runtime/01_unilab_runtime_atlas.data.json`
  - `note/distillation/contracts/active/training/DISTILL-TRAIN-v003.md`
- `E7`: current-session runtime probes:
  - both G1 MuJoCo models compile to 31 bodies, 30 joints, 29 actuators,
    `nq=36`, and `nv=35`;
  - body-name order and joint-name order are identical;
  - one walk NPZ contains 29-DoF joint arrays, 30-body state arrays, and 50 Hz
    metadata.

## Code-Confirmed Facts

1. AMP_mjlab's discriminator consumes a 13-body state. Each body contributes
   relative position 3, orientation 6, local linear velocity 3, and local
   angular velocity 3. One state is 195 floats and one transition is 390 floats.
2. AMP_mjlab's expert loader has no command, gait phase, or contact labels. Its
   discriminator is unconditional and the discriminator reward enters PPO
   online.
3. UniLab's APPO route already separates a spawned collector from a learner and
   owns lifecycle, error propagation, shared actor/critic weights, rollout IPC,
   bounded device staging, V-trace, checkpointing, and timing metrics.
4. The current rollout ring-buffer schema is fixed to actor obs, critic obs,
   actions, behavior log-probabilities, rewards, done/truncated flags, and final
   actor/critic observations. It has no AMP payload extension.
5. The APPO collector owns environment stepping and exact terminal-observation
   resolution. The learner owns V-trace reward consumption and optimizer
   updates. Therefore the lowest-copy AMP route is to transport AMP transitions
   and score them on the learner before V-trace.
6. UniLab backends already expose body pose and velocity through the
   `SimBackend` contract. MuJoCo can inject and cache tracked-body sensors during
   initialization, so AMP features do not require hot-path XML access.
7. The current G1 APPO task still enables `feet_phase`. A new AMP-only task must
   explicitly isolate gait-phase observations/rewards instead of mutating the
   current task silently.
8. The existing distillation persistent runtime has a verified
   `NO_STABLE_SPEEDUP` result. That finding applies to the DAgger workflow, not
   automatically to APPO, but it forbids claiming that persistence/asynchrony
   alone guarantees a shorter end-to-end training time.

## Static Memory Estimate

For `N=2048`, `T=24`, four IPC slots, and float32 current/next AMP states:

```text
2 * 195 * 2048 * 24 * 4 bytes * 4 slots ~= 307 MB shared memory
```

A three-rollout learner staging pool adds roughly 230 MB before discriminator
replay, networks, environment state, and ordinary APPO buffers. At 4096 envs,
the AMP payload cost approximately doubles. These are planning estimates, not
measured peak RSS/VRAM.

## Unverified

- current APPO G1 wall-clock throughput on the target server/GPU;
- whether collector or learner is the current bottleneck;
- float32 AMP IPC overhead and queue occupancy;
- whether float16 AMP transport preserves reward/discriminator behavior;
- number of environment steps required for a useful walk-only AMP policy;
- the claimed 10-20 minute policy-quality target.

## Next Evidence

Run an unchanged APPO G1 walk baseline before AMP implementation and persist:
environment steps/s, collector env-step/inference time, learner update time,
wait fraction, H2D time, staging occupancy, GPU memory, and wall time.

