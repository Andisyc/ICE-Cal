# FADA v012 Unit A formal runtime audit rerun

Status: `READY — GLOBAL OFFLINE TEST`

The test uses the official `scripts/train_offpolicy.py:build_runner` composition root with the
production Hydra profile and a real local G1WalkFlat/MuJoCo environment initialization. It does not
call `learn()` and cannot start long training.

## Preserved production owners

- Hydra task/algo composition and privileged SAC preflight
- real G1WalkFlat/MuJoCo environment materialization
- production `ReplayBuffer` add/sample format
- production privileged SAC Critic and Actor optimizer updates
- production `DoubleBufferOffPolicyRunner` checkpoint dispatch
- production `FADAOracleCheckpointGateway`, strict learner reader, and 20+1 finalizer

## Allowed simplifications

- CPU device, one environment, compile disabled, and batch size four
- deterministic artificial replay transitions instead of a collector subprocess
- one Critic update and one Actor update
- one full learner checkpoint round-trip at iteration 240
- exact 20+1 lifecycle with the same bound gateway and a tiny state payload to avoid duplicating
  hundreds of megabytes of identical network/optimizer tensors

No loss, target, replay layout, optimizer, checkpoint identity, role schedule, reader, finalizer, or
consumer is reimplemented.

## Design-point observations

1. Official composition constructs `FADAPrivilegedSACLearner` with dimensions 98/303/29, the
   materialized 31-body/29-joint order, four canonical config hashes, and a bound checkpoint saver.
2. Replay add/sample produces the real actor/critic batch fields and both optimizer calls return
   finite metrics; at least one Actor parameter changes.
3. A full model/optimizer payload saved through the runner at iteration 240 reloads through the
   strict FADA learner and restores a deliberately mutated Actor parameter.
4. The same bound production gateway admits exactly `{240,480,...,4800,5000}` in a temporary
   lineage and writes a manifest with 21 checkpoint hashes.
5. A schema-1 payload and a mismatched action scale reject before learner mutation; these are
   current module dependencies, rerun in the same evidence campaign.

## Witness and falsifier

Witness: one structured receipt naming resolved identity, replay/update metrics, parameter delta,
strict restoration, and exact lineage manifest.

Falsifier: wrong dimensions/order/hash, absent gateway, replay/update exception or no Actor delta,
checkpoint identity mismatch, failed restoration, or any missing/extra/mixed 20+1 record.

## Evidence boundary

PASS establishes R1 official-offline composition and R2 checkpoint persistence for the declared
owners. It does not prove collector subprocess synchronization, CUDA compilation/transfer, 5,000-
iteration stability, convergence, standing/walking quality, or Unit B readiness. Any remaining
indispensable CUDA/collector fact must become at most one `LIVE_REQUIRED` item.
