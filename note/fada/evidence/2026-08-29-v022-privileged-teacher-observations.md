# v022 Privileged Teacher Observations

Status: `CURRENT RAW/DERIVED EVIDENCE — NOT A SEALED LINEAGE`

## Runtime observations

- Run label reported in the active session: `G1WalkFlat_live_priv_grouped_dr_v022`.
- Electerm displayed high Reward and high episode length near the end of the 5000-iteration run.
- Exact final values, remote log hash, and checkpoint hashes were not re-read during this
  documentation synchronization and remain volatile.

## Persistence failure

The following IDM launch failed before training because its required intermediate Oracle files were
absent: `model_240.pt, model_480.pt, …, model_4800.pt`.

The current effective teacher profile is a policy-quality validation profile:

- `privileged_dr_curriculum_validation=true`
- `checkpoint_mode=validation`
- `save_interval=1000`
- the runtime checkpoint saver returns no sealed FADA gateway in validation mode

Therefore the missing 240-step checkpoints are explained by the selected profile. This observation
does not invalidate the learned policy; it blocks its use as the IDM source lineage.

## Evidence limits

This file records the current conversation/runtime observation and its configuration-level causal
explanation. It is not a formal runtime receipt, checkpoint admission result, or quantitative
policy-quality audit.
