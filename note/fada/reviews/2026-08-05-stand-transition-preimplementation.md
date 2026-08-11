# FADA Standing Curriculum Migration Review - Preimplementation

Date: 2026-08-05
Mode: `migration_review`
Repository discipline: active (`AGENTS.md`, FADA v004 contracts)

Accepted boundary: add static standing and walking-to-standing source scenarios to the existing
persistent-async Planner-IDM DAgger route. Do not train Oracles or launch a formal campaign.

## Verdict

READY. No P0-P2 blocker was found in the donor-to-target mapping.

The existing generic transition collector proves the accepted command-switch semantics and exposes
one atomic state/info refresh boundary. FADA must reuse that owner but retain its own temporal causal
window schema. The persistent FADA worker remains the only owner of walking/standing environment and
Oracle lifetime; static standing uses `G1StandStill`, while walk-to-stand remains in `G1WalkFlat`.

## Isolation contract

- Owner flag: `training.fada.stand_transition_curriculum.enabled`.
- OFF: no standing checkpoint load, no forced commands, no scenario partition, and unchanged natural
  collector calls.
- ON: validate ratios and walking command, require standing checkpoint and standing task owner,
  allocate exact main-source quotas, route scenario environments/Oracles, and persist scenario
  metadata together.
- Forbidden: walking-Oracle fallback for standing labels, command-crossing future chunks, transition
  rows whose history has no active command, and intermediate walking Oracles acting as standing labels.

## Required proof

OFF argument characterization, ON quota/connectivity tests, dual-Oracle provenance, missing-standing
checkpoint rejection, exact total-window accounting, persistent artifact metadata, focused static
checks, and postimplementation migration review.
