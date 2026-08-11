# FADA Standing And Walk-To-Stand Curriculum Plan

Status: implementation complete on 2026-08-05; formal training awaits a standing Oracle checkpoint

Terminal outcome: the persistent-async UniLab FADA route can distill static standing and
walk-to-stand recovery together with the existing walking source, while the old route remains
unchanged when the curriculum flag is off.

## Design Point Register

| Design ID | Human name | Contract | Concept block | Gap closed |
|---|---|---|---|---|
| `FADA-DP-CMD-01` | Command Coverage | `FADA-METHOD-v004#fada-dp-cmd-01--cmd-coverage--command-coverage` | `CMD-COVERAGE` | add zero-command and active-to-zero state coverage |
| `FADA-DP-IDM-02` | Causal IDM Supervision | `FADA-METHOD-v004#fada-dp-idm-02--causal-idm-supervision--causal-idm-supervision` | `CAUSAL-IDM-SUPERVISION` | select scenario-authoritative Oracle causal pairs |
| `FADA-DP-PLAN-02` | Actionable Planner Supervision | `FADA-METHOD-v004#fada-dp-plan-02--actionable-planner-supervision--actionable-planner-supervision` | `ACTIONABLE-PLANNER-SUPERVISION` | label standing/post-switch states with standing Oracle |

## Step 1 / 1

Objective: implement and verify the complete default-off curriculum locally.

Scope: contract/Inspector update, config schema, exact scenario allocation, shared atomic command
input owner, FADA causal-window admission, dual-environment/dual-Oracle persistent worker routing,
artifact/checkpoint metadata, focused OFF/ON/negative tests, bounded fake-env connectivity, and
document evidence.

Non-scope: Oracle training, remote training launch, long MuJoCo campaign, target adaptation, and
standing-to-walking curriculum.

Owner files/modules: `conf/distill/config.yaml`, `collector.py`, `fada_collector.py`,
`fada_async_runtime.py`, `train_distill.py`, focused FADA tests, and FADA governance/Atlas files.

Expected evidence: disabled-path characterization, exact quota unit tests, static-standing zero
windows, active-history/zero-future transition windows, Oracle role separation, missing-checkpoint
failure, dedicated standing-environment routing, persistent-worker scenario metadata, focused
pytest, Ruff, Pyright, and Atlas check.

Stop condition: all deterministic and formal fake-route checks pass, or the first real simulator or
external checkpoint requirement is recorded without claiming policy quality.
