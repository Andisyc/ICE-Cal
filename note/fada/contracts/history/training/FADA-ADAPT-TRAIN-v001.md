# FADA-ADAPT-TRAIN-v001

Status: superseded offline implementation contract; real optimizer training requires a later explicit
long-run authorization.

Superseded on 2026-08-21 by `FADA-ADAPT-TRAIN-v002`.

## Inputs and identities

- source checkpoint: `fada` schema v1/v2, architecture-owned, exact SHA-256;
- target artifact: `fada-target-batch/v1`, matching architecture and source-checkpoint SHA-256;
- LoRA config: rank `8`, alpha `16`, dropout `0.05`, exact module manifest;
- output: self-contained `fada-adapted/v1` checkpoint with source/target identities, frozen base
  state, adapter state, optimizer state, completed updates, samples seen, and runtime config.

The target split is deterministic and episode-owned. At least two episode identities are required;
all rows from one episode stay in one split. Train/validation indices must be non-empty, disjoint,
and cover the artifact exactly.

## Update transaction

One update performs exactly:

1. select one target minibatch without changing row roles;
2. zero adapter optimizer gradients;
3. run the IDM on realized target future;
4. compute first-action MSE only;
5. backpropagate;
6. verify gradients exist only on adapter parameters;
7. optionally clip adapter gradients;
8. execute exactly one optimizer step.

Planner and base IDM parameters must remain byte/value-identical across a synthetic update. The
target artifact and source checkpoint are read-only inputs. Checkpoint persistence is temporary-file
then atomic replace.

## Reproduction defaults not specified by the paper

The paper does not report optimizer, learning rate, batch size, validation split, seed, or update
count. The repository exposes these through Hydra and records them in the adapted checkpoint. The
initial operational defaults are AdamW, learning rate `3e-4`, no weight decay, batch size `512`,
validation fraction `0.1`, seed `0`, and `400` maximum updates. These are explicit reproducibility
choices, not paper claims and not policy-quality evidence.

## Admission

The official CLI defaults to preflight-only. It may construct the frozen policy, validate identities,
split data, inject adapters, create the optimizer, and report counts, but it must not call backward or
optimizer step unless `adaptation.confirm_train=true` is supplied. This work unit stops before that
flag is used.
