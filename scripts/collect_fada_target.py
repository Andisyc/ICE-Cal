"""CLI for one atomic paired FADA Stage-C target bundle."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.distill.fada.target_workflow import run_fada_target_collection


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="fada_target")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_fada_target_collection(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
