"""CLI for same-condition FADA slope evaluation."""
import json

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.distill.fada.target_evaluation import run_fada_target_evaluation


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="fada_slope_evaluate")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_fada_target_evaluation(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
