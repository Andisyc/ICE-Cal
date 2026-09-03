import json

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.distill.fada.target_adaptation_workflow import (
    FADAAdaptationPreflight,
    preflight_fada_adaptation,
    run_fada_adaptation,
    train_fada_adaptation,
)


@hydra.main(version_base="1.3", config_path="../conf/offpolicy", config_name="fada_adapt")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_fada_adaptation(cfg), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
