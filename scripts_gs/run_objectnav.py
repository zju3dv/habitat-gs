#!/usr/bin/env python3
"""
Hydra entry-point for ObjectNav training / evaluation on GS scenes.

Usage:
    # Training (default)
    python scripts_gs/run_objectnav.py

    # Evaluation
    python scripts_gs/run_objectnav.py \
        --config-name=ddppo_objectnav_gs_eval \
        habitat_baselines.eval_ckpt_path_dir=/path/to/ckpt.pth
"""
import random
import sys

import numpy as np
import torch

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)

register_hydra_plugin(HabitatBaselinesConfigPlugin)
register_hydra_plugin(HabitatConfigPlugin)

import hydra
from omegaconf import DictConfig

from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp


@hydra.main(
    version_base=None,
    config_path="../data/scene_datasets/gs_scenes/configs",
    config_name="ddppo_objectnav_gs_train",
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)
    if (
        cfg.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
