# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import pprint
from pathlib import Path

from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed


def process_main(rank: int, cfg: DictConfig, world_size: int, devices):
    """Launch training on a single rank."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Convert to plain dict for the existing training code
    params = OmegaConf.to_container(cfg, resolve=True)

    # Log and persist the resolved config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params.get("folder", "/tmp/vjepa_output")
        params_path = Path(folder) / "params-pretrain.yaml"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, params_path)

    # Init distributed (access to comm between GPUs on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


@hydra_main(config_path="../conf", config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig):
    """Hydra entrypoint for training."""

    devices = cfg.get(
        "devices",
        ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    )
    debugmode = cfg.get("debugmode", False)

    if debugmode:
        process_main(rank=0, cfg=cfg, world_size=1, devices=[devices[0]])
    else:
        num_gpus = len(devices)
        mp.set_start_method("spawn", force=True)
        for rank in range(num_gpus):
            mp.Process(target=process_main, args=(rank, cfg, num_gpus, devices)).start()


if __name__ == "__main__":
    hydra_entry()
