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

from evals.scaffold import main as eval_main
from src.utils.distributed import init_distributed


def process_main(cfg: DictConfig, rank, world_size, devices):
    """Launch evaluation process for a single rank."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info("called-params with Hydra config")

    # Convert OmegaConf to dict for compatibility with existing code
    params = OmegaConf.to_container(cfg, resolve=True)

    # Apply commonly used flags from cfg for convenience
    params["val_only"] = cfg.get("val_only", params.get("val_only", False))
    params["use_fsdp"] = cfg.get("use_fsdp", params.get("use_fsdp", False))

    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the eval with loaded config
    eval_main(params["eval_name"], args_eval=params)


@hydra_main(config_path="../conf", config_name="eval_config", version_base=None)
def hydra_entry(cfg: DictConfig):
    """Hydra entrypoint for evaluation."""

    devices = cfg.get(
        "devices",
        ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
    )
    debugmode = cfg.get("debugmode", False)

    if debugmode:
        # FSDP debugging (use torchrun to set ranks/world size)
        if cfg.get("use_fsdp", False):
            process_main(
                cfg=cfg,
                rank=int(os.environ.get("RANK", 0)),
                world_size=int(os.environ.get("WORLD_SIZE", 1)),
                devices=devices,
            )
        else:
            process_main(cfg=cfg, rank=0, world_size=1, devices=[devices[0]])
    else:
        num_gpus = len(devices)
        mp.set_start_method("spawn", force=True)
        for rank in range(num_gpus):
            mp.Process(
                target=process_main,
                args=(cfg, rank, num_gpus, devices),
            ).start()


if __name__ == "__main__":
    hydra_entry()
