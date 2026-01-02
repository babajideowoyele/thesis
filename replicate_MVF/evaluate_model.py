import atexit
import os
from tadaconv.utils.config import Config
from runs.evaluate import evaluate


if __name__ == "__main__":
    cfg = Config(load=True)

    checkpoints = [
        os.path.join(cfg.OUTPUT_DIR, f) 
        for f in os.listdir(cfg.OUTPUT_DIR) 
        if f.endswith('.pyth')
    ]

    world_size = cfg.NUM_GPUS if cfg.NUM_GPUS is not None else 1
    print(f"World size for evaluation: {world_size}")
    for ckpt in checkpoints:
        print(f"Evaluating checkpoint: {ckpt}")
        # TODO: find original training config.
        cfg.TEST.CHECKPOINT_FILE_PATH = ckpt
        if world_size <= 1:
            evaluate(0, cfg, world_size=world_size)
        else:
            os.remove("sharefile") if os.path.exists("sharefile") else None
            import torch.multiprocessing as mp
            mp.spawn(evaluate, args=(cfg, world_size), nprocs=world_size)
            