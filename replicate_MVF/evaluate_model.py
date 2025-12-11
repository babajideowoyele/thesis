from tadaconv.utils.config import Config
from runs.evaluate import evaluate


if __name__ == "__main__":
    cfg = Config(load=True)
    world_size = cfg.NUM_GPUS if cfg.NUM_GPUS is not None else 1
    print(f"World size for evaluation: {world_size}")
    if world_size <= 1:
        evaluate(0, cfg, world_size=world_size)
    else:
        from torch.multiprocessing.spawn import spawn
        spawn(evaluate, args=(cfg, world_size), nprocs=world_size)