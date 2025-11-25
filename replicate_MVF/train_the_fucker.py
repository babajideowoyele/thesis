
from tadaconv.utils.config import Config
from runs.train import train
if __name__ == "__main__":
    cfg = Config(load=True)
    world_size = cfg.NUM_GPUS
    if world_size <= 1:
        train(cfg, world_size=world_size)
    else:
        from torch.multiprocessing.spawn import spawn
        spawn(train, args=(cfg, world_size), nprocs=world_size)