from tadaconv.utils.config import Config
from runs.train import train
if __name__ == "__main__":
    cfg = Config(load=True)
    assert cfg.NUM_GPUS > 1, "This script requires multiple GPUs."
    train(cfg)