from tadaconv.utils.config import Config
from runs.train import train
if __name__ == "__main__":
    cfg = Config(load=True)
    train(cfg)