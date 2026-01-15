
import hydra
from omegaconf import DictConfig, OmegaConf
from runs.train import train

@hydra.main(config_path="configs", config_name="att_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point using Hydra decorator."""
    world_size = cfg.NUM_GPUS
    if world_size <= 1:
        train(0, cfg, world_size=world_size)
    else:
        from torch.multiprocessing.spawn import spawn
        import socket, os
        s=socket.socket()
        s.bind(('',0))
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
        spawn(train, args=(cfg, world_size), nprocs=world_size)

if __name__ == "__main__":
    main()