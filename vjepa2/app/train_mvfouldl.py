import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from app.mvfoul.dataflow import get_dataflow
from app.mvfoul.model import get_model
from app.mvfoul.train import get_lr_scheduler, get_optimizer
import ignite.distributed as idist
from ignite.utils import manual_seed, setup_logger
from ignite.engine import Events
from ignite.contrib.engines import common
from ignite.handlers import Checkpoint, global_step_from_engine
from app.mvfoul.train import log_basic_info, create_trainer, create_evaluator, log_metrics, get_save_handler, setup_rank_zero, get_criterion
from app.mvfoul.metrics import build_metrics
import wandb

def training(local_rank, config: DictConfig):
    print(f"Running training on local rank {local_rank}")

    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    logger = setup_logger(name="MVFoul Training")
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    use_wandb: bool = config.wandb.enable
    if use_wandb and rank == 0:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.get("entity", None),
            name=config.wandb.get("run_name", None),
            config=OmegaConf.to_container(config, resolve=True),
        )

    train_loader, val_loader = get_dataflow(config)
    model = get_model(config)
    device = idist.device()
    model = model.to(device)
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config)
    criterion_severity, criterion_action = criterion
    config.training.hyper_params.num_iters_per_epoch = len(train_loader)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    trainer = create_trainer(
        model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger
    )

    # -- build metrics from config --------------------------------------------
    metrics, primary_metric = build_metrics(config, criterion_severity, criterion_action)

    train_evaluator = create_evaluator(model, metrics, config)
    val_evaluator = create_evaluator(model, metrics, config)

    # ── W&B training-step logging ────────────────────────────────────────────
    if use_wandb and rank == 0:
        log_freq = config.wandb.log_frequency

        @trainer.on(Events.ITERATION_COMPLETED(every=log_freq))
        def log_training_loss(engine):
            wandb.log(
                {
                    "train/batch_loss": engine.state.output["batch loss"],
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=engine.state.iteration,
            )

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = train_evaluator.run(train_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "train", state.metrics)
        state = val_evaluator.run(val_loader)
        log_metrics(logger, epoch, state.times["COMPLETED"], "val", state.metrics)

        # ── W&B epoch-level metrics ──────────────────────────────────────────
        if use_wandb and rank == 0:
            train_state = train_evaluator.state
            val_state = val_evaluator.state
            log_dict = {"epoch": epoch}
            for m_name in metrics:
                log_dict[f"train/{m_name}"] = train_state.metrics.get(m_name)
                log_dict[f"val/{m_name}"] = val_state.metrics.get(m_name)
            wandb.log(log_dict, step=trainer.state.iteration)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED,
        run_validation,
    )

    if rank == 0:
        evaluators = {"train": train_evaluator, "val": val_evaluator}
        if not use_wandb:
            tb_logger = common.setup_tb_logging(
                config.output_path, trainer, optimizer, evaluators=evaluators
            )

    best_model_handler = Checkpoint(
        {"model": model},
        get_save_handler(config),
        filename_prefix="best",
        n_saved=2,
        global_step_transform=global_step_from_engine(trainer),
        score_name=f"val_{primary_metric}",
        score_function=Checkpoint.get_default_score_fn(primary_metric),
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        best_model_handler,
    )

    try:
        trainer.run(train_loader, max_epochs=config.training.hyper_params.num_epochs)
    except Exception as e:
        logger.exception("")
        raise e

    if rank == 0:
        if use_wandb:
            wandb.finish()
        else:
            tb_logger.close()



@hydra.main(config_path="../conf", config_name="train_scratch", version_base=None)
def main(cfg):
    import os
    # Remove stale SLURM env vars so idist detects torchrun instead of SLURM
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            del os.environ[key]

    backend = cfg.get("backend", "nccl")  # Explicit backend prevents SLURM auto-detection
    print(f"Using distributed backend: {backend}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # If RANK env var exists, we're already spawned by torchrun—don't use idist.Parallel
    if "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        training(local_rank, cfg)
    else:
        # Use idist.Parallel to spawn processes
        with idist.Parallel(backend=backend) as parallel:
            parallel.run(training, cfg)


if __name__ == "__main__":
    main()
