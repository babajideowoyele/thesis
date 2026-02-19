import os
import warnings

warnings.filterwarnings("ignore", message="The video decoding and encoding capabilities of torchvision")

# ── Strip partial SLURM env vars ────────────────────────────────────────────
# sbatch sets SLURM_JOB_ID etc., but NOT SLURM_LOCALID (only srun does).
# Ignite sees the partial SLURM env and assumes full SLURM distributed mode,
# then crashes on the missing variables.  Since we use torchrun as the actual
# launcher (or run locally), we always remove SLURM vars before importing ignite.
for _k in [k for k in os.environ if k.startswith("SLURM_")]:
    del os.environ[_k]

import hydra
import torch
import torch.distributed as dist
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
import numpy as np

def training(local_rank, config: DictConfig):
    print(f"Running training on local rank {local_rank}")

    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    logger = setup_logger(name="MVFoul Training")

    if rank == 0:
        log_basic_info(logger, config)
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
    config.training.hyper_params.num_iters_per_epoch = len(train_loader)
    criterion = get_criterion(config, train_loader.dataset)
    criterion_severity, criterion_action = criterion
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
                tv = train_state.metrics.get(m_name)
                vv = val_state.metrics.get(m_name)
                
                # Handle confusion matrices separately - log as wandb heatmap
                if "ConfusionMatrix" in m_name:
                    for split, cm_val in [("train", tv), ("val", vv)]:
                        if cm_val is None:
                            continue
                        cm = cm_val.cpu().numpy() if isinstance(cm_val, torch.Tensor) else cm_val
                        # Confusion matrix table
                        log_dict[f"{split}/{m_name}"] = wandb.Table(
                            columns=[f"Pred_{i}" for i in range(cm.shape[1])],
                            data=cm.tolist()
                        )
                        # Actual class distribution (row sums of CM)
                        class_counts = cm.sum(axis=1)
                        dist_table = wandb.Table(
                            columns=["class", "count"],
                            data=[[i, int(c)] for i, c in enumerate(class_counts)],
                        )
                        log_dict[f"{split}/{m_name.replace('ConfusionMatrix', 'Distribution')}"] = (
                            wandb.plot.bar(dist_table, "class", "count",
                                           title=f"{split} {m_name.replace('ConfusionMatrix', '')} class distribution")
                        )
                else:
                    log_dict[f"train/{m_name}"] = tv.item() if isinstance(tv, torch.Tensor) else float(tv)
                    log_dict[f"val/{m_name}"] = vv.item() if isinstance(vv, torch.Tensor) else float(vv)
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
        trainer.run(
            train_loader, 
            max_epochs=config.training.hyper_params.num_epochs,
        )
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
    backend = cfg.get("backend", "nccl")

    if "RANK" in os.environ:
        # Launched by torchrun: init process group and let ignite discover it.
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend)
        idist.sync()
        try:
            training(local_rank, cfg)
        finally:
            dist.destroy_process_group()
    else:
        # Local / single-GPU: let idist.Parallel handle everything.
        with idist.Parallel(backend=backend) as parallel:
            parallel.run(training, cfg)

    exit(0)


if __name__ == "__main__":
    main()
