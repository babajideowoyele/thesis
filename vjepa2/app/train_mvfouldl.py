import fire
import hydra
from omegaconf import DictConfig, OmegaConf
from app.mvfoul.dataflow import get_dataflow
from app.mvfoul.model import get_model
from app.mvfoul.train import get_lr_scheduler, get_optimizer
import ignite.distributed as idist
from ignite.utils import manual_seed, setup_logger
from ignite.engine import Events
from ignite.contrib.engines import common
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from app.mvfoul.train import log_basic_info, create_trainer, create_evaluator, log_metrics, get_save_handler, setup_rank_zero, get_criterion
import wandb

def training(local_rank, config: DictConfig):

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
    optimizer = get_optimizer(config, model)
    criterion = get_criterion(config)
    config.num_iters_per_epoch = len(train_loader)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    trainer = create_trainer(
        model, optimizer, criterion, lr_scheduler, train_loader.sampler, config, logger
    )

    metrics = {
        "Accuracy": Accuracy(),
        "Loss": Loss(criterion),
    }

    train_evaluator = create_evaluator(model, metrics, config)
    val_evaluator = create_evaluator(model, metrics, config)

    # ── W&B training-step logging ────────────────────────────────────────────
    if use_wandb and rank == 0:
        log_freq = config.wandb.log_frequenc

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
            wandb.log(
                {
                    "epoch": epoch,
                    "train/accuracy": train_state.metrics["Accuracy"],
                    "train/loss": train_state.metrics["Loss"],
                    "val/accuracy": val_state.metrics["Accuracy"],
                    "val/loss": val_state.metrics["Loss"],
                    "train/bal_accuracy": train_state.metrics.get("BalancedAccuracy", None),
                    "val/bal_accuracy": val_state.metrics.get("BalancedAccuracy", None),
                },
                step=trainer.state.iteration,
            )

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
        score_name="val_accuracy",
        score_function=Checkpoint.get_default_score_fn("Accuracy"),
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
    def run(backend=None, **spawn_kwargs):
        cfg.backend = backend
        
        with idist.Parallel(backend=cfg.backend, **spawn_kwargs) as parallel:
            parallel.run(training, cfg)
    
    fire.Fire({"run": run})
        