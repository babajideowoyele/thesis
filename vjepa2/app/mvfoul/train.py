import datetime
from pathlib import Path
import ignite
import ignite.distributed as idist
from ignite.engine import Engine
from ignite.handlers import Checkpoint
import torch
from ignite.contrib.engines import common
from src.utils.loss import get_loss_fn
import torch.optim as optim
from ignite.handlers import PiecewiseLinear

def create_trainer(model, optimizer, criterion, lr_scheduler, sampler, cfg, logger):
    device = idist.device()
    criterion_severity, criterion_action = criterion
    use_amp = cfg.training.with_amp

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")

    def _update(engine, batch):
        model.train()
        offence_sev_target, action_target, clips, _ = batch
        clips = clips.to(device, non_blocking=True).float()
        offence_sev_target = offence_sev_target.to(device, non_blocking=True)
        action_target = action_target.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                action_pred, offence_sev_pred = model(clips)
                loss = criterion_severity(offence_sev_pred, offence_sev_target) + criterion_action(action_pred, action_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            action_pred, offence_sev_pred = model(clips)
            loss = criterion_severity(offence_sev_pred, offence_sev_target) + criterion_action(action_pred, action_target)
            loss.backward()
            optimizer.step()

        return {"batch loss": loss.item()}

    trainer = Engine(_update)
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=sampler,
        to_save=to_save,
        save_every_iters=cfg.training.checkpoint_every,
        save_handler=get_save_handler(cfg),
        lr_scheduler=lr_scheduler,
        output_names=["batch loss"] if cfg.training.log_every_iters > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    if cfg.training.resume_from is not None:
        checkpoint = load_checkpoint(cfg.training.resume_from)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer

def setup_rank_zero(logger, config):
    device = idist.device()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = config.output_path if config.output_path is not None else "outputs"
    folder_name = (
        f"{config.model.name}_backend-{idist.backend()}-{idist.get_world_size()}_{now}"
    )
    output_path = Path(output_path) / folder_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    config.output_path = output_path.as_posix()
    logger.info(f"Output path: {config.output_path}")

    if config.clearml.enable:
        from clearml import Task

        task = Task.init("MVFOUL-Training", task_name=output_path.stem)
        task.connect_configuration(config)
        # Log hyper parameters
        hyper_params = [
            "model",
            "batch_size",
            "momentum",
            "weight_decay",
            "num_epochs",
            "learning_rate",
            "num_warmup_epochs",
        ]
        task.connect({k: v for k, v in config.training.hyper_params.items()})



def create_evaluator(model, metrics, config):
    device = idist.device()
    use_amp = config.training.with_amp

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            offence_sev_target, action_target, clips, _ = batch
            clips = clips.to(device, non_blocking=True).float()
            offence_sev_target = offence_sev_target.to(device, non_blocking=True)
            action_target = action_target.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    action_pred, offence_sev_pred = model(clips)
            else:
                action_pred, offence_sev_pred = model(clips)

        return {
            "offence_sev_pred": offence_sev_pred,
            "action_pred": action_pred,
            "offence_sev_target": offence_sev_target,
            "action_target": action_target,
        }

    evaluator = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator

def get_save_handler(config):
    if config.clearml.enable:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config.output_path)

    return config.output_path

def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint

def log_basic_info(logger, config):
    logger.info(f"Train on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )

def get_criterion(cfg):
    return get_loss_fn(cfg.training.severity.loss_fn), get_loss_fn(cfg.training.action.loss_fn)

def get_optimizer(config, model):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.hyper_params.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=config.training.hyper_params.weight_decay,
        eps=1e-8,

    )
    optimizer = idist.auto_optim(optimizer)

    return optimizer

def get_lr_scheduler(config, optimizer):
    milestones_values = [
        (0, 0.0),
        (config.training.hyper_params.num_iters_per_epoch * config.training.hyper_params.num_warmup_epochs, config.training.hyper_params.learning_rate),
        (config.training.hyper_params.num_iters_per_epoch * config.training.hyper_params.num_epochs, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
    return lr_scheduler