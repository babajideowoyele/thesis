#!/usr/bin/env python3

"""Train a video classification model."""
from datetime import datetime
import numpy as np
import pprint
from omegaconf import DictConfig, OmegaConf
import torch

import os
import torch.nn as nn
import wandb

from runs.evaluate import test_model
import tadaconv.models.utils.losses as losses
import tadaconv.models.utils.optimizer as optim
import tadaconv.utils.checkpoint as cu
from tadaconv.utils.mvfoul_translation import report_bad_examples
import tadaconv.utils.tensor as tu
import tadaconv.utils.distributed as du
import tadaconv.utils.logging as logging
import tadaconv.utils.metrics as metrics
import tadaconv.utils.misc as misc
import tadaconv.utils.bucket as bu
from tadaconv.utils.meters import TestMeter, TrainMeter, ValMeter

from tadaconv.models.base.builder import build_model
from tadaconv.datasets.base.builder import build_loader, shuffle_dataset

from tadaconv.datasets.utils.mixup import Mixup

logger = logging.get_logger(__name__)


def train_epoch(
    train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        model_ema (model): the ema model to update.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (Config): The global config object.
    """
    # Enable train mode.
    model.train()
    norm_train = False
    num_norms = 0
    # Examine the training status of the batch norm modules.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            num_norms += 1
            if module.training:
                norm_train = True
                break
    logger.info(f"Norm training: {norm_train if num_norms >0 else 'No norm'}")
    # Separately examine the training status of the batch norm 1D modules,
    # as the batch norm 1D is usually used in heads, which needs to be trained
    # despite of the frozen BN in the backbone.
    norm_train = False
    num_norms = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d)):
            num_norms += 1
            if module.training:
                norm_train = True
                break
    logger.info(f"Norm 1d training: {norm_train if num_norms >0 else 'No norm'}")
    train_meter.iter_tic()
    data_size = len(train_loader)

    torch.cuda.memory_summary()
    accum_steps = cfg.TRAIN.ACCUMULATE_EVERY if cfg.TRAIN.ACCUMULATE_EVERY is not None and cfg.TRAIN.ACCUMULATE_EVERY > 0 else 1.
    bad_examples = []

    for cur_iter, (inputs, masks, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device.        
        if misc.get_num_gpus(cfg):
            if cfg.NUM_GPUS > 0 or cfg.AUGMENTATION.USE_GPU:
                inputs = tu.tensor2cuda(inputs)
                labels = tu.tensor2cuda(labels)
                masks = tu.tensor2cuda(masks)

        # perform mixup on the input
        if mixup_fn is not None:
            inputs, labels["supervised_mixup"] = mixup_fn(inputs, labels["supervised"])

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        preds, logits = model(inputs, masks)

        loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / data_size)

        loss = loss / accum_steps
        
        # Check if loss is NaN
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at epoch {cur_epoch}, iteration {cur_iter}")
            raise ValueError("NaN loss encountered during training")
        # Perform the backward pass.

        loss.backward()

        do_step = ((cur_iter + 1) % accum_steps == 0) or (cur_iter + 1 == data_size)
        
        if do_step:
            optimizer.step()
            logger.info(f"Epoch {cur_epoch+1} Iteration {cur_iter}: performed optimizer step.")
            optimizer.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        loss_for_log = loss.detach() * accum_steps

        if cfg.PRETRAIN.ENABLE or cfg.LOCALIZATION.ENABLE:
            if misc.get_num_gpus(cfg) > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                None, None, loss, lr, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            train_meter.update_custom_stats(loss_in_parts)
        else:
            assert isinstance(labels["supervised"], dict)
            balanced_acc = metrics.balanced_accuracy(preds, labels["supervised"], ks={k: v.shape[1] for k, v in preds.items()})
            
            if misc.get_num_gpus(cfg) > 1:
                loss_for_log = du.all_reduce([loss_for_log])[0].item()
                for k, v in loss_in_parts.items():
                    loss_in_parts[k] = du.all_reduce([v])[0].item()
            else:
                loss_for_log = loss_for_log.item()
                for k, v in loss_in_parts.items():
                    loss_in_parts[k] = v.item()
            bad_examples += report_bad_examples(cfg, preds, labels)
            bad_examples.sort(key=lambda x: x['score'], reverse=True)


            train_meter.set_bad_examples(bad_examples)

            train_meter.update_custom_stats(balanced_acc)
            train_meter.update_custom_stats(loss_in_parts)
            

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                # TODO: replace with relevant metrics
                loss_for_log,
                lr,
                inputs[0].size(0)
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch+cfg.TRAIN.NUM_FOLDS-1)
    train_meter.reset()


def train(rank, cfg, world_size=1):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (Config): The global config object.
    """
    # Set up environment.
    if world_size > 1:
        du.ddp_setup(rank, world_size)
    # Set random seed from configs.
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

    # Setup logging format.
    logging.setup_logging(cfg, cfg.TRAIN.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg)) if not isinstance(cfg, DictConfig) else logger.info(OmegaConf.to_yaml(cfg))

    # Build the video model and print model statistics.
    model, model_ema = build_model(cfg, rank)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    if cfg.OSS.ENABLE:
        model_bucket_name = cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        model_bucket = bu.initialize_bucket(cfg.OSS.KEY, cfg.OSS.SECRET, cfg.OSS.ENDPOINT, model_bucket_name)
    else:
        model_bucket = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    
    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(cfg, model, model_ema, optimizer, model_bucket)

    if cfg.WANDB.SYNC_ENABLE and du.is_master_proc():
        env_path = misc.find_dotenv_in_parents()
        from dotenv import load_dotenv
        load_dotenv(env_path)
        wandb.login(key=os.getenv("WANDB"))
        wandb_run = wandb.init(
            # Set the wandb run name.
            name=f"{cfg.WANDB.RUN_NAME}-{datetime.now().strftime('%m%d-%H%M')}",
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=cfg.WANDB.ENTITY_NAME,
            # Set the wandb project where this run will be logged.
            project=cfg.WANDB.PROJECT_NAME,
            # Track hyperparameters and run metadata.
            config=OmegaConf.to_container(cfg),
        )
        log_freq = cfg.WANDB.LOG_FREQUENCY if cfg.WANDB.LOG_FREQUENCY > 0 else 500
        wandb.watch(model, log="all", log_freq=log_freq)
    else:
        wandb_run = None


    # Create the video train and val loaders.
    train_loader = build_loader(cfg, "train", rank, world_size)
    # Create the video train and val loaders.
    val_loader = build_loader(cfg, "val", rank, world_size) if cfg.TRAIN.EVAL_PERIOD != 0 else None

    # Create meters.
    val_meter = TestMeter(len(val_loader), wandb=wandb_run) if val_loader is not None else None

    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg, wandb=wandb_run)

    if cfg.AUGMENTATION.MIXUP.ENABLE or cfg.AUGMENTATION.CUTMIX.ENABLE:
        logger.info("Enabling mixup/cutmix.")
        mixup_fn = Mixup(cfg)
        cfg.TRAIN.LOSS_FUNC = "soft_target"
    else:
        logger.info("Mixup/cutmix disabled.")
        mixup_fn = None
    
    if cfg.AUGMENTATION.LABEL_SMOOTHING > 0.0:
        logger.info("Enabling label smoothing.")
        cfg.TRAIN.LOSS_FUNC = "soft_target"

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    assert (cfg.OPTIMIZER.MAX_EPOCH-start_epoch)%cfg.TRAIN.NUM_FOLDS == 0, "Total training epochs should be divisible by cfg.TRAIN.NUM_FOLDS."

    for cur_epoch in range(start_epoch, cfg.OPTIMIZER.MAX_EPOCH, cfg.TRAIN.NUM_FOLDS):
        torch.cuda.memory_summary()
        # Shuffle the dataset.
        shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(
            train_loader, model, model_ema, optimizer, train_meter, cur_epoch, mixup_fn, cfg
        )
        torch.cuda.empty_cache()

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, model_ema, optimizer, cur_epoch+cfg.TRAIN.NUM_FOLDS-1, cfg, model_bucket)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch+cfg.TRAIN.NUM_FOLDS-1):
            assert val_loader is not None and val_meter is not None
            with torch.no_grad():   
                test_model(val_loader, model, val_meter, cfg)
            val_meter.reset()

    if model_bucket is not None:
        filename = os.path.join(cfg.OUTPUT_DIR, cfg.TRAIN.LOG_FILE)
        bu.put_to_bucket(
            model_bucket, 
            cfg.OSS.CHECKPOINT_OUTPUT_PATH + 'log/',
            filename,
            cfg.OSS.CHECKPOINT_OUTPUT_PATH.split('/')[2]
        )

