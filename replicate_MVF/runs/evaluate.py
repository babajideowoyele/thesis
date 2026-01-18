import os
import pprint
import torch
import numpy as np
import wandb
from tadaconv.datasets.base.builder import build_loader
from tadaconv.datasets.utils.mixup import Mixup
from tadaconv.models.base.builder import build_model
from tadaconv.utils import logging, metrics, misc
import tadaconv.utils.checkpoint as cu
import tadaconv.utils.distributed as du
from tadaconv.utils.meters import TestMeter
import tadaconv.utils.tensor as tu



logger = logging.get_logger(__name__)

@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Discontinued
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model/model_ema to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (Config): The global config object.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, masks, labels) in enumerate(val_loader):
        if misc.get_num_gpus(cfg):
            if cfg.NUM_GPUS > 0 or cfg.AUGMENTATION.USE_GPU:
                inputs = tu.tensor2cuda(inputs)
                labels = tu.tensor2cuda(labels)
                masks = tu.tensor2cuda(masks)

        preds, logits = model(inputs, masks)
        if cfg.PRETRAIN.ENABLE and (cfg.PRETRAIN.GENERATOR == 'MoSIGenerator'):
            if "move_x" in preds.keys():
                preds["move_joint"] = preds["move_x"]
            elif "move_y" in preds.keys():
                preds["move_joint"] = preds["move_y"]
            num_topks_correct = metrics.topks_correct(preds["move_joint"], labels["self-supervised"]["move_joint"].reshape(preds["move_joint"].shape[0]), (1, 4))
            top1_err, top5_err = [
                (1.0 - x / preds["move_joint"].shape[0]) * 100.0 for x in num_topks_correct
            ]
            if misc.get_num_gpus(cfg) > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            top1_err, top5_err = top1_err.item(), top5_err.item()
            val_meter.iter_toc()
            val_meter.update_stats(
                top1_err,
                top5_err,
                preds["move_joint"].shape[0]
                * max(
                    misc.get_num_gpus(cfg), 1
                ),
            )
            val_meter.update_predictions(preds, labels)
        elif cfg.LOCALIZATION.ENABLE:
            loss, loss_in_parts, weight = losses.calculate_loss(cfg, preds, logits, labels, cur_epoch + cfg.TRAIN.NUM_FOLDS * float(cur_iter) / len(val_loader))
            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                0, 0, inputs["video"].shape[0] if isinstance(inputs, dict) else inputs.shape[0]
            )
            loss_in_parts["loss"] = loss
            val_meter.update_custom_stats(loss_in_parts)
        else:
            top1_err, top5_err = None, None
            if isinstance(labels["supervised"], dict):
                top1_err_all = {}
                top5_err_all = {}
                num_topks_correct, b = metrics.joint_topks_correct(preds, labels["supervised"], (1, 4))
                ks = {
                    i.lower(): [0.0]*len(n) for i, n in cfg.DATA.FREQUENCIES.items()
                    }
                for name, label in labels["supervised"].items():
                    for c in label:
                        ks[name][c] += 1.0
                balanced_acc = metrics.balanced_accuracy(preds, labels["supervised"], ks=ks)
                for k, v in num_topks_correct.items():
                    # Compute the errors.
                    top1_err_split, top5_err_split = [
                        (1.0 - x / b) * 100.0 for x in v
                    ]

                    # Gather all the predictions across all the devices.
                    if misc.get_num_gpus(cfg) > 1:
                        top1_err_split, top5_err_split = du.all_reduce(
                            [top1_err_split, top5_err_split]
                        )

                    # Copy the stats from GPU to CPU (sync point).
                    top1_err_split, top5_err_split = (
                        top1_err_split.item(),
                        top5_err_split.item(),
                    )
                    if "joint" not in k:
                        top1_err_all["top1_err_"+k] = top1_err_split
                        top5_err_all["top5_err_"+k] = top5_err_split
                    else:
                        top1_err = top1_err_split
                        top5_err = top5_err_split
                val_meter.update_custom_stats(balanced_acc)
                val_meter.update_custom_stats(top1_err_all)
                val_meter.update_custom_stats(top5_err_all)
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels["supervised"], (1, 4))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if misc.get_num_gpus(cfg) > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0)
                * max(
                    misc.get_num_gpus(cfg), 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()

@torch.no_grad()
def test_model(test_loader, model, test_meter: TestMeter, cfg):
    """
    Evaluate the model on the test set.
    Args:
        test_loader (loader): data loader to provide test data.
        model (model): model/model_ema to evaluate the performance.
        test_meter (TestMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (Config): The global config object.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    test_meter.iter_tic()
    for cur_iter, (inputs, masks, labels) in enumerate(test_loader):
        if misc.get_num_gpus(cfg):
            if cfg.NUM_GPUS > 0:
                inputs = tu.tensor2cuda(inputs)
                labels = tu.tensor2cuda(labels)
                masks = tu.tensor2cuda(masks)

        preds, logits = model(inputs, masks)

        assert isinstance(labels["supervised"], dict)
        acc = metrics.accuracy(
            preds, labels["supervised"], 
            ks={k: v.shape[1] for k, v in preds.items()})
        bal_acc = metrics.balanced_accuracy(
            preds, labels["supervised"], 
            ks={k: v.shape[1] for k, v in preds.items()})
        
        # Gather all the predictions across all the devices.
        if misc.get_num_gpus(cfg) > 1:
            for k in bal_acc.keys():
                bal_acc[k] = du.all_reduce([bal_acc[k]])[0]
            for k in acc.keys():
                acc[k] = du.all_reduce([acc[k]])[0]
    
        bal_acc["joint_acc"] = torch.mean(torch.stack([bal_acc[k] for k in bal_acc.keys()]))
        for k in acc.keys():
            bal_acc["acc_"+k] = acc[k]


        test_meter.iter_toc()
        # Update and log stats.
        test_meter.log_stats(
            bal_acc,
            cur_iter,
        )

        test_meter.update_aggregation(bal_acc)
        test_meter.iter_tic()

    # Log epoch stats.
    test_meter.log_test()

def evaluate(rank, cfg, world_size=1):
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
    logging.setup_logging(cfg, cfg.TEST.LOG_FILE)

    # Print config.
    if cfg.LOG_CONFIG_INFO:
        logger.info("Test with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, _ = build_model(cfg, gpu_id=rank)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)
    
    # Load a checkpoint to resume training if applicable.
    cu.load_test_checkpoint(cfg, model, None)

    if cfg.WANDB.SYNC_ENABLE and du.is_master_proc():
        env_path = misc.find_dotenv_in_parents()
        from dotenv import load_dotenv
        load_dotenv(env_path)
        wandb.login(key=os.getenv("WANDB"))
        wandb_run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=cfg.WANDB.ENTITY_NAME,
            # Set the wandb project where this run will be logged.
            project=cfg.WANDB.PROJECT_NAME,
            # Track hyperparameters and run metadata.
            config=cfg.cfg_dict,
        )
    else:
        wandb_run = None


    # Create the video train and val loaders.
    test_loader = build_loader(cfg, "test")

    # Create meters.
    test_meter = TestMeter(len(test_loader), wandb=wandb_run)

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


    torch.cuda.memory_summary()
    # Shuffle the dataset.
    torch.cuda.empty_cache()
    # Evaluate the model on validation set.
    test_model(test_loader, model, test_meter, cfg)


