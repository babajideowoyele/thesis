from calendar import c
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
            for k in preds.keys():
                bal_acc[k] = du.all_reduce([bal_acc[k]])[0]
            acc = du.all_reduce([acc])[0]

        bal_acc["joint"] = torch.mean(torch.stack([bal_acc[k] for k in bal_acc.keys()]))
        bal_acc["accuracy"] = acc

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
        logger.info("Train with config:")
        logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model, _ = build_model(cfg)

    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)
    
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


