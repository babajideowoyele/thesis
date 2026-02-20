#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Builder for the dataloader."""

from ignite.distributed.auto import DistributedProxySampler
import torch
import tadaconv.utils.misc as misc
from tadaconv.utils.sampler import MultiFoldDistributedSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from tadaconv.utils.val_dist_sampler import MultiSegValDistributedSampler
from tadaconv.datasets.utils.collate_functions import COLLATE_FN_REGISTRY


from tadaconv.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")

def get_weights(dataset):
    """
        Returns the weights for each sample in the dataset.
        Args:
            dataset (Dataset): constructed dataset. 
        Returns:
            weights (list): list of weights for each sample in the dataset. 
    """
    if hasattr(dataset, "get_weights"):
        return dataset.get_weights()
    else:
        return [1.0/len(dataset)] * len(dataset)

def get_sampler(cfg, dataset, split, shuffle, rank, world_size):
    """
        Returns the sampler object for the dataset.
        Args:
            dataset (Dataset): constructed dataset. 
            split   (str):     which split is the dataset for.
            shuffle (bool):    whether or not to shuffle the dataset.
        Returns:
            sampler (Sampler): dataset sampler. 
    """
    if misc.get_num_gpus(cfg) > 1:
        if split == "train" and cfg.TRAIN.NUM_FOLDS > 1:
            return MultiFoldDistributedSampler(
                dataset, cfg.TRAIN.NUM_FOLDS
            )
        elif cfg.USE_MULTISEG_VAL_DIST and cfg.TRAIN.ENABLE is False:
            return MultiSegValDistributedSampler(dataset, shuffle=False)
        elif split == "train" and cfg.TRAIN.NUM_FOLDS == 1:
            weights = get_weights(dataset)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True
            )
            return DistributedProxySampler(sampler, world_size, rank)
        else:
            return DistributedSampler(
                dataset,
                shuffle=shuffle
            )
    else:
        weights = get_weights(dataset)
        return torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=len(weights),
                replacement=True
            )

def build_loader(cfg, split, rank=0, world_size=1):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object. 
    """
    batch_size = 0 # make pylint happy
    shuffle = True  # make pylint happy
    drop_last = False  # make pylint happy
    assert split in ["train", "val", "test", "submission"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE)
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE)
        shuffle = False
        drop_last = False
    elif split in ["test", "submission"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE)
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = get_sampler(cfg, dataset, split, shuffle, rank, world_size)
    # Create a loader
    if hasattr(cfg.DATA_LOADER, "COLLATE_FN") and cfg.DATA_LOADER.COLLATE_FN is not None:
        collate_fn = COLLATE_FN_REGISTRY.get(cfg.DATA_LOADER.COLLATE_FN)(cfg)
    else:
        collate_fn = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return loader


def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the sampler for the dataset.
    Args:
        loader      (loader):   data loader to perform shuffle.
        cur_epoch   (int):      number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler, MultiFoldDistributedSampler, WeightedRandomSampler, DistributedProxySampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, (DistributedSampler, MultiFoldDistributedSampler, DistributedProxySampler)):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

def build_dataset(dataset_name, cfg, split):
    """
    Builds a dataset according to the "dataset_name".
    Args:
        dataset_name (str):     the name of the dataset to be constructed.
        cfg          (Config):  global config object. 
        split        (str):     the split of the data loader.
    Returns:
        Dataset      (Dataset):    a dataset object constructed for the specified dataset_name.
    """
    name = dataset_name.capitalize()
    return DATASET_REGISTRY.get(name)(cfg, split)
