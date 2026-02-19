from src import get_dataset_class
import torch
from torch.utils.data import Subset
import ignite.distributed as idist

def get_dataflow(config):
    dataset_class = get_dataset_class(config.dataset.name)
    train_dataset = dataset_class(config, split="train")
    val_dataset = dataset_class(config, split="val")

    limit_batches = config.get("limit_batches", None)

    # When limit_batches is set, use a fixed Subset so the same samples are
    # used every epoch / evaluator run (even with shuffle=True).
    if limit_batches is not None:
        train_bs = config.training.hyper_params.train.batch_size
        val_bs = config.training.hyper_params.val.batch_size

        n_train = min(limit_batches * train_bs, len(train_dataset))
        n_val = min(limit_batches * val_bs, len(val_dataset))

        # Use a seeded RNG so the chosen indices are reproducible across runs.
        g = torch.Generator().manual_seed(config.seed)
        train_indices = torch.randperm(len(train_dataset), generator=g)[:n_train].tolist()
        val_indices = torch.randperm(len(val_dataset), generator=g)[:n_val].tolist()

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = idist.auto_dataloader(
        train_dataset,
        batch_size=config.training.hyper_params.train.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = idist.auto_dataloader(
        val_dataset,
        batch_size=config.training.hyper_params.val.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
