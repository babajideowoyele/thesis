from src import get_dataset_class
import ignite.distributed as idist

def get_dataflow(config):
    dataset_class = get_dataset_class(config.dataset.name)
    train_dataset = dataset_class(config, split="train")
    val_dataset = dataset_class(config, split="val")

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
