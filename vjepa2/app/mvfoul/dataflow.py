from src import get_dataset_class
from torch.utils.data import DataLoader

def get_dataflow(config):
    dataset_class = get_dataset_class(config.dataset.name)
    train_dataset = dataset_class(config, split="train")
    val_dataset = dataset_class(config, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
