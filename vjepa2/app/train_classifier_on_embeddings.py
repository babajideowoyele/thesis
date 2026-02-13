#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a classifier on precomputed VJEPA embeddings.

Single-GPU usage:
    python app/train_classifier_on_embeddings.py --config-name embedding_train

Distributed (multi-GPU, single node) usage:
    torchrun --nproc_per_node=4 app/train_classifier_on_embeddings.py --config-name embedding_train
"""

from datetime import datetime
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ignite.distributed.auto import DistributedProxySampler
from omegaconf import DictConfig, OmegaConf
import hydra
from src import get_model_class
from src.utils.loss import get_loss_fn
import wandb

from src.utils.embedding_loader import (
    PrecomputedEmbeddingDataset,
    train_classifier,
)


def find_dotenv_in_parents():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at root
        env_path = os.path.join(current, '.env')
        if os.path.exists(env_path):
            return env_path
        current = os.path.dirname(current)
    return None


def setup_distributed():
    """
    Initialize distributed training from torchrun env vars.
    Returns (local_rank, rank, world_size, is_distributed).
    Falls back to single-GPU if not launched via torchrun.
    """
    if 'RANK' not in os.environ:
        return 0, 0, 1, False

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Disable P2P to avoid hangs on cross-NUMA GPU topologies (SYS link)
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size, True


def cleanup_distributed():
    """Destroy distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(config_path="../conf", config_name="embedding_train", version_base=None)
def main(cfg: DictConfig):
    """Train classifier using Hydra configuration."""

    # --- Distributed setup ---
    local_rank, rank, world_size, is_distributed = setup_distributed()
    is_main = rank == 0

    if is_distributed:
        device = f"cuda:{local_rank}"
    else:
        device = cfg.optimization.device

    # Initialize Weights & Biases (rank 0 only)
    run = None
    if cfg.WANDB.SYNC_ENABLE and is_main:
        from dotenv import load_dotenv
        env_path = find_dotenv_in_parents()
        load_dotenv(env_path)
        wandb.login(key=os.getenv("WANDB"))
        run = wandb.init(
            entity=cfg.WANDB.ENTITY_NAME,
            project=cfg.WANDB.PROJECT_NAME,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.WANDB.RUN_NAME}-{datetime.now().strftime('%m%d-%H%M')}",
        )
    
    if cfg.logging.verbose and is_main:
        print("\n" + "="*60)
        print("Training Classifier on VJEPA Embeddings")
        if is_distributed:
            print(f"  Distributed: {world_size} GPUs")
        print("="*60)
        print(OmegaConf.to_yaml(cfg))
        print("="*60 + "\n")
    
    # Set seed for reproducibility
    if cfg.meta.seed is not None:
        torch.manual_seed(cfg.meta.seed + rank)  # Different seed per rank for data augmentation diversity
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.meta.seed + rank)
    
    # Load dataset
    if cfg.logging.verbose and is_main:
        print(f"Loading embeddings from {cfg.embeddings.train_path}...")
    
    dataset = PrecomputedEmbeddingDataset(cfg.embeddings.train_path, consolidate=cfg.data.consolidate, keep_in_memory=cfg.data.keep_in_memory)
    
    if cfg.logging.verbose and is_main:
        print(f"✓ Loaded {len(dataset)} samples")
        print(f"  Embedding dimension: {dataset.get_embedding_dim()}")
        print(f"  Number of classes: {dataset.get_num_classes()}")
        
        # Print class distribution
        class_dist = dataset.get_class_distribution()
        print(f"  Class distribution:")
        for class_id in sorted(class_dist.keys()):
            class_name = dataset.get_class_name(class_id) or "unknown"
            print(f"    {class_name}: {class_dist[class_id]}")
    

    if os.path.exists(cfg.embeddings.val_path):
        if cfg.logging.verbose and is_main:
            print(f"\nLoading validation embeddings from {cfg.embeddings.val_path}...")
        
        val_dataset = PrecomputedEmbeddingDataset(cfg.embeddings.val_path, consolidate=cfg.data.consolidate)
        train_dataset = dataset  # Use full dataset as training set
        
        if cfg.logging.verbose and is_main:
            print(f"✓ Loaded {len(val_dataset)} validation samples")
    else:
        if cfg.logging.verbose and is_main:
            print(f"\nNo separate validation embeddings found, splitting training data...")
        # Split into train/val
        total_samples = len(dataset)
        val_samples = int(total_samples * cfg.data.val_split)
        train_samples = total_samples - val_samples
    
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_samples, val_samples],
            generator=torch.Generator().manual_seed(cfg.meta.seed or 42)
        )
    
    if cfg.logging.verbose and is_main:
        print(f"\n  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
    
    # Create weighted sampler for class balance
    all_weights = dataset.get_weights()
    if hasattr(train_dataset, 'indices'):
        sample_weights = [all_weights[idx] for idx in train_dataset.indices]
    else:
        sample_weights = all_weights
    
    weighted_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    
    if is_distributed:
        # Wrap WeightedRandomSampler with DistributedProxySampler to preserve
        # class-balanced sampling while partitioning data across ranks.
        train_sampler = DistributedProxySampler(
            weighted_sampler,
            num_replicas=world_size,
            rank=rank,
        )
        val_sampler = None  # Val loader doesn't need weighted sampling or shuffling
        if cfg.logging.verbose and is_main:
            print(f"\n  Using DistributedProxySampler(WeightedRandomSampler) ({world_size} replicas)")
    else:
        train_sampler = weighted_sampler
        val_sampler = None
        if cfg.logging.verbose and is_main:
            print(f"\n  Using WeightedRandomSampler for class balance")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    
    # Create model
    embedding_dim = dataset.get_embedding_dim()
    num_classes = dataset.get_num_classes()
    
    # Override config if not set
    if cfg.model.embedding_dim == 1024 and embedding_dim != 1024:
        cfg.model.embedding_dim = embedding_dim
    if cfg.model.num_classes == 9 and num_classes != 9:
        cfg.model.num_classes = num_classes

    if cfg.model.dropout is not None:
        cfg.model.attn_drop = cfg.model.dropout
        cfg.model.proj_drop = cfg.model.dropout

    model_class = get_model_class(cfg.model.name)
    model = model_class(cfg.model)
    
    
    # Move model to device and optionally wrap in DDP
    model = model.to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if cfg.logging.verbose and is_main:
        # Count params on the underlying model (not the DDP wrapper)
        base_model = model.module if is_distributed else model
        num_params = sum(p.numel() for p in base_model.parameters())
        print(f"\n✓ Model created with {num_params:,} parameters")
        if is_distributed:
            print(f"  Wrapped in DistributedDataParallel")
    
    # Train
    if cfg.logging.verbose and is_main:
        print(f"\nTraining for {cfg.training.num_epochs} epochs...")
    
    loss_func = get_loss_fn(cfg.loss)
    
    # Pass the DistributedSampler for set_epoch() calls (only for distributed)
    dist_train_sampler = train_sampler if is_distributed else None
    
    history = train_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        device=device,
        verbose=cfg.logging.verbose and cfg.logging.log_interval > 0,
        loss_func=loss_func,
        cfg=cfg,
        log_interval=cfg.logging.log_interval,
        min_lr=cfg.training.scheduler.min_lr,
        rank=rank,
        world_size=world_size,
        train_sampler=dist_train_sampler,
    )
    
    # Everything below is rank 0 only
    if is_main:
        # Log training history to wandb
        if cfg.WANDB.SYNC_ENABLE and run is not None:
            for epoch in range(len(history['train_loss'])):
                wandb.log({
                    "epoch": epoch,
                    "train_loss": history['train_loss'][epoch],
                    "train_acc": history['train_acc'][epoch],
                    "val_loss": history['val_loss'][epoch] if history["val_loss"] else None,
                    "val_acc": history['val_acc'][epoch] if history["val_acc"] else None,
                    "val_bal_acc": history['val_bal_acc'][epoch] if history["val_bal_acc"] else None,
                })
        
        # Save model (unwrap DDP if needed)
        output_dir = Path(cfg.logging.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_model = model.module if is_distributed else model
        checkpoint_path = output_dir / "classifier.pt"
        torch.save({
            "model_state_dict": base_model.state_dict(),
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "history": history,
            "config": OmegaConf.to_container(cfg),
        }, checkpoint_path)
        
        if cfg.logging.verbose:
            print(f"\n✓ Model saved to {checkpoint_path}")
            
            # Print final results
            print(f"\n{'='*60}")
            print(f"Final Results")
            print(f"{'='*60}")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Train Acc: {history['train_acc'][-1]:.4f}")
            if history["val_loss"]:
                print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
            print(f"  Val Acc: {max(history['val_acc']):.4f}")
            if history["val_bal_acc"]:
                print(f"  Val Balanced Acc: {max(history['val_bal_acc']):.4f}")
            print(f"{'='*60}\n")
        
        # Log final results to wandb
        if cfg.WANDB.SYNC_ENABLE and run is not None:
            best_val_acc_idx = history['val_acc'].index(max(history['val_acc']))
            wandb.log({
                "final_train_loss": history['train_loss'][-1],
                "final_train_acc": history['train_acc'][-1],
                "final_val_loss": history['val_loss'][best_val_acc_idx] if history["val_loss"] else None,
                "final_val_acc": max(history['val_acc']) if history["val_acc"] else None,
                "final_val_bal_acc": max(history['val_bal_acc']) if history["val_bal_acc"] else None,
            })
            run.finish()
    
    # Clean up distributed
    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
