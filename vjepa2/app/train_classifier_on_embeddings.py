#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a classifier on precomputed VJEPA embeddings.

Usage:
    python app/train_classifier_on_embeddings.py --config-name embedding_train \
        embeddings.train_path=./embeddings/mvfoul_train_embeddings.pt
        
    Or with defaults:
    python app/train_classifier_on_embeddings.py --config-name embedding_train
"""

from datetime import datetime
import os
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb

from src.models.attentive_pooler import AttentiveClassifier, AttentivePooler
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

@hydra.main(config_path="../conf", config_name="embedding_train", version_base=None)
def main(cfg: DictConfig):
    """Train classifier using Hydra configuration."""
    
    # Initialize Weights & Biases
    if cfg.WANDB.SYNC_ENABLE:
        from dotenv import load_dotenv
        env_path = find_dotenv_in_parents()
        load_dotenv(env_path)
        wandb.login(key=os.getenv("WANDB"))
        wandb.init(
            entity=cfg.WANDB.ENTITY_NAME,
            project=cfg.WANDB.PROJECT_NAME,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=f"{cfg.WANDB.RUN_NAME}-{datetime.now().strftime('%m%d-%H%M')}",
        )
    
    if cfg.logging.verbose:
        print("\n" + "="*60)
        print("Training Classifier on VJEPA Embeddings")
        print("="*60)
        print(OmegaConf.to_yaml(cfg))
        print("="*60 + "\n")
    
    # Set seed for reproducibility
    if cfg.meta.seed is not None:
        torch.manual_seed(cfg.meta.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.meta.seed)
    
    # Load dataset
    if cfg.logging.verbose:
        print(f"Loading embeddings from {cfg.embeddings.train_path}...")
    
    dataset = PrecomputedEmbeddingDataset(cfg.embeddings.train_path)
    
    if cfg.logging.verbose:
        print(f"✓ Loaded {len(dataset)} samples")
        print(f"  Embedding dimension: {dataset.get_embedding_dim()}")
        print(f"  Number of classes: {dataset.get_num_classes()}")
        
        # Print class distribution
        dist = dataset.get_class_distribution()
        print(f"  Class distribution:")
        for class_id in sorted(dist.keys()):
            class_name = dataset.get_class_name(class_id) or "unknown"
            print(f"    {class_name}: {dist[class_id]}")
    
    # Split into train/val
    total_samples = len(dataset)
    val_samples = int(total_samples * cfg.data.val_split)
    train_samples = total_samples - val_samples
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_samples, val_samples]
    )
    
    if cfg.logging.verbose:
        print(f"\n  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.data.shuffle,
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
    
    model = AttentiveClassifier(
        embed_dim=cfg.model.embedding_dim,
        num_classes=cfg.model.num_classes,
    )
    
    if cfg.logging.verbose:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Model created with {num_params:,} parameters")
    
    # Train
    if cfg.logging.verbose:
        print(f"\nTraining for {cfg.training.num_epochs} epochs...")
    
    history = train_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.training.num_epochs,
        learning_rate=cfg.training.learning_rate,
        device=cfg.optimization.device,
        verbose=cfg.logging.verbose and cfg.logging.log_interval > 0,
    )
    
    # Log training history to wandb
    for epoch in range(len(history['train_loss'])):
        wandb.log({
            "epoch": epoch,
            "train_loss": history['train_loss'][epoch],
            "train_acc": history['train_acc'][epoch],
            "val_loss": history['val_loss'][epoch] if history["val_loss"] else None,
            "val_acc": history['val_acc'][epoch] if history["val_acc"] else None,
        })
    
    # Save model
    output_dir = Path(cfg.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / "classifier.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
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
            print(f"  Val Acc: {history['val_acc'][-1]:.4f}")
        print(f"{'='*60}\n")
    
    # Log final results and model to wandb
    wandb.log({
        "final_train_loss": history['train_loss'][-1],
        "final_train_acc": history['train_acc'][-1],
        "final_val_loss": history['val_loss'][-1] if history["val_loss"] else None,
        "final_val_acc": history['val_acc'][-1] if history["val_acc"] else None,
    })
    
    # Save model artifact to wandb
    artifact = wandb.Artifact("classifier", type="model")
    artifact.add_file(str(checkpoint_path))
    wandb.log_artifact(artifact)
    
    wandb.finish()


if __name__ == "__main__":
    main()
