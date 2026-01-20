# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility module for loading and using precomputed VJEPA embeddings.
This enables training fast downstream classifiers without real-time feature extraction.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset that loads precomputed VJEPA embeddings from disk.
    
    Each sample is: (embedding, label, metadata)
    """
    
    def __init__(self, embeddings_path: str, config_path: Optional[str] = None):
        """
        Initialize dataset from precomputed embeddings.
        
        Args:
            embeddings_path: Path to .pt file containing embeddings dict
            config_path: Path to .json config file (auto-generated if not specified)
        """
        self.embeddings_path = Path(embeddings_path)
        
        # Load embeddings
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        self.embeddings_dict = torch.load(embeddings_path, map_location="cpu")
        
        # Load config if provided
        if config_path is None:
            config_path = str(self.embeddings_path).replace("embeddings.pt", "config.json")
        
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = None
        
        # Create index mapping (handle sparse indices)
        self.indices = sorted(self.embeddings_dict.keys())
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get embedding and label for index.
        
        Returns:
            (embedding, label)
        """
        sample_idx = self.indices[idx]
        sample = self.embeddings_dict[sample_idx]
        
        return sample["features"], sample["label"]
    
    def get_by_id(self, sample_id: int) -> Dict[str, Any]:
        """Get full sample info by ID."""
        if sample_id not in self.embeddings_dict:
            raise KeyError(f"Sample ID {sample_id} not in embeddings")
        return self.embeddings_dict[sample_id]
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.config:
            return self.config["embed_dim"]
        # Infer from first sample
        first_sample = self.embeddings_dict[self.indices[0]]
        return first_sample["features"].shape[0]
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        if self.config and "classes" in self.config:
            return len(self.config["classes"])
        # Infer from labels
        labels = {sample["label"] for sample in self.embeddings_dict.values()}
        return max(labels) + 1
    
    def get_class_name(self, class_id: int) -> Optional[str]:
        """Get class name for ID."""
        if self.config and "classes" in self.config:
            return self.config["classes"].get(str(class_id))
        return None
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get count of samples per class."""
        distribution = {}
        for sample in self.embeddings_dict.values():
            label = sample["label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def load_embedding_dataloader(
    embeddings_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    config_path: Optional[str] = None,
) -> Tuple[DataLoader, PrecomputedEmbeddingDataset]:
    """
    Create DataLoader from precomputed embeddings.
    
    Args:
        embeddings_path: Path to embeddings .pt file
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory
        config_path: Optional path to config file
    
    Returns:
        (dataloader, dataset)
    """
    dataset = PrecomputedEmbeddingDataset(embeddings_path, config_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )
    
    return dataloader, dataset


def train_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
) -> Dict[str, list]:
    """
    Train classifier on precomputed embeddings.
    
    Args:
        model: Classifier model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary of training history
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_samples = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_acc += (outputs.argmax(1) == labels).sum().item()
            num_samples += labels.size(0)
        
        train_loss /= num_samples
        train_acc /= num_samples
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            num_val_samples = 0
            
            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(embeddings)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    val_acc += (outputs.argmax(1) == labels).sum().item()
                    num_val_samples += labels.size(0)
            
            val_loss /= num_val_samples
            val_acc /= num_val_samples
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
        
        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1}/{num_epochs}: "
            msg += f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
            if val_loader:
                msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            print(msg)
    
    return history
