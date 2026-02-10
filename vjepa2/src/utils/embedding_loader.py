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
from torch.optim.lr_scheduler import CosineAnnealingLR


class PrecomputedEmbeddingDataset(Dataset):
    """
    Dataset that loads precomputed VJEPA embeddings from disk.
    
    Each sample is: (embedding, label, metadata)
    """
    
    def __init__(self, embeddings_path: str, config_path: Optional[str] = None, consolidate: bool = False):
        """
        Initialize dataset from precomputed embeddings.
        
        Args:
            embeddings_path: Path to .pt file containing embeddings dict (or pattern for chunked files)
            config_path: Path to .json config file (auto-generated if not specified)
            consolidate: Whether to consolidate samples by action_dir
        """
        self.embeddings_path = Path(embeddings_path)
        
        # Load config first to check if chunked
        if config_path is None:
            # Try to infer config path
            config_path = str(self.embeddings_path).replace("_embeddings.pt", "_config.json")
            # Also handle chunked file naming
            if "_embeddings_chunk" in str(self.embeddings_path):
                config_path = str(self.embeddings_path).split("_embeddings_chunk")[0] + "_config.json"
        
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = None
        
        # Check if embeddings are chunked
        is_chunked = self.config and self.config.get("chunked", False)
        
        if is_chunked:
            # Load from multiple chunk files
            self.embeddings_dict = self._load_chunked_embeddings()
        else:
            # Load from single file (original behavior)
            if not self.embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            self.embeddings_dict = torch.load(embeddings_path, map_location="cpu")
        
        # Consolidate by action_dir if requested
        if consolidate:
            self._consolidate_by_action_dir()
        
        # Create index mapping (handle sparse indices)
        self.indices = sorted(self.embeddings_dict.keys())
        self.labels = [self.embeddings_dict[idx]["label"] for idx in self.indices]
    
    def _load_chunked_embeddings(self) -> Dict[int, Dict[str, Any]]:
        """Load embeddings from multiple chunk files."""
        if self.config is None:
            raise ValueError("Config file required for chunked embeddings")
        
        num_chunks = self.config.get("num_chunks", 1)
        
        # Determine chunk file pattern from embeddings_path
        # Handle both direct path to chunk and base path
        base_path_str = str(self.embeddings_path)
        if "_embeddings_chunk" in base_path_str:
            # Extract base pattern (e.g., "mvfoul_train_embeddings")
            base_path_str = base_path_str.split("_embeddings_chunk")[0] + "_embeddings"
        else:
            # Remove .pt extension
            base_path_str = base_path_str.replace("_embeddings.pt", "_embeddings")
        
        # Load all chunks
        combined_dict = {}
        for chunk_idx in range(num_chunks):
            chunk_path = Path(f"{base_path_str}_chunk{chunk_idx:04d}.pt")
            if not chunk_path.exists():
                raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
            
            chunk_dict = torch.load(chunk_path, map_location="cpu")
            combined_dict.update(chunk_dict)
        
        return combined_dict
        
    
    def _consolidate_by_action_dir(self):
        """
        Consolidate samples from the same action_dir by concatenating their features.
        Handles both (S,D) and (D,) shaped features.
        Updates self.embeddings_dict with consolidated samples.
        """
        # Group by action_dir
        action_dir_groups = {}
        max_seq_len = 0
        for idx, sample in self.embeddings_dict.items():
            action_dir = sample.get('action_dir')
            if action_dir is None:
                # Skip samples without action_dir
                continue
            if action_dir not in action_dir_groups:
                action_dir_groups[action_dir] = []
            if len(sample['features'].shape) == 1:
                # (D,) -> (1,D)
                sample['features'] = sample['features'].unsqueeze(0)
            action_dir_groups[action_dir].append((idx, sample))
        
        for item in action_dir_groups.values():
            num_vids = len(item)
            seq_len=num_vids*item[0][1]["features"].shape[0]
            if seq_len > max_seq_len:
                max_seq_len = seq_len
            
        
        if not action_dir_groups:
            # No action_dir found, skip consolidation
            return
        
        # Create new consolidated dict
        consolidated_dict = {}
        new_idx = 0
        
        for action_dir, samples in action_dir_groups.items():
            # Sort by original index to maintain consistent ordering
            samples = sorted(samples, key=lambda x: x[0])
            
            # Extract features and normalize shape
            features_list = []
            for _, sample in samples:
                features = sample['features']
                # Handle shape: (D,) -> (1,D), (S,D) stays (S,D)
                features_list.append(features)
            if sum(f.shape[0] for f in features_list) > max_seq_len:
                max_seq_len = sum(f.shape[0] for f in features_list)
            
            # Concatenate along sequence dimension
            consolidated_features = torch.zeros((max_seq_len, features_list[0].shape[1]))
            unpad_features = torch.cat(features_list, dim=0)
            consolidated_features[:unpad_features.shape[0]] = unpad_features  # (S_total, D)
            
            # Use metadata from first sample (they should all have same label/action_class)
            first_sample = samples[0][1]
            first_sample['features'] = consolidated_features
            consolidated_dict[new_idx] = first_sample.copy()
            new_idx += 1
        
        self.embeddings_dict = consolidated_dict
    
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
        return first_sample["features"].shape[-1]
    
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
    
    def get_weights(self) -> list:
        """Calculate sample weights for balanced sampling (inverse frequency)."""
        # Count samples per class
        class_counts = {}
        for sample_idx in self.indices:
            sample = self.embeddings_dict[sample_idx]
            label = sample["label"]
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Calculate inverse frequency weights
        factors = {label: 1.0 / torch.sqrt(torch.tensor(count)).item() for label, count in class_counts.items()}
        
        # Assign weight to each sample based on its class
        weights = []
        for sample_idx in self.indices:
            sample = self.embeddings_dict[sample_idx]
            label = sample["label"]
            weights.append(factors[label])
        
        return weights


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
    
    Automatically handles both single-file and chunked embeddings based on config.
    
    Args:
        embeddings_path: Path to embeddings .pt file (or any chunk file for chunked embeddings)
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
    min_lr: float = 1e-6,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
    loss_func: Optional[torch.nn.Module] = None,
    cfg: Optional[Dict[str, Any]] = None,
    log_interval: int = 10,
) -> Dict[str, list]:
    """
    Train classifier on precomputed embeddings.
    
    Args:
        model: Classifier model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        min_lr: Minimum learning rate for cosine annealing
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary of training history (includes per-epoch lr)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.training.weight_decay if cfg else 0)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    criterion = torch.nn.CrossEntropyLoss() if loss_func is None else loss_func.to(device)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal_acc": [],
        "lr": [],
    }
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_samples = 0
        
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            if len(embeddings.shape) == 2:
                embeddings = embeddings.unsqueeze(1)  # Add sequence dim if missing
            assert len(embeddings.shape) == 3, "Embeddings must be (B, S, D)"
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
        history["lr"].append(current_lr)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            num_val_samples = 0
            num_classes = None
            val_correct_per_class = None
            val_total_per_class = None
            
            with torch.no_grad():
                for embeddings, labels in val_loader:
                    embeddings = embeddings.to(device)
                    labels = labels.to(device)
                    if len(embeddings.shape) == 2:
                        embeddings = embeddings.unsqueeze(1)  # Add sequence dim if missing
                    outputs = model(embeddings)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * labels.size(0)
                    val_acc += (outputs.argmax(1) == labels).sum().item()

                    # Balanced accuracy: average recall across classes
                    preds = outputs.argmax(1)
                    if num_classes is None:
                        num_classes = outputs.size(1)
                        val_correct_per_class = torch.zeros(num_classes, device=device)
                        val_total_per_class = torch.zeros(num_classes, device=device)

                    for cls_idx in range(num_classes):
                        cls_mask = labels == cls_idx
                        cls_total = cls_mask.sum()
                        val_total_per_class[cls_idx] += cls_total
                        if cls_total > 0:
                            val_correct_per_class[cls_idx] += (preds[cls_mask] == cls_idx).sum()
                    num_val_samples += labels.size(0)
            
            val_loss /= num_val_samples
            val_acc /= num_val_samples
            if num_classes is None:
                val_bal_acc = 0.0
            else:
                class_recalls = []
                for cls_idx in range(num_classes):
                    cls_total = val_total_per_class[cls_idx].item()
                    if cls_total == 0:
                        class_recalls.append(0.0)
                    else:
                        class_recalls.append(val_correct_per_class[cls_idx].item() / cls_total)
                val_bal_acc = sum(class_recalls) / num_classes
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_bal_acc"].append(val_bal_acc)
        
        if verbose and (epoch + 1) % log_interval == 0:
            msg = f"Epoch {epoch+1}/{num_epochs}: "
            msg += f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, lr={current_lr:.6f}"
            if val_loader:
                msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_bal_acc={val_bal_acc:.4f}"
            print(msg)

        scheduler.step()
    
    return history
