#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Loss function registry and implementations.

This module provides a registry-based system for loss functions, allowing
loss functions and classes to be registered and instantiated via Hydra configs.
"""

from typing import Callable, Dict, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class LossRegistry:
    """Registry for loss functions and classes."""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a loss function or class.
        
        Args:
            name: Optional name for the loss. If not provided, uses the class/function name.
        
        Returns:
            The decorator function.
        """
        def decorator(fn_or_cls):
            loss_name = name or fn_or_cls.__name__
            self._registry[loss_name] = fn_or_cls
            return fn_or_cls
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Retrieve a registered loss function or class.
        
        Args:
            name: The name of the loss to retrieve.
            
        Returns:
            The loss function or class.
            
        Raises:
            ValueError: If the loss is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Loss '{name}' not found. Available losses: {available}"
            )
        return self._registry[name]
    
    def list_losses(self) -> list:
        """Return a list of all registered loss names."""
        return list(self._registry.keys())


# Global registry instance
LOSS_REGISTRY = LossRegistry()


# ============================================================================
# Loss Function Implementations
# ============================================================================

@LOSS_REGISTRY.register("l1")
class L1Loss(nn.Module):
    """L1 loss (Mean Absolute Error)."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("mse")
class MSELoss(nn.Module):
    """Mean Squared Error loss."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("smooth_l1")
class SmoothL1Loss(nn.Module):
    """Smooth L1 loss."""
    
    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        super().__init__()
        self.loss = nn.SmoothL1Loss(reduction=reduction, beta=beta)
        self.beta = beta
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("cross_entropy")
class CrossEntropyLoss(nn.Module):
    """Cross Entropy loss."""
    
    def __init__(
        self,
        reduction: str = "mean",
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction=reduction,
            weight=weight,
            label_smoothing=label_smoothing,
        )
        self.label_smoothing = label_smoothing
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("bce")
class BCELoss(nn.Module):
    """Binary Cross Entropy loss."""
    
    def __init__(self, reduction: str = "mean", weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.BCELoss(reduction=reduction, weight=weight)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("bce_with_logits")
class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy loss with logits."""
    
    def __init__(self, reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=pos_weight)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("huber")
class HuberLoss(nn.Module):
    """Huber loss."""
    
    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__()
        self.loss = nn.HuberLoss(reduction=reduction, delta=delta)
        self.delta = delta
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("kl_divergence")
class KLDivergenceLoss(nn.Module):
    """KL Divergence loss."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction=reduction)
    
    def forward(self, pred, target):
        return self.loss(pred, target)


@LOSS_REGISTRY.register("cosine_embedding")
class CosineEmbeddingLoss(nn.Module):
    """Cosine Embedding loss."""
    
    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
        self.margin = margin
    
    def forward(self, pred, target, target_label):
        return self.loss(pred, target, target_label)


@LOSS_REGISTRY.register("triplet")
class TripletLoss(nn.Module):
    """Triplet loss."""
    
    def __init__(self, margin: float = 1.0, p: float = 2.0, swap: bool = False, reduction: str = "mean"):
        super().__init__()
        self.loss = nn.TripletMarginLoss(
            margin=margin,
            p=p,
            swap=swap,
            reduction=reduction,
        )
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        return self.loss(anchor, positive, negative)


@LOSS_REGISTRY.register("vjepa_lp")
class VJEPALpLoss(nn.Module):
    """
    VJEPA Lp loss: computes mean absolute difference raised to power p.
    
    This is the loss used in V-JEPA training for predicting masked tokens.
    """
    
    def __init__(self, p: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions, shape (...,)
            target: Target values, shape (...,)
        
        Returns:
            Scalar loss value.
        """
        loss = torch.mean(torch.abs(pred - target) ** self.p) / self.p
        return loss


@LOSS_REGISTRY.register("cosine_similarity")
class CosineSimilarityLoss(nn.Module):
    """Cosine similarity based loss (1 - cosine_similarity)."""
    
    def __init__(self, dim: int = 1, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions
            target: Target values
        
        Returns:
            Loss value
        """
        similarity = F.cosine_similarity(pred, target, dim=self.dim, eps=self.eps)
        loss = 1 - similarity
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@LOSS_REGISTRY.register("contrastive")
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for metric learning.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, pred1, pred2, y):
        """
        Args:
            pred1: First set of embeddings
            pred2: Second set of embeddings
            y: Binary labels (1 for similar, 0 for dissimilar)
        
        Returns:
            Loss value
        """
        distance = F.pairwise_distance(pred1, pred2)
        loss = y * distance.pow(2) + (1 - y) * torch.clamp(self.margin - distance, min=0).pow(2)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@LOSS_REGISTRY.register("two_way_multilabel")
class TwoWayMultiLabelLoss(nn.Module):
    """
    Two-way loss for multi-label classification.
    
    Combines sample-wise and class-wise losses to enhance discrimination
    in both dimensions for video classification with multiple labels.
    
    The overall loss is: L = L_sample_wise + alpha * L_class_wise
    
    Args:
        alpha: Balancing parameter for class-wise loss (default: 1.0)
        gamma: Temperature parameter for softplus (default: 1.0)
        label_smoothing: Label smoothing factor between 0 and 1 (default: 0.0)
                        Positive labels become (1 - label_smoothing)
                        Negative labels become (label_smoothing / num_classes)
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        if not 0.0 <= label_smoothing <= 0.5:
            raise ValueError(f"label_smoothing must be between 0 and 0.5, got {label_smoothing}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: Logit matrix of shape (M, C) where M is the number of video samples
                   and C is the number of classes
            targets: Either:
                    - Integer tensor of shape (M,) with class indices
                    - Binary label matrix of shape (M, C) where 1 indicates positive
                      and 0 indicates negative class for each sample
        
        Returns:
            Scalar loss value combining sample-wise and class-wise losses
        """
        M, C = logits.shape
        
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            # targets is (M,) with class indices, convert to one-hot (M, C)
            targets = F.one_hot(targets, num_classes=C).float()
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            targets = targets * (1.0 - self.label_smoothing) + (self.label_smoothing / C)
        
        # Sample-wise loss: discriminate positive and negative classes for each sample
        sample_loss = 0.0
        for m in range(M):
            # Get positive and negative class indices for this sample
            if self.label_smoothing > 0.0:
                # With label smoothing, use threshold to determine positive/negative
                threshold = 0.5
                pos_mask = targets[m] >= threshold
                neg_mask = targets[m] < threshold
            else:
                pos_mask = targets[m] == 1
                neg_mask = targets[m] == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Positive class logits
                x_pos = logits[m][pos_mask]
                # Negative class logits
                x_neg = logits[m][neg_mask]
                
                # Compute sample-wise loss component
                term1 = torch.logsumexp(x_neg, dim=0)
                term2 = self.gamma * torch.logsumexp(-x_pos / self.gamma, dim=0)
                sample_loss += F.softplus(term1 + term2)
        
        sample_loss = sample_loss / M
        
        # Class-wise loss: discriminate samples within each class
        class_loss = 0.0
        for c in range(C):
            # Get positive and negative sample indices for this class
            if self.label_smoothing > 0.0:
                # With label smoothing, use threshold to determine positive/negative
                threshold = 0.5
                pos_mask = targets[:, c] >= threshold
                neg_mask = targets[:, c] < threshold
            else:
                pos_mask = targets[:, c] == 1
                neg_mask = targets[:, c] == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                # Positive sample logits for this class
                x_pos = logits[pos_mask, c]
                # Negative sample logits for this class
                x_neg = logits[neg_mask, c]
                
                # Compute class-wise loss component
                term1 = torch.logsumexp(x_neg, dim=0)
                term2 = self.gamma * torch.logsumexp(-x_pos / self.gamma, dim=0)
                class_loss += F.softplus(term1 + term2)
        
        class_loss = class_loss / C
        
        # Combine losses
        total_loss = sample_loss + self.alpha * class_loss
        
        return total_loss


# ============================================================================
# Loss Instantiation Function
# ============================================================================

def get_loss_fn(cfg: DictConfig) -> nn.Module:
    """
    Instantiate a loss function from a Hydra config.
    
    Args:
        cfg: Hydra config dictionary with 'name' and optional parameters.
             Example:
                 loss:
                     name: vjepa_lp
                     p: 2.0
                     reduction: mean
    
    Returns:
        Instantiated loss module.
    
    Example:
        >>> cfg = DictConfig({'name': 'mse', 'reduction': 'mean'})
        >>> loss_fn = get_loss_fn(cfg)
        >>> pred = torch.randn(10, 5)
        >>> target = torch.randn(10, 5)
        >>> loss = loss_fn(pred, target)
    """
    if isinstance(cfg, str):
        # If cfg is just a string, use default parameters
        loss_name = cfg
        loss_cls = LOSS_REGISTRY.get(loss_name)
        return loss_cls()
    
    # Extract loss name from config
    loss_name = cfg.name
    if loss_name is None:
        raise ValueError("Loss config must have a 'name' field")
    
    # Get the loss class
    loss_cls = LOSS_REGISTRY.get(loss_name)
    
    # Extract parameters, excluding the 'name' field
    params = {k: v for k, v in cfg.items() if k != "name"}
    
    # Instantiate the loss
    return loss_cls(**params)


__all__ = [
    "LOSS_REGISTRY",
    "LossRegistry",
    "get_loss_fn",
    "L1Loss",
    "MSELoss",
    "SmoothL1Loss",
    "CrossEntropyLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "HuberLoss",
    "KLDivergenceLoss",
    "CosineEmbeddingLoss",
    "TripletLoss",
    "VJEPALpLoss",
    "CosineSimilarityLoss",
    "ContrastiveLoss",
    "TwoWayMultiLabelLoss",
]
