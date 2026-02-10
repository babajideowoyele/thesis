#!/usr/bin/env python3
"""Test script for loss registry."""

import torch
from omegaconf import DictConfig
from src.utils.loss import (
    LOSS_REGISTRY,
    get_loss_fn,
    VJEPALpLoss,
    MSELoss,
    CosineSimilarityLoss,
)


def test_registry():
    """Test that all losses are registered."""
    print("=" * 60)
    print("Testing Loss Registry")
    print("=" * 60)
    
    available_losses = LOSS_REGISTRY.list_losses()
    print(f"\nAvailable losses: {available_losses}")
    assert len(available_losses) > 0, "No losses registered!"
    print(f"✓ Registry contains {len(available_losses)} losses")


def test_vjepa_lp_loss():
    """Test VJEPA Lp loss."""
    print("\n" + "-" * 60)
    print("Testing VJEPA Lp Loss")
    print("-" * 60)
    
    loss_fn = VJEPALpLoss(p=2.0)
    pred = torch.randn(4, 8, 768)
    target = torch.randn(4, 8, 768)
    loss = loss_fn(pred, target)
    
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ VJEPA Lp loss computed successfully: {loss.item():.6f}")


def test_mse_loss():
    """Test MSE loss."""
    print("\n" + "-" * 60)
    print("Testing MSE Loss")
    print("-" * 60)
    
    loss_fn = MSELoss(reduction="mean")
    pred = torch.randn(4, 8, 768)
    target = torch.randn(4, 8, 768)
    loss = loss_fn(pred, target)
    
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ MSE loss computed successfully: {loss.item():.6f}")


def test_cosine_similarity_loss():
    """Test cosine similarity loss."""
    print("\n" + "-" * 60)
    print("Testing Cosine Similarity Loss")
    print("-" * 60)
    
    loss_fn = CosineSimilarityLoss(dim=1)
    pred = torch.randn(4, 768)
    target = torch.randn(4, 768)
    loss = loss_fn(pred, target)
    
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"✓ Cosine similarity loss computed successfully: {loss.item():.6f}")


def test_get_loss_fn_from_config():
    """Test getting loss from Hydra config."""
    print("\n" + "-" * 60)
    print("Testing get_loss_fn with Hydra Config")
    print("-" * 60)
    
    # Test with vjepa_lp config
    cfg = DictConfig({
        "name": "vjepa_lp",
        "p": 2.0,
        "reduction": "mean"
    })
    loss_fn = get_loss_fn(cfg)
    assert isinstance(loss_fn, VJEPALpLoss), "Should return VJEPALpLoss instance"
    
    pred = torch.randn(4, 8, 768)
    target = torch.randn(4, 8, 768)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"✓ Config-based loss instantiation works: {loss.item():.6f}")


def test_get_loss_fn_string():
    """Test getting loss from string."""
    print("\n" + "-" * 60)
    print("Testing get_loss_fn with String")
    print("-" * 60)
    
    loss_fn = get_loss_fn("mse")
    assert isinstance(loss_fn, MSELoss), "Should return MSELoss instance"
    
    pred = torch.randn(4, 8, 768)
    target = torch.randn(4, 8, 768)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"✓ String-based loss instantiation works: {loss.item():.6f}")


def test_loss_backward():
    """Test that loss gradients flow properly."""
    print("\n" + "-" * 60)
    print("Testing Gradient Flow")
    print("-" * 60)
    
    loss_fn = VJEPALpLoss(p=2.0)
    pred = torch.randn(4, 8, 768, requires_grad=True)
    target = torch.randn(4, 8, 768)
    
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradients should flow through loss"
    assert (pred.grad != 0).any(), "Gradients should be non-zero"
    print(f"✓ Gradients flow correctly through loss")


def test_all_losses():
    """Test instantiating all registered losses."""
    print("\n" + "-" * 60)
    print("Testing All Registered Losses")
    print("-" * 60)
    
    losses_to_test = [
        "l1", "mse", "smooth_l1", "cross_entropy", "bce",
        "bce_with_logits", "huber", "vjepa_lp", "cosine_similarity"
    ]
    
    for loss_name in losses_to_test:
        try:
            loss_fn = get_loss_fn(loss_name)
            assert loss_fn is not None
            print(f"  ✓ {loss_name}")
        except Exception as e:
            print(f"  ✗ {loss_name}: {e}")
            raise


def main():
    """Run all tests."""
    test_registry()
    test_vjepa_lp_loss()
    test_mse_loss()
    test_cosine_similarity_loss()
    test_get_loss_fn_from_config()
    test_get_loss_fn_string()
    test_loss_backward()
    test_all_losses()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
