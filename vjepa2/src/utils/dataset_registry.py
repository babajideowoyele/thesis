from typing import Callable, Dict, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from utils.model_registry import ModelRegistry
from utils.model_registry import ModelRegistry


class DatasetRegistry:
    """Registry for dataset classes."""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a dataset class.
        
        Args:
            name: Optional name for the dataset. If not provided, uses the class/function name.
        
        Returns:
            The decorator function.
        """
        def decorator(fn_or_cls):
            dataset_name = name or fn_or_cls.__name__
            self._registry[dataset_name] = fn_or_cls
            return fn_or_cls
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Retrieve a registered dataset class.
        
        Args:
            name: The name of the dataset to retrieve.
            
        Returns:
            The dataset class.
            
        Raises:
            ValueError: If the dataset is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Dataset '{name}' not found. Available datasets: {available}"
            )
        return self._registry[name]
    
    def list_datasets(self) -> list:
        """Return a list of all registered dataset names."""
        return list(self._registry.keys())


# Global registry instance
DATASET_REGISTRY = DatasetRegistry()


