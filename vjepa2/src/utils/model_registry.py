from typing import Callable, Dict, Optional, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class ModelRegistry:
    """Registry for model classes."""
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
    
    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a model class.
        
        Args:
            name: Optional name for the model. If not provided, uses the class/function name.
        
        Returns:
            The decorator function.
        """
        def decorator(fn_or_cls):
            model_name = name or fn_or_cls.__name__
            self._registry[model_name] = fn_or_cls
            return fn_or_cls
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Retrieve a registered model class.
        
        Args:
            name: The name of the model to retrieve.
            
        Returns:
            The model class.
            
        Raises:
            ValueError: If the model is not registered.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return self._registry[name]
    
    def list_models(self) -> list:
        """Return a list of all registered model names."""
        return list(self._registry.keys())


# Global registry instance
MODEL_REGISTRY = ModelRegistry()


