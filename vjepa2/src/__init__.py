from typing import Type
from src.models import aggregate_prediction, attentive_pooler
from src.utils.model_registry import MODEL_REGISTRY


def get_model_class(name: str) -> Type:
    """Helper function to retrieve a model class from the registry."""
    return MODEL_REGISTRY.get(name)