from typing import Type
from src.models import aggregate_prediction, attentive_pooler
from src.models.vision_transformer import *
from src.VARS.dataset import get_dataset
from src.datasets.mvfoul import Mvfoul
from src.utils.model_registry import MODEL_REGISTRY
from src.utils.dataset_registry import DATASET_REGISTRY


def get_model_class(name: str) -> Type:
    """Helper function to retrieve a model class from the registry."""
    return MODEL_REGISTRY.get(name)

def get_dataset_class(name: str) -> Type:
    """Helper function to retrieve a dataset class from the registry."""
    return DATASET_REGISTRY.get(name)