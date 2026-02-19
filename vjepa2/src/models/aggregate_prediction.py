from omegaconf import DictConfig
from src.utils.model_registry import MODEL_REGISTRY
from torch import nn

@MODEL_REGISTRY.register("aggregate_prediction")
class AggregatePrediction(nn.Module):
    """Aggregate Prediction"""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.linear1 = nn.Linear(cfg.embedding_dim, cfg.intermediate_dim, bias=True)
        self.linear2 = nn.Linear(cfg.intermediate_dim, cfg.num_classes, bias=True)
        if cfg.method == "mean":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif cfg.method == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unsupported pooling method: {cfg.method}")

    def forward(self, x):
        x = self.pooling(x.transpose(1, 2)).squeeze(2)  # Pool across the sequence dimension
        x = self.linear1(x)
        x = self.linear2(x)
        return x
    

class MVHead(nn.Module):
    """Multi-View Head for separate severity and action predictions"""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        

    def forward(self, x):
        severity_pred = self.severity_head(x)
        action_pred = self.action_head(x)
        return severity_pred, action_pred