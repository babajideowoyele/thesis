from multiprocessing import pool
import torch
from tadaconv.models.base.base_blocks import HEAD_REGISTRY, BaseHead, BaseHeadx2
from tadaconv.models.utils.init_helper import _init_transformer_weights
from tadaconv.utils import logging

logger = logging.get_logger(__name__)


@HEAD_REGISTRY.register()
class FOULHead(BaseHeadx2):
    def __init__(self, cfg):
        super(FOULHead, self).__init__(cfg)
        self.apply(_init_transformer_weights)

    def _construct_head(self, dim, num_classes, dropout_rate, activation_func):
        self.emb = torch.nn.Embedding(2, dim)

        self.global_avg_pool = torch.nn.AdaptiveAvgPool3d(1)

        if dropout_rate > 0.0:
            self.dropout = torch.nn.Dropout(dropout_rate)

        self.linear1 = torch.nn.Linear(dim, num_classes[0], bias=True)
        self.linear2 = torch.nn.Linear(dim, num_classes[1], bias=True)

        if activation_func == "softmax":
            self.activation = torch.nn.Softmax(dim=-1)
        elif activation_func == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(activation_func)
            )

    def forward(self, x, mask):
        """
        Returns:
            x (Tensor): classification predictions.
            logits (Tensor): global average pooled features.
        """
        mask_bool = mask.bool()
        mask_filtered = mask[mask_bool]
        assert len(x.shape) == 3, "Input tensor must be 3D"

        x_reshaped = x[mask_bool]
        #Not applicable: x_pooled = self.global_avg_pool(x_reshaped)  # (B*V, C, 1, 1, 1)

        emb = self.emb(mask_filtered - 1)

        assert (
            emb.shape == x_reshaped.shape
        ), f"emb shape {emb.shape} and out shape {x_reshaped.shape} do not match"

        out = x_reshaped + emb

        B, V = mask.shape
        T = out.shape[1:]
        out_mask = torch.full((B, V) + T, float('-inf'), dtype=out.dtype, device=out.device)
        out_mask[mask_bool] = out
        out, idx = torch.max(out_mask, dim=1)


        if hasattr(self, "dropout"):
            out1 = self.dropout(out)
        else:
            out1 = out
        out2 = out1

        

        out1 = self.linear1(out1)
        out2 = self.linear2(out2)


        if not self.training:
            out1 = self.activation(out1)
            out2 = self.activation(out2)
        
        out1 = out1.view(out1.shape[0], -1)
        out2 = out2.view(out2.shape[0], -1)

        return {"type": out1, "severity": out2}, out.view(x.shape[0], -1)
