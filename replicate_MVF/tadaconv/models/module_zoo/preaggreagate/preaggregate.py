from httpx import patch
import torch
from tadaconv.models.base.base_blocks import PREAGGREGATE_REGISTRY
import torch.nn as nn

from tadaconv.models.base.transformer import Transformer

@PREAGGREGATE_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


@PREAGGREGATE_REGISTRY.register()
class TemporalPooling(nn.Module):
    def __init__(self, cfg):
        super(TemporalPooling, self).__init__()
        self.T = cfg.DATA.NUM_INPUT_FRAMES
        self.F = cfg.DATA.TAKE_NUM_FRAMES
    
    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (N, T, H, W, C)"
        assert x.size(1) % self.T == 0, f"Input tensor temporal dimension must divisible by {self.T} but has size {x.size()}"

        N, T, H, W, C = x.shape
        divide = self.F // T
        x = x.view(N, self.T // divide, self.F // self.T, H, W, C)
        x = x.max(dim=2).values
        assert x.shape == (N, self.T // divide, H, W, C)
        return x
    
@PREAGGREGATE_REGISTRY.register()
class AttentionPooling(nn.Module):
    def __init__(self, cfg,):
        super(AttentionPooling, self).__init__()
        self.width = cfg.VIDEO.BACKBONE.NUM_FEATURES
        self.num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS
        dim_head = self.width // self.num_heads
        self.scale = dim_head ** -0.5

        input_resolution    = cfg.VIDEO.BACKBONE.INPUT_RES
        patch_size          = cfg.VIDEO.BACKBONE.PATCH_SIZE

        self.patch_number = (input_resolution // patch_size) ** 2
        self.carry_patches = self.patch_number * 8

        self.MLP         = nn.Linear(self.width, self.carry_patches)
        self.T = cfg.DATA.NUM_INPUT_FRAMES
        self.F = cfg.DATA.TAKE_NUM_FRAMES

    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (N, T*V, H, W, C)"
        N, T, H, W, C = x.shape

        assert C == self.width, f"Input tensor channel dimension must be {self.width} but has size {x.size()}"
        x = x.reshape(N, -1, C)
        x = torch.softmax(self.MLP(x), dim=1).transpose(-1, -2) @ x

        return x.reshape(N, -1, H, W, C)

@PREAGGREGATE_REGISTRY.register()
class TransformerPooling(nn.Module):
    def __init__(self, cfg):
        super(TransformerPooling, self).__init__()
        self.attention = Transformer(
            ...
        )
        self.T = cfg.DATA.NUM_INPUT_FRAMES
        self.F = cfg.DATA.TAKE_NUM_FRAMES

    def forward(self, x):
        ...
        
        
        
        

        