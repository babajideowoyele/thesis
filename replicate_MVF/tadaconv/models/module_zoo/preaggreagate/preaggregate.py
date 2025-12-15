from tadaconv.models.base.base_blocks import PREAGGREGATE_REGISTRY
import torch.nn as nn

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
    
    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (N, T, H, W, C)"
        assert x.size(1) % self.T == 0, f"Input tensor temporal dimension must divisible by {self.T}"

        N, T, H, W, C = x.shape
        x = x.view(N, self.T, T // self.T, H, W, C)
        x = x.mean(dim=2)
        assert x.shape == (N, self.T, H, W, C)
        return x
    