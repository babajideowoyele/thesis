from tadaconv.models.base.base_blocks import PREAGGREGATE_REGISTRY
import torch.nn as nn

@PREAGGREGATE_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self, cfg):
        pass

    def forward(self, x):
        return x


@PREAGGREGATE_REGISTRY.register()
class TemporalPooling(nn.Module):
    def __init__(self, cfg):
        super(TemporalPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cfg = cfg 

    def forward(self, x):
        # x: (B, N, C) where N = T*H*W
        B, N, C = x.shape
        T = self.cfg.DATA.NUM_INPUT_FRAMES // self.cfg.VIDEO.BACKBONE.TUBLET_STRIDE
        H_W = N // T
        x = x.reshape(B, T, H_W, C).permute(0, 3, 2, 1)  # (B, C, H*W, T)
        x = self.pool(x).squeeze(-1)  # (B, C, H*W)
        x = x.permute(0, 2, 1)  # (B, H*W, C)
        return x