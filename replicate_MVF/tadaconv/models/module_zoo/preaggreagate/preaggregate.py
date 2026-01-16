import torch
from tadaconv.models.base.base_blocks import PREAGGREGATE_REGISTRY
import torch.nn as nn


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


class AttentionBased(nn.Module):
    def __init__(self):
        super().__init__()

@PREAGGREGATE_REGISTRY.register()
class AttentionPooling(AttentionBased):
    def __init__(self, cfg,):
        super(AttentionPooling, self).__init__()
        self.width = cfg.VIDEO.BACKBONE.NUM_FEATURES
        self.num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS
        dim_head = self.width // self.num_heads
        self.scale = dim_head ** -0.5
        frame_equivalence = 8

        scale = self.width ** -0.5

        input_resolution    = cfg.VIDEO.BACKBONE.INPUT_RES
        patch_size          = cfg.VIDEO.BACKBONE.PATCH_SIZE

        self.patch_number = (input_resolution // patch_size) ** 2
        self.carry_patches = self.patch_number * frame_equivalence

        self.MLP         = nn.Linear(self.width, self.carry_patches)
        self.T = cfg.DATA.NUM_INPUT_FRAMES
        self.F = cfg.DATA.TAKE_NUM_FRAMES * cfg.DATA.NUM_VIEWS // cfg.VIDEO.BACKBONE.TUBLET_STRIDE
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.F * self.patch_number, self.width)))

    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (N, T*V, H, W, C)"
        N, T, H, W, C = x.shape
        residual = x.mean(dim=1)

        assert C == self.width, f"Input tensor channel dimension must be {self.width} but has size {x.size()}"
        x = x.reshape(N, -1, C)
        x = x + self.positional_embedding.to(x.dtype)
        x = torch.softmax(self.MLP(x), dim=1).transpose(-1, -2) @ x

        return x.reshape(N, -1, H, W, C) + residual.unsqueeze(1)
        
@PREAGGREGATE_REGISTRY.register()
class TransformerPooling(AttentionBased):
    def __init__(self, cfg):
        super(TransformerPooling, self).__init__()
        self.width = cfg.VIDEO.BACKBONE.NUM_FEATURES
        self.num_heads = cfg.VIDEO.BACKBONE.NUM_HEADS
        dim_head = self.width // self.num_heads
        self.scale = dim_head ** -0.5
        frame_equivalance = 8

        input_resolution    = cfg.VIDEO.BACKBONE.INPUT_RES
        patch_size          = cfg.VIDEO.BACKBONE.PATCH_SIZE

        self.patch_number = (input_resolution // patch_size) ** 2
        self.carry_patches = self.patch_number * frame_equivalance

        self.to_qkv = nn.Linear(self.width, self.width * 3, bias=False)
        self.to_out = nn.Linear(self.width, self.width)

        self.T = cfg.DATA.NUM_INPUT_FRAMES
        self.F = cfg.DATA.TAKE_NUM_FRAMES

    def forward(self, x):
        assert x.dim() == 5, "Input tensor must be 5D (N, T*V, H, W, C)"
        N, T, H, W, C = x.shape
        residual = x.mean(dim=1)

        assert C == self.width, f"Input tensor channel dimension must be {self.width} but has size {x.size()}"
        x = x.reshape(N, -1, C)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(N, -1, self.num_heads, C // self.num_heads).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(N, -1, C)
        out = self.to_out(out)

        return out.reshape(N, -1, H, W, C) + residual.unsqueeze(1) 
        

        