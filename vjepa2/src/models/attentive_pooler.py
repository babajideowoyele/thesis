# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
from omegaconf import DictConfig
import torch
import torch.nn as nn

from src.models.utils.modules import Block, CrossAttention, CrossAttentionBlock
from src.utils.tensors import trunc_normal_
from src.utils.model_registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("attentive_pooler")
class AttentivePooler(nn.Module):
    """Attentive Pooler"""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        num_queries=cfg.num_queries
        embed_dim=cfg.embedding_dim
        num_heads=cfg.num_heads
        mlp_ratio=cfg.mlp_ratio
        depth=cfg.depth
        norm_layer=nn.LayerNorm
        init_std=cfg.init_std
        qkv_bias=cfg.qkv_bias
        complete_block=cfg.complete_block
        use_activation_checkpointing=cfg.use_activation_checkpointing
        attn_drop=cfg.attn_drop
        proj_drop=cfg.proj_drop
        attention_mechanism=cfg.att_pos
        self.use_activation_checkpointing = use_activation_checkpointing
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))

        self.complete_block = complete_block
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
            )
        else:
            self.cross_attention_block = CrossAttention(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=False,
                        norm_layer=norm_layer,
                        drop=proj_drop,
                        attn_drop=attn_drop,
                        drop_path=proj_drop,
                        use_rope=attention_mechanism == "rope",
                    )
                    for i in range(depth - 1)
                ]
            )

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        if self.complete_block:
            rescale(self.cross_attention_block.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                if self.use_activation_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(blk, x, False, None, use_reentrant=False)
                else:
                    x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        q = self.cross_attention_block(q, x)
        return q

@MODEL_REGISTRY.register("attentive_classifier")
class AttentiveClassifier(nn.Module):
    """Attentive Classifier"""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            cfg,
        )
        self.aggregate_logits = nn.AdaptiveMaxPool1d(1)
        embedding_dim = cfg.embedding_dim
        self.linear = nn.Linear(embedding_dim, cfg.num_classes, bias=True)

    def forward(self, x, v: int = 1):
        x = self.pooler(x)
        if v > 1:
            assert x.shape[0] % v == 0, f"Batch size {x.shape[0]} must be divisible by number of views {v}"
            x = x.view(x.size(0) // v, x.size(1)*v, -1)
        if len(x.shape) == 3:
            x = self.aggregate_logits(x.transpose(2,1))
            if len(x.shape) == 3:
                x = x.squeeze(2)
        x = self.linear(x)
        return x
