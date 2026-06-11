from __future__ import annotations

import torch
from mmengine.model import BaseModule
from mmseg.models.necks import MultiLevelNeck
from opencd.registry import MODELS


@MODELS.register_module()
class DINOv3FeatureFusionPyramid(BaseModule):
    """Fuse bi-temporal DINOv3 features and build a feature pyramid."""

    def __init__(
        self,
        policy: str = "abs_diff",
        embed_dim: int = 1024,
        out_channels: int | None = None,
        scales=(4, 2, 1, 0.5),
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        if policy not in ("concat", "sum", "diff", "abs_diff"):
            raise ValueError(f"Unsupported fusion policy: {policy}")
        self.policy = policy
        fused_dim = embed_dim * 2 if policy == "concat" else embed_dim
        out_channels = out_channels or fused_dim
        self.pyramid = MultiLevelNeck(
            in_channels=[fused_dim] * num_inputs,
            out_channels=out_channels,
            scales=list(scales),
            norm_cfg=norm_cfg,
        )

    def _fuse(self, x1, x2):
        if self.policy == "concat":
            return torch.cat([x1, x2], dim=1)
        if self.policy == "sum":
            return x1 + x2
        if self.policy == "diff":
            return x2 - x1
        return torch.abs(x1 - x2)

    def forward(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("Feature lists from both dates must match.")
        fused = tuple(self._fuse(feat1, feat2) for feat1, feat2 in zip(x1, x2))
        return self.pyramid(fused)
