import torch
from mmengine.model import BaseModule
from mmseg.models.necks import Feature2Pyramid

from opencd.registry import MODELS


@MODELS.register_module()
class FeatureFusionPyramid(BaseModule):
    """Fuse bi-temporal features, then build a ViT feature pyramid."""

    def __init__(self,
                 policy='abs_diff',
                 embed_dim=768,
                 rescales=(4, 2, 1, 0.5),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        if policy not in ('concat', 'sum', 'diff', 'abs_diff'):
            raise ValueError(f'Unsupported fusion policy: {policy}')
        self.policy = policy
        self.out_indices = out_indices
        self.pyramid = Feature2Pyramid(
            embed_dim=embed_dim, rescales=rescales, norm_cfg=norm_cfg)

    def _fuse(self, x1, x2):
        if self.policy == 'concat':
            return torch.cat([x1, x2], dim=1)
        if self.policy == 'sum':
            return x1 + x2
        if self.policy == 'diff':
            return x2 - x1
        return torch.abs(x1 - x2)

    def forward(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError('Feature lists from both dates must match.')
        fused = tuple(self._fuse(x1[i], x2[i]) for i in self.out_indices)
        return self.pyramid(fused)
