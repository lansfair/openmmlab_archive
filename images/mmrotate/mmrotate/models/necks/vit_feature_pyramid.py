import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule

from mmrotate.registry import MODELS


@MODELS.register_module()
class VitFeaturePyramid(BaseModule):
    """Build multi-scale feature pyramid from ViT's single-resolution outputs.

    ViT backbone outputs all feature maps at the same spatial resolution
    (img_size // patch_size = stride 16). Unlike the fusion-based approach,
    this neck maps each ViT layer to a different output stride directly:

        layer 2  (shallow)  →  upsample 4×  →  stride 4   (small objects)
        layer 5              →  upsample 2×  →  stride 8   (medium-small)
        layer 8              →  keep         →  stride 16  (medium)
        layer 11 (deep)      →  downsample    →  stride 32  (large objects)

    No feature fusion between levels — each ViT layer independently
    contributes to its assigned output scale.

    Args:
        in_channels (int): Number of input channels from ViT backbone.
            Default 768 (vit_base embed_dim).
        out_channels (int): Number of output channels. Default 256.
        num_inputs (int): Number of input features from backbone. Default 4.
        num_outs (int): Number of output pyramid levels. Default 4.
    """

    def __init__(self,
                 in_channels=768,
                 out_channels=256,
                 num_inputs=4,
                 num_outs=4):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outs = num_outs

        for i in range(num_inputs):
            self.add_module(
                f'lateral_{i}',
                nn.Conv2d(in_channels, out_channels, kernel_size=1))

        for i in range(num_outs):
            self.add_module(
                f'out_conv_{i}',
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.downsample = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        assert len(features) >= self.num_inputs, \
            f'Expected {self.num_inputs} input features, got {len(features)}'
        features = features[:self.num_inputs]

        laterals = [getattr(self, f'lateral_{i}')(features[i])
                    for i in range(self.num_inputs)]

        stride4 = F.interpolate(
            laterals[0], scale_factor=4, mode='bilinear', align_corners=False)
        stride4 = self.out_conv_0(stride4)

        stride8 = F.interpolate(
            laterals[1], scale_factor=2, mode='bilinear', align_corners=False)
        stride8 = self.out_conv_1(stride8)

        stride16 = self.out_conv_2(laterals[2])

        stride32 = self.downsample(laterals[3])
        stride32 = self.out_conv_3(stride32)

        for i, f in enumerate([stride4, stride8, stride16, stride32]):
            if torch.isnan(f).any():
                print(f'[VitFeaturePyramid] NaN in stride {(4,8,16,32)[i]}: '
                      f'mean={f[~torch.isnan(f)].mean():.2f}, '
                      f'num_nan={torch.isnan(f).sum()}')
                raise RuntimeError(f'NaN detected in neck output stride {(4,8,16,32)[i]}')

        return stride4, stride8, stride16, stride32
