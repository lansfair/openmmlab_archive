import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SelectBands(BaseTransform):
    """从多波段图像中保留指定波段，13通道 → 12通道"""
    def __init__(self, bands):
        self.bands = bands

    def transform(self, results):
        img = results["img"]  # shape: (H, W, 13)
        img = img[..., self.bands]  # 只保留指定波段 → (H, W, 12)
        results["img"] = img
        return results