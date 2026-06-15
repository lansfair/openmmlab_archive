import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from osgeo import gdal


@TRANSFORMS.register_module()
class Load12senImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file, auto remove B10 band.

    Required Keys:
    - img_path

    Modified Keys:
    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. Defaults to True.
    """

    # DFC2020-S2 原始13波段顺序，用于定位 B10
    ALL_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
    # 需要保留的波段：剔除 B10
    KEEP_BANDS = tuple(band for band in ALL_BANDS if band != "B10")
    # 计算保留波段对应的数组索引
    KEEP_INDICES = [idx for idx, band in enumerate(ALL_BANDS) if band != "B10"]

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: dict) -> dict:
        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        
        # GDAL 读取: (C, H, W)
        img_cwh = ds.ReadAsArray()
        
        # 核心：剔除 B10 波段，仅保留指定通道
        img_cwh = img_cwh[self.KEEP_INDICES, :, :]
        
        # 维度转换 CWH -> HWC
        img = np.einsum('ijk->jki', img_cwh)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str