import os
from datetime import date, datetime

import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import TRANSFORMS

try:
    from osgeo import gdal, osr
except ImportError:
    gdal = None
    osr = None




@TRANSFORMS.register_module()
class NormalizeMultibandImage(BaseTransform):
    """Normalize a multi-band image before geometric augmentation."""

    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)

    def transform(self, results):
        results['img'] = (results['img'].astype(np.float32) -
                          self.mean) / self.std
        return results


@TRANSFORMS.register_module()
class LoadCoBenchSegAnnotations(BaseTransform):
    """Load segmentation labels with optional class-id remapping."""

    def __init__(self, label_mapping=None, default_value=None):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.label_mapping = label_mapping
        self.default_value = default_value

    def transform(self, results):
        ds = gdal.Open(results['seg_map_path'])
        if ds is None:
            raise FileNotFoundError(
                f'Unable to open file: {results["seg_map_path"]}')
        seg_map = ds.ReadAsArray()
        if seg_map.ndim == 3:
            seg_map = seg_map[0]

        if self.label_mapping is not None:
            if self.default_value is None:
                remapped = seg_map.copy()
            else:
                remapped = np.full(
                    seg_map.shape, self.default_value, dtype=np.int64)
            for old_label, new_label in self.label_mapping.items():
                remapped[seg_map == int(old_label)] = int(new_label)
            seg_map = remapped

        if results.get('label_map', None) is not None:
            remapped = seg_map.copy()
            for old_id, new_id in results['label_map'].items():
                seg_map[remapped == old_id] = new_id

        results['gt_seg_map'] = seg_map.astype(np.uint8)
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results



