from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
try:
    from osgeo import gdal
except ImportError:
    gdal = None


def _load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "Reading GeoTIFF inputs requires rasterio."
            ) from exc
        with rasterio.open(path) as src:
            array = src.read()
        if array.shape[0] == 1:
            return array[0]
        return array
    raise ValueError(f"Only GeoTIFF arrays are supported, got: {path}")


def _tchw_to_hw_flat(image: np.ndarray) -> np.ndarray:
    if image.ndim != 4:
        raise ValueError(f"Expected stacked GeoTIFF image as TCHW, got {image.shape}")
    t, c, h, w = image.shape
    return image.transpose(2, 3, 1, 0).reshape(h, w, c * t)


def _load_multitif(paths: list[str | Path]) -> np.ndarray:
    images = []
    for path in paths:
        image = _load_array(path)
        if image.ndim == 2:
            image = image[None, ...]
        if image.ndim != 3:
            raise ValueError(
                "Each path in img_paths must load to CHW or HW GeoTIFF data, "
                f"got shape {image.shape} from {path}"
            )
        images.append(image)
    shape_set = {image.shape for image in images}
    if len(shape_set) != 1:
        raise ValueError(
            "All img_paths in one sample must have the same CHW shape, "
            f"got {sorted(shape_set)}"
        )
    return np.stack(images, axis=0)


@TRANSFORMS.register_module()
class LoadOlmoEarthArrays(BaseTransform):
    """Load OLMoEarth GeoTIFF images, labels, optional masks and timestamps."""

    def __init__(
        self,
        ignore_index: int = 255,
        source_ignore_values: tuple[int, ...] = (-1,),
        reduce_zero_label: bool = False,
    ) -> None:
        self.ignore_index = ignore_index
        self.source_ignore_values = source_ignore_values
        self.reduce_zero_label = reduce_zero_label

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        results["seg_fields"] = ["gt_seg_map"]
        if "img_paths" not in results:
            raise KeyError(
                "Manifest samples must provide 'img_paths' with one GeoTIFF "
                "per timestep."
            )
        image = _load_multitif(results["img_paths"]).astype(
            np.float32,
            copy=False,
        )
        results["img"] = _tchw_to_hw_flat(image)
        results["img_shape"] = results["img"].shape[:2]
        results["ori_shape"] = results["img"].shape[:2]

        label = _load_array(results["seg_map_path"]).squeeze().astype(np.int64)
        if self.reduce_zero_label:
            label = label.copy()
            label[label == 0] = self.ignore_index
            label = label - 1
            label[label == self.ignore_index - 1] = self.ignore_index
        if self.source_ignore_values:
            label = label.copy()
            for value in self.source_ignore_values:
                label[label == value] = self.ignore_index
        results["gt_seg_map"] = label

        valid_mask_path = results.get("valid_mask_path")
        if valid_mask_path is not None:
            valid = _load_array(valid_mask_path).squeeze().astype(np.float32)
            results["gt_valid_mask"] = valid
            results["seg_fields"].append("gt_valid_mask")

        if "timestamps" in results:
            results["timestamps"] = np.asarray(
                results["timestamps"],
                dtype=np.int64,
            )

        return results


@TRANSFORMS.register_module()
class LoadSinglePNGImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename = results['img_path']
        img = np.array(Image.open(filename))

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
    

@TRANSFORMS.register_module()
class LoadLocalPtsdamAnnotations(BaseTransform):
    """Load and remap original DFC2020 labels to 8 valid classes."""

    cls_mapping = {
        0: 5,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5
    }

    def __init__(self):
        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results):
        ds = gdal.Open(results['seg_map_path'])
        if ds is None:
            raise FileNotFoundError(
                f'Unable to open file: {results["seg_map_path"]}')
        seg_map = ds.ReadAsArray()
        remapped = np.full(seg_map.shape, 5, dtype=np.uint8)
        for old_label, new_label in self.cls_mapping.items():
            remapped[seg_map == old_label] = new_label
        if results.get('label_map', None) is not None:
            remapped_copy = remapped.copy()
            for old_id, new_id in results['label_map'].items():
                remapped[remapped_copy == old_id] = new_id
        results['gt_seg_map'] = remapped
        results.setdefault('seg_fields', []).append('gt_seg_map')
        return results
