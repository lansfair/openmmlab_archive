from __future__ import annotations

from typing import Any

import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import PixelData
from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample


@TRANSFORMS.register_module()
class PackOlmoEarthSegInputs(BaseTransform):
    """Pack OLMoEarth arrays for MMSeg while keeping temporal metadata."""

    def __init__(
        self,
        meta_keys=(
            "img_paths",
            "seg_map_path",
            "valid_mask_path",
            "ori_shape",
            "img_shape",
            "dataset_name",
            "sample_id",
            "olmoearth_modality",
            "olmoearth_num_timesteps",
            "olmoearth_band_names",
            "present_bands",
            "timestamps",
            "olmoearth_rgb_adapter",
            "olmoearth_raw_img",
            "olmoearth_raw_band_names",
        ),
    ) -> None:
        self.meta_keys = meta_keys

    @staticmethod
    def _image_to_tensor(array: np.ndarray):
        if array.ndim < 3:
            array = np.expand_dims(array, -1)
        chw = np.ascontiguousarray(array.transpose(2, 0, 1))
        return to_tensor(chw).contiguous()

    def transform(self, results: dict[str, Any]) -> dict[str, Any]:
        packed: dict[str, Any] = {
            "inputs": self._image_to_tensor(results["img"])
        }

        sample = SegDataSample()
        gt_seg = results["gt_seg_map"]
        if gt_seg.ndim == 2:
            gt_seg = gt_seg[None, ...]
        sample.gt_sem_seg = PixelData(data=to_tensor(gt_seg.astype(np.int64)))

        if "gt_valid_mask" in results:
            valid = results["gt_valid_mask"]
            if valid.ndim == 2:
                valid = valid[None, ...]
            sample.set_data(
                {
                    "gt_valid_mask": PixelData(
                        data=to_tensor(valid.astype(np.float32))
                    )
                }
            )

        metainfo = {
            key: results[key] for key in self.meta_keys if key in results
        }
        sample.set_metainfo(metainfo)
        packed["data_samples"] = sample
        return packed

@TRANSFORMS.register_module()
class PackDinoSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']  # shape: (H, W, 12)

            # ====================== 在这里修改 ======================
            # 只抽取 BGR 三个通道（你指定的通道：2=蓝,3=绿,4=红）
            # numpy 索引 [:, :, [蓝,绿,红]]
            img = img[:, :, [1, 2, 3]]  # 抽 3 个通道，变成 (H, W, 3)

            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                data = to_tensor(results['gt_seg_map'].astype(np.int64))
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str