
import os.path as osp
import numpy as np
import warnings
import tifffile
from typing import Optional, Sequence

from mmengine.runner import Runner
from mmengine.visualization import Visualizer
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmengine.hooks import Hook
from mmseg.engine.hooks import SegVisualizationHook

@HOOKS.register_module()
class S2SegVisualizationHook(SegVisualizationHook):
    """Sentinel-2 遥感可视化钩子，关闭文字标签"""
    def __init__(self,
                 draw: bool = True,
                 interval: int = 1,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args=None):
        super().__init__(
            draw=draw,
            interval=interval,
            show=show,
            wait_time=wait_time,
            backend_args=backend_args
        )

    def _s2_to_rgb(self, img_path):
        """S2 13通道转真彩色RGB+百分位拉伸"""
        img = tifffile.imread(img_path)
        rgb = img[..., [3, 2, 1]].copy().astype(np.float32)
        
        for i in range(3):
            p1 = np.percentile(rgb[..., i], 1)
            p99 = np.percentile(rgb[..., i], 99)
            rgb[..., i] = np.clip((rgb[..., i] - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
        
        return rgb.astype(np.uint8)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return
        if runner.iter % self.interval != 0:
            return

        ds = outputs[0]
        rgb = self._s2_to_rgb(ds.img_path)

        meta = runner.val_dataset.metainfo
        self._visualizer.dataset_meta = dict(
            classes=meta.get("classes", []),
            palette=meta.get("palette", [])
        )

        self._visualizer.add_datasample(
            name=f'val_{osp.basename(ds.img_path)}',
            image=rgb,
            data_sample=ds,
            show=self.show,
            wait_time=self.wait_time,
            step=runner.iter,
            with_labels=False  # 关闭类别文字标签
        )

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return

        meta = runner.test_dataloader.dataset.metainfo
        self._visualizer.dataset_meta = dict(
            classes=meta.get("classes", []),
            palette=meta.get("palette", [])
        )

        for ds in outputs:
            self._test_index += 1
            rgb = self._s2_to_rgb(ds.img_path)
            
            self._visualizer.add_datasample(
                name=f'test_{osp.basename(ds.img_path)}',
                image=rgb,
                data_sample=ds,
                show=self.show,
                wait_time=self.wait_time,
                step=self._test_index,
                with_labels=False  # 关闭类别文字标签
            )