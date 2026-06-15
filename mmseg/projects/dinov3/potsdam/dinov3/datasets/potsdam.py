from pathlib import Path
import os
from mmseg.registry import DATASETS
from mmseg.datasets import PotsdamDataset

@DATASETS.register_module()
class LocalPotsdamDataset(PotsdamDataset):
    """Potsdam dataset that adds OLMoEarth project metadata.

    This keeps Potsdam on the normal OpenMMLab image-dataset path:
    ``LoadImageFromFile`` reads RGB tiles and ``LoadAnnotations`` reads label
    PNGs. The OLMoEarth-specific conversion happens later in the transform
    pipeline via ``RGBToOlmoEarthS2``.
    """

    RVSA_CLASSES = (
        "impervious_surface",
        "building",
        "low_vegetation",
        "tree",
        "car",
    )
    RVSA_PALETTE = [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 255],
        [0, 255, 0],
        [255, 255, 0],
    ]
    OFFICIAL_TO_RVSA_LABEL_MAP = {
        0: 5,
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
    }

    def __init__(
        self,
        **kwargs,
    ) -> None:
        kwargs.setdefault("reduce_zero_label", False)
        kwargs.setdefault("ignore_index", 5)
        metainfo = dict(kwargs.pop("metainfo", {}) or {})
        metainfo.setdefault("classes", self.RVSA_CLASSES)
        metainfo.setdefault("palette", self.RVSA_PALETTE)
        kwargs["metainfo"] = metainfo
        super().__init__(**kwargs)

    def load_data_list(self) -> list[dict]:
        data_list = super().load_data_list()
        for item in data_list:
            item["label_map"] = dict(self.OFFICIAL_TO_RVSA_LABEL_MAP)
            item["reduce_zero_label"] = False
        return data_list