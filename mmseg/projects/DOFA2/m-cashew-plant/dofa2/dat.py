import os
from pathlib import Path
from typing import List
from typing import Dict

import cv2
import numpy as np
import rasterio
from tqdm import tqdm
from mmcv.transforms import BaseTransform
from mmseg.datasets import BaseSegDataset
from mmseg.registry import TRANSFORMS
from mmseg.registry import DATASETS

from . import utils


def extract_from_geobench(srcdir: Path, dstdir: Path, bands=None, ext='.tif', dtype=None):
    if '.png' == ext:
        if not bands or len(bands) > 3:
            raise ValueError()
        if dtype not in (np.uint8, np.uint16):
            raise TypeError()
    os.environ['GEO_BENCH_DIR'] = srcdir.parent.parent.as_posix()
    import geobench
    if not os.path.isdir(srcdir):
        raise ValueError(f"srcdir '{srcdir}' is not existed or not a directory.")
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    elif not os.path.isdir(dstdir):
        raise ValueError(f"dstdir '{dstdir}' is not a directory.")
    task = geobench.load_task_specs(srcdir)
    if bands is None:
        bands = [band.name for band in task.bands_info]
    filenames, categories = {}, {}
    for partition in ('train', 'valid', 'test'):
        img_dir_path = dstdir/'img_dir'/partition
        ann_dir_path = dstdir/'ann_dir'/partition
        os.makedirs(img_dir_path, exist_ok=True)
        os.makedirs(ann_dir_path, exist_ok=True)
        dataset = task.get_dataset(split=partition, band_names=bands)
        # bands_vrange = []
        # for band_name in dataset.band_names:
        #     band_stats = dataset.band_stats[band_name]
        #     bands_vrange.append([band_stats.max, band_stats.min])
        # bands_vrange = np.asarray(bands_vrange)
        partition_filenames, partition_categories = [], set()
        for sample in tqdm(dataset, desc=f'{dstdir.name}({partition})'):
            img = sample.pack_to_3d(band_names=bands)[0]
            if dtype is not None and img.dtype is not dtype:
                # img = (img-bands_vrange[:, 1]) / (bands_vrange[:, 0]-bands_vrange[:, 1])
                img = (img-img.min()) / (img.max()-img.min() or 1)
                img *= utils.numpy_dtype_maximum[dtype]
                img = img.astype(dtype)
            img_path = img_dir_path/f'{sample.sample_name}{ext}'
            if '.png' == ext:
                cv2.imwrite(img_path, img)
            elif '.tif' == ext:
                h, w, c = img.shape
                with rasterio.open(img_path, 'w', driver="GTiff", height=h, width=w, count=c, dtype=img.dtype) as f:
                    f.write(np.moveaxis(img, 2, 0))
            else:
                raise TypeError()
            ann = sample.label.data.astype(np.uint8)
            partition_categories.update(np.unique(ann).tolist())
            ann = ann.reshape(*ann.shape, 1).repeat(3, axis=-1)
            ann_path = ann_dir_path/f'{sample.sample_name}.png'
            cv2.imwrite(ann_path, ann)
            partition_filenames.append(sample.sample_name)
        filenames[partition] = partition_filenames
        categories[partition] = list(partition_categories)
    return filenames, categories


geobench_dataset = None


@TRANSFORMS.register_module(force=True)
class LoadSingleRSImgFromHDF5(BaseTransform):
    def __init__(self, bands):
        self.bands = bands

    def transform(self, results: Dict) -> Dict:
        if geobench_dataset:
            sample = geobench_dataset.get_sample(results['sample_name'])
            img = sample.pack_to_3d(band_names=self.bands)[0]
        else:
            img = cv2.imread(results['img_path'])
            if img is None:
                raise
            img = img[..., [2, 1, 0]]
        results['img'] = img
        results['img_shape'] = img.shape[: 2]
        results['ori_shape'] = img.shape[: 2]
        return results
    

@TRANSFORMS.register_module(force=True)
class LoadSingleRSAnnFromHDF5(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        if not geobench_dataset:
            return results
        sample = geobench_dataset.get_sample(results['sample_name'])
        gt_semantic_seg = sample.label.data.astype('uint8')
        if results['reduce_zero_label']:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')
        return results


class GEOBenchSegDataset(BaseSegDataset):
    def __init__(self, data_root: str, split: str = 'train', **kwargs) -> None:
        data_root = Path(data_root)
        os.environ['GEO_BENCH_DIR'] = data_root.parent.parent.as_posix()
        import geobench
        task = geobench.load_task_specs(data_root)
        global geobench_dataset
        geobench_dataset = task.get_dataset(split=split, band_names=None)
        super().__init__(data_root=data_root.as_posix(), **kwargs)

    def load_data_list(self) -> List[dict]:
        data_list = []
        sample_name_list = geobench_dataset.active_partition.partition_dict[geobench_dataset.split]
        for sample_name in sample_name_list:
            data_info = dict(
                sample_name=sample_name,
                label_map=self.label_map,
                reduce_zero_label=self.reduce_zero_label,
                seg_fields = []
            )
            data_list.append(data_info)
        return data_list
    

@DATASETS.register_module(force=True)
class CashewPlantSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0], 
            [255, 255, 0], 
            [0, 0, 255],
            [159, 129, 183], 
            [0, 255, 0], 
            [255, 195, 128]
        ]
    }


@DATASETS.register_module(force=True)
class ChesapeakeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0], 
            [255, 255, 0], 
            [0, 0, 255],
            [159, 129, 183], 
            [0, 255, 0], 
            [255, 195, 128]
        ]
    }


@DATASETS.register_module(force=True)
class NeonTreeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class NZCattleSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class Pv4gerSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1'
        ],
        'palette': [
            [255, 255, 255], 
            [255, 0, 0]
        ]
    }


@DATASETS.register_module(force=True)
class SACropTypeSegDataset(GEOBenchSegDataset):
    METAINFO = {
        'classes': [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9'
        ],
        'palette': [
            [0, 0, 0], 
            [128, 0, 0], 
            [0, 128, 0], 
            [128, 128, 0],
            [0, 0, 128], 
            [128, 0, 128], 
            [0, 128, 128], 
            [128, 128, 128],
            [64, 0, 0], 
            [192, 0, 0]
        ]
    }
