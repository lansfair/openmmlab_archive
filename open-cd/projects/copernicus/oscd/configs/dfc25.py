# dataset settings for official OSCD Sentinel-2 band-wise GeoTIFFs
dataset_type = 'DFCDataset'
data_root = '/mnt/ht2-nas2/EO_test/dataset/DFC2025 BRIGHT/dfc25_track2_trainval'
crop_size = (256, 256)

s2_band_stats = dict(
    mean=[
        1353.7, 1117.2, 1041.8, 946.5, 1199.1, 2003.0, 2374.0, 2301.2,
        2599.7, 732.1, 12.1, 1820.6, 1118.2
    ],
    std=[
        897.3, 736.0, 684.8, 620.0, 791.9, 1341.3, 1595.4, 1545.5,
        1750.1, 475.1, 98.3, 1216.5, 736.7
    ],
    # mean=[
    #     0,0,0,0
    # ],
    # std=[
    #     1,1,1,1
    # ],
)

train_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFileDFC'),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(
        type='MultiImgNormalizeMultibandImageDFC',
        mean=s2_band_stats['mean']*2,
        std=s2_band_stats['std']*2),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFileDFC'),
    dict(
        type='MultiImgNormalizeMultibandImageDFC',
        mean=s2_band_stats['mean']*2,
        std=s2_band_stats['std']*2),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img1_dir='train/pre-event',
            img2_dir='train/post-event',
            ann_dir='train/True_2_False_1',
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img1_dir='val/pre-event',
        img2_dir='val/post-event',
        ann_dir='val/True_2_False_1',
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img1_dir='val/pre-event',
        img2_dir='val/post-event',
        ann_dir='val/target',
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator
