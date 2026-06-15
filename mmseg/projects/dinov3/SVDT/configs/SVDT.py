import os
data_root = os.path.join(os.environ.get('MM_ARCHIVE_DATA_HOME'), 'SVDT')
dataset_type = 'LocalSVDTDataset'

ignore_index = 255
num_classes = 2
crop_size = 256
patch_size = 16

train_pipeline = [
    dict(type="LoadSinglePNGImageFromFile", to_float32=True),
    dict(type="LoadLocalSVDTAnnotations"),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(
        type="RandomRotate",
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadSinglePNGImageFromFile", to_float32=True),
    dict(type="LoadLocalSVDTAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
        ann_file='train.txt',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
        ann_file='valid.txt',
        ignore_index=ignore_index,
        reduce_zero_label=False,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    use_valid_mask=False,
)
test_evaluator = val_evaluator