import os


DATA_ROOT = os.path.join(
    os.environ.get('MM_ARCHIVE_DATA_HOME'), 
    'geo-bench-1.0',
    'segmentation_v1.0',
    'm-cashew-plant'
)

DATASET_TYPE = 'CashewPlantSegDataset'

BANDS_NAME = ['red', 'green', 'blue']


train_pipeline = [
    dict(type='LoadSingleRSImgFromHDF5', bands=BANDS_NAME),
    dict(type='LoadSingleRSAnnFromHDF5'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='train',
        pipeline=train_pipeline,
        reduce_zero_label=False
    ),
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True)
)

test_pipeline = [
    dict(type='LoadSingleRSImgFromHDF5', bands=BANDS_NAME),
    dict(type='LoadSingleRSAnnFromHDF5'),
    dict(type='PackSegInputs')
]
val_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='valid',
        test_mode=True,
        pipeline=test_pipeline,
        reduce_zero_label=False
    ),
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='test',
        test_mode=True,
        pipeline=test_pipeline,
        reduce_zero_label=False
    ),
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False)
)
