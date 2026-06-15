import os

DATA_ROOT = os.path.join(os.environ.get('MM_ARCHIVE_DATA_HOME'), 'geo-bench-1.0/segmentation_v1.0/m-cashew-plant')

DATASET_TYPE = 'CashewPlantSegDataset'

BANDS_NAME = [
    '01 - Coastal aerosol',
    '02 - Blue',
    '03 - Green',
    '04 - Red',
    '05 - Vegetation Red Edge',
    '06 - Vegetation Red Edge',
    '07 - Vegetation Red Edge',
    '08 - NIR',
    '08A - Vegetation Red Edge',
    '09 - Water vapour',
    '11 - SWIR',
    '12 - SWIR',
    'Cloud Probability'
]

BANDS_MEAN = [
    520.1185302734375,   # 01 - Coastal aerosol
    634.7583618164062,   # 02 - Blue
    892.4611816406250,   # 03 - Green
    880.7075805664062,   # 04 - Red
    1380.6409912109375,  # 05 - Vegetation Red Edge
    2233.432373046875,   # 06 - Vegetation Red Edge
    2549.379638671875,   # 07 - Vegetation Red Edge
    2643.248046875,      # 08 - NIR
    2643.531982421875,   # 08A - Vegetation Red Edge
    2852.87451171875,    # 09 - Water vapour
    2463.933349609375,   # 11 - SWIR
    1600.9207763671875,  # 12 - SWIR
    0.010281000286340714 # Cloud Probability
]

BANDS_STD = [
    204.20234680175780,  # 01 - Coastal aerosol
    227.25344848632812,  # 02 - Blue
    222.32545471191406,  # 03 - Green
    350.47235107421875,  # 04 - Red
    280.64367675781250,  # 05 - Vegetation Red Edge
    373.75210571289060,  # 06 - Vegetation Red Edge
    449.92361450195310,  # 07 - Vegetation Red Edge
    414.64981079101560,  # 08 - NIR
    415.10195922851560,  # 08A - Vegetation Red Edge
    413.89804077148440,  # 09 - Water vapour
    494.97430419921875,  # 11 - SWIR
    514.42297363281250,  # 12 - SWIR
    0.3447800576686859   # Cloud Probability
]


custom_imports = dict(
    imports=[
        "projects.dinov3.m-cashew-plant.dinov3",
        "projects.olmoearth.m-cashew-plant.olmoearth",
    ],
    allow_failed_imports=False,
)

dinov3_repo_dir = "projects/dinov3/m-cashew-plant/dinov3-main"
dinov3_weights_path = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

# ignore_index = 5
num_classes = 7
crop_size = 256
patch_size = 8
hidden_dim = 1024
batch_size = 16

train_pipeline = [
    dict(type="LoadSingleRSImgFromGEOBench", bands=[BANDS_NAME[i] for i in [3, 2, 1]]),
    dict(type="LoadSingleRSAnnFromGEOBench"),
    # dict(
    #     type="RandomCrop",
    #     crop_size=(crop_size, crop_size),
    #     cat_max_ratio=0.75,
    # ),
    # dict(
    #     type="RandomRotate",
    #     prob=0.5,
    #     degree=90,
    #     pad_val=0,
    #     seg_pad_val=0,
    # ),
    # dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    # dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadSingleRSImgFromGEOBench", bands=[BANDS_NAME[i] for i in [3, 2, 1]]),
    dict(type="LoadSingleRSAnnFromGEOBench"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='train',
        test_mode=False,
        pipeline=train_pipeline,
        reduce_zero_label=False
    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        split='valid',
        test_mode=True,
        pipeline=test_pipeline,
        reduce_zero_label=False
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    use_valid_mask=False,
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[BANDS_MEAN[i] for i in [3, 2, 1]],
    std=[BANDS_STD[i] for i in [3, 2, 1]],
    size=(crop_size, crop_size)
    # pad_val=0,
    # seg_pad_val=0,
    # size_divisor=patch_size,
    # test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3ViTBackbone",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        patch_size=patch_size,
        out_channels=hidden_dim,
        freeze=True,
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
        in_index=0,
        num_classes=num_classes,
        patch_size=patch_size,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    # test_cfg=dict(
    #     mode="slide",
    #     crop_size=(crop_size, crop_size),
    #     stride=(crop_size // 2, crop_size // 2),
    # ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.01, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-6,
        begin=0,
        end=15,
        by_epoch=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-5,
        begin=15,
        end=150,
        T_max=135,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=150, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=5,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="OlmoEarthVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

default_scope = "mmseg"
log_level = "INFO"
load_from = None
resume = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
