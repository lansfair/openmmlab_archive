_base_ = ['../../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=["projects.olmoearth.potsdam.olmoearth"],
    allow_failed_imports=False,
)

data_root = "/mnt/ht2-nas2/EO_test/mty/potsdam"
# data_root = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/potsdam"
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/OlmoEarth-v1-Base"
model_config_path = f"{olmoearth_model_dir}/config.json"
weights_path = f"{olmoearth_model_dir}/weights.pth"

ignore_index = 5
num_classes = 5
num_timesteps = 1
crop_size = 512
patch_size = 8
hidden_dim = 768

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    # dict(type='LoadSinglePNGImageFromFile'),
    # dict(type='LoadLocalPtsdamAnnotations'),
    dict(
        type="RandomResize",
        scale=(crop_size, crop_size),
        ratio_range=(0.5, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_size=(crop_size, crop_size),
        cat_max_ratio=0.75,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type='LoadSinglePNGImageFromFile'),
    # dict(type="Resize", scale=(crop_size, crop_size), keep_ratio=True),
    dict(type="LoadAnnotations"),
    # dict(type='LoadLocalPtsdamAnnotations'),
    dict(
        type="RGBToOlmoEarthS2",
        num_timesteps=num_timesteps,
        rgb_channel_order="BGR",
        input_value_range="0_255",
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/train",
            seg_map_path="ann_dir/train",
        ),
        label_mapping="official_to_rvsa_class5_ignore5",
        pipeline=train_pipeline,
    ),
    # dataset=dict(
    #     type='LocalPotsdamDataset',
    #     data_root=data_root,
    #     data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
    #     ann_file='train.txt',
    #     ignore_index=5,
    #     reduce_zero_label=False,
    #     pipeline=train_pipeline,
    # ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="OlmoEarthPotsdamDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="img_dir/val",
            seg_map_path="ann_dir/val",
        ),
        label_mapping="official_to_rvsa_class5_ignore5",
        pipeline=test_pipeline,
    ),
    # dataset=dict(
    #     type='LocalPotsdamDataset',
    #     data_root=data_root,
    #     data_prefix=dict(img_path='img_dir', seg_map_path='ann_dir'),
    #     ann_file='valid.txt',
    #     ignore_index=5,
    #     reduce_zero_label=False,
    #     pipeline=test_pipeline,
    # ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    iou_metrics=["mIoU", "mFscore"],
    use_valid_mask=False,
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="OlmoEarthSegDataPreProcessor",
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
    size=(crop_size, crop_size),
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=768,
        pooling_type="mean",
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
        in_index=0,
        num_classes=num_classes,
        patch_size=patch_size,
        ignore_index=ignore_index,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

custom_hooks = []

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
    ),
    clip_grad=None,
)

param_scheduler = [
    dict(
        type="PolyLR",
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=80000, val_interval=8000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
# custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=8000,
        save_best="mIoU",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="OlmoEarthVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(by_epoch=False)
default_scope = "mmseg"
log_level = "INFO"
load_from = None
resume = False
