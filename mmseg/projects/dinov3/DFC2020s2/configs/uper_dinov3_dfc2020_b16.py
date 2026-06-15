_base_ = [
    '../../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=[
        "projects.dinov3.DFC2020s2.dinov3",
    ],
    allow_failed_imports=False,
)

data_root = '/mnt/ht2-nas2/EO_test/cyz/Copernicus-FM/copernicus/dataset/dfc2020_s1s2'

dinov3_repo_dir = "projects/dinov3/LoveDA/dinov3-main"
dinov3_weights_path =  "/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/pretrained/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
work_dir = "./work_dirs/dinov3-vitl16_4xb16_dfc2020_uper"

ignore_index = 255
num_classes = 8
crop_size = 256
patch_size = 16
hidden_dim = 1024

norm_cfg = dict(type="SyncBN", requires_grad=True)

train_pipeline = [
    dict(type="LoadSingleRSImageFromFile"),
    dict(type="LoadDFC2020Annotations"),
    dict(type="Resize", scale=crop_size, keep_ratio=False),
    dict(
        type="RandomRotate",
        prob=0.5,
        degree=90,
        pad_val=0,
        seg_pad_val=ignore_index,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"), 
    dict(type="PackDinoSegInputs"),
]


test_pipeline = [
    dict(type="LoadSingleRSImageFromFile"),
    dict(type="LoadDFC2020Annotations"),
    dict(type="PackDinoSegInputs"),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="DFC2020S2Dataset",
        data_root=data_root,
        ann_file="dfc-train-new.csv",
        data_prefix=dict(img_path="s2", seg_map_path="dfc"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DFC2020S2Dataset",
        data_root=data_root,
        ann_file="dfc-val-new.csv",
        data_prefix=dict(img_path="s2", seg_map_path="dfc"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="DFC2020S2Dataset",
        data_root=data_root,
        ann_file="test10.csv",
        data_prefix=dict(img_path="s2", seg_map_path="dfc"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    iou_metrics=["mIoU"],
    use_valid_mask=False,
)
test_evaluator = val_evaluator


data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[1117.2, 1041.8, 946.5],
    std=[736.0, 684.8, 620.0],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=ignore_index,
    size_divisor=patch_size,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3ViTBackbone2",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        patch_size=patch_size,
        out_channels=hidden_dim,
        freeze=True,
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[hidden_dim],
        out_channels=hidden_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=hidden_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),

)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.01, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [ 
    dict(
        type='OneCycleLR',
        eta_max=1e-2,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    )
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)
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
    # visualization=dict(type="OlmoEarthVisualizationHook"),
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