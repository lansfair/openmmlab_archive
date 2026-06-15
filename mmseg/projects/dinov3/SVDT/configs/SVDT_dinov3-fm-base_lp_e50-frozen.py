import os
_base_ = [
    '../../../../configs/_base_/default_runtime.py',
    './SVDT.py'
]

custom_imports = dict(
    imports=['projects.dinov3.SVDT.dinov3'],
    allow_failed_imports=False)

dinov3_repo_dir = "projects/dinov3/SVDT/dinov3-main"
dinov3_weights_path = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

ignore_index = 255
num_classes = 2
crop_size = 256
patch_size = 16
hidden_dim = 1024

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    iou_metrics=["mIoU", "mFscore"],
    use_valid_mask=False,
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[72.4085, 89.7399, 69.6123],
    std=[32.854, 23.995, 23.123],
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
    train_cfg=dict(),
    test_cfg=dict(
        mode="slide",
        crop_size=(crop_size, crop_size),
        stride=(crop_size // 2, crop_size // 2),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-2,
        begin=0,
        end=5,
        by_epoch=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-5,
        begin=5,
        end=50,
        T_max=45,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100, log_metric_by_epoch=True),
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
    visualization=dict(type='SegVisualizationHook')
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