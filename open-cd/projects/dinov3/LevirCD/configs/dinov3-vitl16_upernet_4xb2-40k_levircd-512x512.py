import os

_base_ = [
    "../../../../configs/_base_/default_runtime.py",
    './levir_cd.py'
]

custom_imports = dict(
    imports=["projects.dinov3.LevirCD.opencd_dinov3"],
    allow_failed_imports=False)

# dinov3_root = "/mnt/ht2-nas2/EO_test/dataset/dinov3_pretrained"
dinov3_repo_dir = "projects/dinov3/LevirCD/dinov3-main"
# dinov3_weights_path = (
#     f"{dinov3_root}/DINOv3 ViT SAT-493M/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
# )

dinov3_weights_path = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

crop_size = (512, 512)
patch_size = 16
embed_dim = 1024
num_classes = 2

train_dataloader = dict(batch_size=2)

norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=patch_size,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="SiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=3,
    backbone=dict(
        type="DINOv3ViTBackbone",
        repo_dir=dinov3_repo_dir,
        model_name="dinov3_vitl16",
        weights_path=dinov3_weights_path,
        patch_size=patch_size,
        out_channels=embed_dim,
        freeze=True,
    ),
    neck=dict(
        type="DINOv3FeatureFusionPyramid",
        policy="abs_diff",
        embed_dim=embed_dim,
        out_channels=embed_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
        num_inputs=1,
    ),
    decode_head=dict(
        type="mmseg.UPerHead",
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0,
            ),
            dict(
                type="mmseg.DiceLoss",
                use_sigmoid=False,
                loss_weight=0.5,
                ignore_index=255,
            ),
        ],
    ),
    auxiliary_head=dict(
        type="mmseg.FCNHead",
        in_channels=embed_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        ignore_index=255,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss",
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(256, 256)),
)

img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

auto_scale_lr = dict(enable=False, base_batch_size=8)

# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000,
                    save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='CDVisualizationHook', interval=1, 
                       img_shape=(1024, 1024, 3)))
