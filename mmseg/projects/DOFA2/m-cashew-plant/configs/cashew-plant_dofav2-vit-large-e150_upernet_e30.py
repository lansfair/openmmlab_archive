import os

_base_ = [
    '../../../../configs/_base_/models/upernet_vit-b16_ln_mln.py',
    '../../../../configs/_base_/default_runtime.py',
    './dataset/cashew-plant.py'
]

custom_imports = dict(imports=['projects.DOFA2.m-cashew-plant.dofa2'])

DATA_SIZE = 256

BANDS_MEAN = [
    880.7075805664062, # R
    892.4611816406250, # G
    634.7583618164062  # B
]
BANDS_STD = [
    350.47235107421875, # R
    222.32545471191406, # G
    227.25344848632812  # B
]

NUM_CLASSES = 7
TRAIN_EPOCH = 30

BACKBONE_ARCH_NAME = 'large'
BACKBONE_ARCH_EMBED_DIM = {'base': 768, 'large': 1024}
BACKBONE_OUT_INDICES = [5, 11, 17, 23]
BACKBONE_EMBED_DIM = BACKBONE_ARCH_EMBED_DIM[BACKBONE_ARCH_NAME]

NECK_IN_CHANNELS = [BACKBONE_EMBED_DIM] * len(BACKBONE_OUT_INDICES)
NECK_OUT_CHANNELS = BACKBONE_EMBED_DIM
NECK_SCALES = [4, 2, 1, 0.5]

CHECKPOINT = os.path.join(os.environ.get('MM_ARCHIVE_CKPT_HOME'), 'dofav2_vit_large_e150.pth')


model = dict(
    data_preprocessor=dict(
        mean=BANDS_MEAN, 
        std=BANDS_STD, 
        size=(DATA_SIZE, DATA_SIZE)
    ),
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type="DOFAV2ViT",
        arch=BACKBONE_ARCH_NAME,
        img_size=DATA_SIZE,
        patch_size=14,
        model_bands=["RED", "GREEN", "BLUE"],
        out_indices=BACKBONE_OUT_INDICES,
        pos_interpolation_mode="bicubic",
        convert_patch_14_to_16=True,
        frozen_stages=True,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint=CHECKPOINT)
    ),
    neck=dict(
        in_channels=NECK_IN_CHANNELS,
        out_channels=NECK_OUT_CHANNELS,
        scales=NECK_SCALES
    ),
    decode_head=dict(
        in_channels=[NECK_OUT_CHANNELS] * len(NECK_SCALES),
        num_classes=NUM_CLASSES
    ),
    auxiliary_head=dict(
        in_channels=NECK_OUT_CHANNELS,
        num_classes=NUM_CLASSES
    )
)

param_scheduler = [dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=TRAIN_EPOCH)]
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.005, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=TRAIN_EPOCH, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

log_processor = dict(by_epoch=True)
