_base_ = "./self-olmoearth-base-10m_1xb8-50e_m-cashew-plant-s2-linear.py"

# Align olmoearth_pretrain finetune eval for m-cashew-plant:
# ft_batch_size=4, num_workers=4, epochs=50, patch_size=16,
# NORM_NO_CLIP_2_STD, and 20% frozen-backbone warm start.
patch_size = 16

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
)

data_preprocessor = dict(
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    backbone=dict(
        patch_size=patch_size,
        fast_pass=True,
    ),
    decode_head=dict(patch_size=patch_size),
)

custom_hooks = [
    dict(
        type="FreezeBackboneUntilEpochHook",
        unfreeze_epoch=10,
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [ 
    dict(
        type='ReduceOnPlateauLR',
        monitor='mIoU',
        rule='greater',
        factor=0.2,
        patience=2,
        cooldown=10,
        min_value=0.0,
        by_epoch=True
    )
]


train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=10)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
)

auto_scale_lr = dict(enable=False, base_batch_size=4)
