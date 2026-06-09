_base_ = [
    './copernicusfm_upernet.py',
    './oscd_s2_256.py',
    './default_runtime.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=True)
randomness = dict(seed=0)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01))

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        rule='greater',
        save_last=True),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    visualization=dict(type='CDVisualizationHook', interval=1))
    
