_base_ = "./dinov3-vitl16_b16-p8-150e_m-cashew-plant_rgb.py"


num_classes = 7
hidden_dim = 1024
norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    backbone=dict(
        out_indices=(7, 11, 15, 23),
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
        out_channels=hidden_dim,
        scales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        _delete_=True,
        type="UPerHead",
        in_channels=[hidden_dim, hidden_dim, hidden_dim, hidden_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
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
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
        ),
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.01),
)
