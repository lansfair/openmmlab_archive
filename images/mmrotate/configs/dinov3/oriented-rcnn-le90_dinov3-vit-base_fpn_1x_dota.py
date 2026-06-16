# =============================================================================
# Oriented R-CNN with DINOv3 (ViT-Base) Backbone + FPN
# =============================================================================
#
# 本配置文件演示如何在 mmrotate 中使用 DINOv3 作为骨干网络。
# DINOv3 是 Meta 发布的基于 ViT 的自监督预训练视觉模型，
# 使用 RoPE 位置编码和 SwiGLU FFN。
#
# 关键区别与 ResNet 骨干网络:
#   - DINOv3 输出 4 个尺度的特征图，但所有尺度的通道数相同（embed_dim=768）
#   - FPN 的 in_channels 全部设为 768
#   - 不需要 norm_cfg 和 frozen_stages（ViT 通常全参数微调或冻结 patch_embed）
#
# 预训练权重获取:
#   DINOv3 官方权重可从 Meta 发布页面下载，或使用 torch.hub 加载。
#   可通过 pretrained 参数指定 .pth 文件路径。
#
# 运行命令:
#   python tools/train.py configs/dinov3/oriented-rcnn-le90_dinov3-vit-base_fpn_1x_dota.py
# =============================================================================

_base_ = [
    '../_base_/datasets/dior_dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=[
        'mmrotate.models.backbones.dinov3_backbone',
        'mmrotate.engine.optimizers.nan_safe_optim_wrapper'],
    allow_failed_imports=False)

angle_version = 'le90'

# 日志输出频率：每 10 个 iteration 输出一次
default_hooks = dict(logger=dict(interval=10))

# DDP 多卡训练：DINOv3 backbone 中的 cls_token/norm 等参数不完全参与
# 检测 loss 的计算，需开启 find_unused_parameters 避免梯度同步报错
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

# ========== 数据加载配置 ==========
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True)

# ========== 训练轮次配置 ==========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=4)

# ========== 模型配置 ==========
model = dict(
    type='mmdet.FasterRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mmrotate.DinoVisionTransformerBackbone',  # 自定义的 DINOv3 适配器（跨注册表引用）
        arch='vit_small',                       # 架构: vit_large (embed_dim=1024, depth=24)
        img_size=800,                          # 输入图像尺寸
        patch_size=16,                         # patch 大小，800/16 = 50 个 patch
        out_indices=(2, 5, 8, 11),             # 从 4 个不同深度输出多尺度特征（24层均分）
        norm_layer='layernorm',                # 使用 LayerNorm
        ffn_layer='mlp',                       # 使用 FFN（DINOv3 默认，与预训练权重匹配）
        n_storage_tokens=4,                    # register tokens 数量，必须与预训练模型一致
        mask_k_bias=True,                      # QKV 中使用 LinearKMaskedBias，与预训练一致
        ffn_bias=True,                         # FFN bias
        proj_bias=True,                        # attention projection bias
        pos_embed_rope_dtype='fp32',           # RoPE 精度，与预训练一致
        pos_embed_rope_normalize_coords='separate',  # RoPE 坐标归一化方式
        pos_embed_rope_rescale_coords=2.0,     # 大图坐标缩放因子
        frozen_stages=0,                       # 冻结全部 12 层 ViT blocks（仅训练 neck + 检测头）
        forward_force_fp32=True,
        replace_nonfinite=True,
        max_missing_keys_ratio=0.20,
        check_pretrained_finite=True,
        layerscale_init=1e-5,
        pretrained='/mnt/ht2-nas2/EO_test/weights/Dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        # init_cfg=dict(type='Pretrained', checkpoint='/mnt/ht2_nas2/EO_test/weights/Dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),                         # 使用 DINOv3 自带的初始化
        init_cfg=None
    ),
    neck=dict(
        type='ViTDetFPN',
        in_channels=384,
        out_channels=256,
        num_outs=4,
        start_level=0,
        add_extra_convs=False,
        se_reduction=16,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        act_cfg=dict(type='GELU'),
    ),
    rpn_head=dict(
        type='mmrotate.OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
        anchor_generator=dict(  # 锚框生成器配置
            type='mmdet.AnchorGenerator',
            scales=[8],         # 基础anchor大小为8（在stride尺度空间的像素值，实际在原图上为8*stride）
            ratios=[0.5, 1.0, 2.0], # 3种宽高比，每个位置生成3个anchor
            strides=[4, 8, 16, 32], # 4个FPN特征层对应的下采样步长，决定anchor在各自层级负责检测的目标尺度范围
            use_box_type=True),     # 使用HBBoxes包装anchor，以便与旋转框的坐标转换兼容
        bbox_coder=dict(    #  边界框编解码器
            type='mmrotate.MidpointOffsetCoder',    # 中点偏移编码：anchor-旋转框，encode_size=6
            angle_version=angle_version,            # le90 = 长边表示法+角度范围[-90°，90°]
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # 编码时对6个偏移量(dx,dy,dw,dh,da,db)减去的均值
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),    # 编码时对6个偏移量的标准差归一化（角度偏移da,db使用更小的0.5，降低角度敏感度）
        loss_cls=dict(  # 分类损失
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # 使用sigmoid二分类（前景/背景），而非softmax多分类。RPN阶段不区分类别
        loss_bbox=dict( # 边界框回归损失
            type='mmdet.SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)), # beta = 1/9，SmoothL1在±beta范围内使用L2，超出使用L1，控制对离群点的敏感度
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(    # RoI特征提取器
            type='mmrotate.RotatedSingleRoIExtractor',  # 旋转RoI提取器
            roi_layer=dict(
                type='RoIAlignRotated', # 旋转RoIAlign：在旋转区域上做双线性插值，处理旋转框
                out_size=7,             # 输出特征图大小 7×7
                sample_num=2,           # 每个bin内的采样点数为2×2=4
                clockwise=True),        # 角度顺时针为正方向
            out_channels=256,           # 输出通道数，与FPN对齐
            featmap_strides=[4, 8, 16, 32]),    # FPN各层stride，用于根据RoI尺度选择对应的特征层级
        bbox_head=dict(     # 第二阶段检测头 (Shared2FCBBoxHead)
            type='mmdet.Shared2FCBBoxHead', # 两个全连接层，分类和回归共享
            predict_box_type='rbox',        # 预测旋转框（rbox），而非水平框（hbox）
            in_channels=256,                # 输入通道：RoIAlign后每个proposal是 256×7×7，flatten后为 256*7*7
            fc_out_channels=1024,           # 两个共享FC层的输出通道数（256×7×7 → 1024 → 1024）
            roi_feat_size=7,                # RoIAlign输出的空间尺寸，用于计算FC输入维度
            num_classes=20,                 # 类别数：DIOR数据集20类
            reg_predictor_cfg=dict(type='mmdet.Linear'),    # 回归分支：单个Linear层，输出 num_classes * code_size 或 1 * code_size（取决于reg_class_agnostic）
            cls_predictor_cfg=dict(type='mmdet.Linear'),    # 分类分支：单个Linear层，输出 num_classes+1（含背景）
            bbox_coder=dict(    # 第二阶段边界框编解码器
                type='mmrotate.DeltaXYWHTRBBoxCoder',   # 基于proposal回归(dx,dy,dw,dh,dt)到旋转框
                angle_version=angle_version,            # le90
                norm_factor=None,                       # 不对回归目标做额外归一化（使用target_stds即可）
                edge_swap=True,                         # 允许宽高交换 + 角度同步调整90°，解决角度周期性和边界问题
                proj_xy=True,                           # dx,dy投影到proposal的长边和短边方向后再计算偏移，增强对旋转的鲁棒性
                target_means=(.0, .0, .0, .0, .0),      # 编码时减去的均值（5个参数：dx,dy,dw,dh,dt）
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)), # 编码时除以的标准差。xy用0.1，wh用0.2，角度用0.1
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # ========== 训练配置 ==========
    train_cfg=dict(
        # --- RPN 训练配置 ---
        rpn=dict(
            # 正负样本分配器：将 anchor 与 GT 旋转框进行匹配
            assigner=dict(
                type='mmdet.MaxIoUAssigner',              # 最大IoU分配器
                pos_iou_thr=0.7,                           # IoU ≥ 0.7 的 anchor 标记为正样本
                neg_iou_thr=0.3,                           # IoU < 0.3 的 anchor 标记为负样本
                min_pos_iou=0.3,                           # 保证每个 GT 至少有一个 IoU ≥ 0.3 的正样本
                match_low_quality=True,                    # 即使最高 IoU 在 [0.3, 0.7) 之间也分配一个正样本（避免某些 GT 完全无匹配）
                ignore_iof_thr=-1,                         # -1 表示不忽略任何 anchor（否则 IoF 高于此阈值的会被标记忽略）
                iou_calculator=dict(type='mmrotate.RBbox2HBboxOverlaps2D')),
                                                           # ↑ IoU 计算方式：将旋转 GT 框转为外接水平框后，与水平 anchor 计算 IoU
            # 采样器：从分配的样本中选取固定数量的正负样本参与 loss 计算
            sampler=dict(
                type='mmdet.RandomSampler',                # 随机采样
                num=256,                                   # 每张图采样 256 个 anchor 用于 RPN loss
                pos_fraction=0.5,                          # 正样本占比 50%（128 个正样本，128 个负样本）
                neg_pos_ub=-1,                             # 负样本上限比例，-1 表示不限制
                add_gt_as_proposals=False),                # 不将 GT 本身作为额外的 proposal 加入采样池
            allowed_border=0,                              # 允许 anchor 超出图像边界的像素数，0 = 不允许（超出边界的 anchor 被忽略）
            pos_weight=-1,                                 # 正样本 loss 权重，-1 表示不使用额外权重（均衡正负样本）
            debug=False),                                  # 不输出调试信息
        # RPN proposal 生成配置（训练时 RPN 产生 proposals 送入 RoI Head）
        rpn_proposal=dict(
            nms_pre=2000,                                  # NMS 前保留的 top-k 个 proposal（按 score 排序）
            max_per_img=2000,                              # NMS 后每张图保留的最大 proposal 数量
            nms=dict(type='nms', iou_threshold=0.8),       # 标准 NMS，IoU 阈值 0.8（去除高度重叠的 proposal）
            min_bbox_size=0),                              # 最小边界框尺寸，0 表示不过滤小框
        # --- RoI Head (第二阶段) 训练配置 ---
        rcnn=dict(
            # 正负样本分配器：将 proposals 与 GT 旋转框匹配
            assigner=dict(
                type='mmdet.MaxIoUAssigner',               # 最大IoU分配器
                pos_iou_thr=0.5,                           # IoU ≥ 0.5 的 proposal 标记为正样本
                neg_iou_thr=0.5,                           # IoU < 0.5 的 proposal 标记为负样本
                                                           # ↑ pos_iou_thr == neg_iou_thr 表示没有「忽略区间」
                min_pos_iou=0.5,                           # 保证每个 GT 至少有一个 IoU ≥ 0.5 的正样本
                match_low_quality=False,                   # 不进行低质量匹配（与 RPN 不同，RoI 阶段对质量要求更高）
                iou_calculator=dict(type='mmrotate.RBboxOverlaps2D'),
                                                           # ↑ 旋转框之间的 IoU 计算（旋转 proposal vs 旋转 GT）
                ignore_iof_thr=-1),                        # 不忽略任何 proposal
            # 采样器
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,                                   # 每张图采样 512 个 proposal 用于 loss 计算
                pos_fraction=0.25,                         # 正样本占比 25%（128 个正样本，384 个负样本）
                                                           # ↑ RoI 阶段正样本比例低于 RPN（0.25 vs 0.5），因为正样本更稀少
                neg_pos_ub=-1,                             # 不限制负样本上限
                add_gt_as_proposals=True),                 # 将 GT 框也加入采样池（确保每个 GT 都有匹配的正样本）
            pos_weight=-1,                                 # 不使用额外的正样本权重
            debug=False)),
    # ========== 推理/测试配置 ==========
    test_cfg=dict(
        # RPN 推理配置
        rpn=dict(
            nms_pre=2000,                                  # NMS 前保留的 top-k 个 proposal
            max_per_img=2000,                              # NMS 后每张图保留的最大 proposal 数量
            nms=dict(type='nms', iou_threshold=0.8),       # 标准水平框 NMS（RPN 输出的是水平 anchor → 转换为水平框做 NMS）
            min_bbox_size=0),                              # 不过滤小框
        # RoI Head 推理配置
        rcnn=dict(
            nms_pre=2000,                                  # NMS 前按 score 保留的 top-k 个检测框
            min_bbox_size=0,                               # 不过滤小框
            score_thr=0.05,                                # score 低于 0.05 的检测框直接丢弃（减少后处理计算量）
            nms=dict(type='nms_rotated', iou_threshold=0.1), # 旋转框 NMS，IoU 阈值 0.1（旋转检测中 NMS 阈值通常较低）
            max_per_img=2000)))                             # 最终每张图最多保留 2000 个检测结果

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
#     clip_grad=dict(max_norm=35, norm_type=2))
# optimizer = dict(
#     type='AdamW',
#     lr=1e-4,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone.backbone': dict(lr_mult=0.25),
#             'backbone.layer_norms': dict(lr_mult=1.0),
#         },
#         norm_decay_mult=0.0,
#         bias_decay_mult=0.0,
#     ),
# )
# 
# optimizer_config = dict(
#     grad_clip=dict(max_norm=35, norm_type=2),
# )
# 
# 
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     min_lr_ratio=1e-3,
# )
# 
# ========== 优化器配置 (AdamW + NaNSafeOptimWrapper) ==========
# _delete_=True 确保完全替换基类 SGD 配置，避免 momentum 泄漏到 AdamW
#
# 关键: 使用 NaNSafeOptimWrapper 而非普通 OptimWrapper。
# ViT 微调时偶发的非有限梯度（NaN/Inf）会被 clip_grad_norm 放大到所有参数，
# 导致一次坏步即可让全部权重永久变为 NaN（不可恢复）。
# NaNSafeOptimWrapper 在 backward 后检测梯度有限性，若存在 NaN/Inf 则跳过
# 本步 optimizer.step() 并清零梯度，从而保护权重不被污染。
optim_wrapper = dict(
    _delete_=True,
    type='NaNSafeOptimWrapper',
    max_skips=200,  # 累计跳步超过此值则中止训练（防止静默死循环）
    optimizer=dict(
        type='AdamW',   # ViT微调标配，比SGD更稳定
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05), # ViT模型参数多，需较强正则化
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1), # 预训练权重只需小步微调
        },
        norm_decay_mult=0.0,
        bias_decay_mult=0.0),
    clip_grad=dict(max_norm=1.0, norm_type=2))   # 梯度裁剪，防止单步梯度爆炸

# ========== 学习率调度 (CosineAnnealing) ==========
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=True,
        begin=0,
        end=10),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        end=100,
        by_epoch=True,
        eta_min_ratio=1e-3,
    )
]