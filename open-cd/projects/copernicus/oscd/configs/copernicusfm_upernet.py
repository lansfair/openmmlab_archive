custom_imports = dict(
    imports=['projects.copernicus.oscd'],
    allow_failed_imports=False)

# model settings
backbone_inchannels = 13    # OSCD
# backbone_inchannels = 3     # DFC
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    size=(256, 256),
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

copernicus_fm_checkpoint = "/mnt/ht2-nas2/EO_test/openmmlab-archive/v1/open-cd/pretrained/CopernicusFM_ViT_base_varlang_e100.pth"
s2_band_wavelengths = [
    440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190
]
s2_band_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]
patch_area = (16 * 10 / 1000)**2

model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=backbone_inchannels,
    backbone=dict(
        type='mmseg.CopernicusFMBackbone',
        arch='base',
        frozen_exclude=[],
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=copernicus_fm_checkpoint),
        band_wavelengths=s2_band_wavelengths,
        band_bandwidths=s2_band_bandwidths,
        var_option='spectrum',
        input_mode='spectral',
        kernel_size=16,
        patch_area=patch_area),
    neck=dict(
        type='FeatureFusionPyramid',
        policy='abs_diff',
        embed_dim=768,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='mmseg.UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(128, 128)))
