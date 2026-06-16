# Copyright (c) OpenMMLab. All rights reserved.
"""
DINOv3 骨干网络单元测试

测试 DinoVisionTransformerBackbone 的:
    1. 模块注册
    2. 不同架构的构建
    3. 前向传播输出形状
    4. 多尺度特征提取
    5. frozen_stages 冻结
"""

import sys
from unittest import TestCase

import torch

# 确保 DINOv3 源码路径在 sys.path 中
_dinov3_path = '/mnt/ht2_nas2/00-model/00-limx/Codes/dinov3-main'
if _dinov3_path not in sys.path:
    sys.path.insert(0, _dinov3_path)

from mmrotate.models.backbones.dinov3_backbone import (
    ARCH_SETTINGS,
    DEFAULT_OUT_INDICES,
    DinoVisionTransformerBackbone,
)
from mmrotate.registry import MODELS


class TestDinoVisionTransformerBackbone(TestCase):
    """DINOv3 骨干网络测试类"""

    def test_register(self):
        """测试 DinoVisionTransformerBackbone 是否已注册到 MODELS 注册表"""
        self.assertIn(
            'DinoVisionTransformerBackbone',
            MODELS.module_dict,
            'DinoVisionTransformerBackbone 未被正确注册到 MODELS',
        )

    def test_arch_settings(self):
        """测试所有预设架构参数的有效性"""
        expected_keys = ['embed_dim', 'depth', 'num_heads']
        for arch_name, arch_cfg in ARCH_SETTINGS.items():
            for key in expected_keys:
                self.assertIn(
                    key, arch_cfg,
                    f'架构 {arch_name} 缺少参数 {key}',
                )
            self.assertGreater(arch_cfg['embed_dim'], 0)
            self.assertGreater(arch_cfg['depth'], 0)
            self.assertGreater(arch_cfg['num_heads'], 0)
            # embed_dim 必须能被 num_heads 整除
            self.assertEqual(
                arch_cfg['embed_dim'] % arch_cfg['num_heads'],
                0,
                f'{arch_name} 的 embed_dim 不能被 num_heads 整除',
            )

    def test_default_out_indices(self):
        """测试默认 out_indices 的有效性"""
        for arch_name, out_indices in DEFAULT_OUT_INDICES.items():
            self.assertEqual(len(out_indices), 4,
                             f'{arch_name} 默认应该有 4 个输出层')
            depth = ARCH_SETTINGS[arch_name]['depth']
            for idx in out_indices:
                self.assertTrue(0 <= idx < depth,
                                f'{arch_name} 的 out_indices[{idx}] 超出范围 (depth={depth})')

    def test_vit_small_build(self):
        """测试构建 vit_small 架构"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=512,
            patch_size=16,
        )
        self.assertEqual(backbone.embed_dim, 384)
        self.assertEqual(backbone.depth, 12)
        self.assertEqual(backbone.num_heads, 6)
        self.assertEqual(backbone.num_patches, (512 // 16) ** 2)

    def test_vit_base_build(self):
        """测试构建 vit_base 架构"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_base',
            img_size=1024,
            patch_size=16,
        )
        self.assertEqual(backbone.embed_dim, 768)
        self.assertEqual(backbone.depth, 12)
        self.assertEqual(backbone.num_heads, 12)
        self.assertEqual(backbone.num_patches, (1024 // 16) ** 2)

    def test_vit_large_build(self):
        """测试构建 vit_large 架构"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_large',
            img_size=1024,
            patch_size=16,
        )
        self.assertEqual(backbone.embed_dim, 1024)
        self.assertEqual(backbone.depth, 24)
        self.assertEqual(backbone.num_heads, 16)

    def test_vit_giant2_build(self):
        """测试构建 vit_giant2 架构"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_giant2',
            img_size=1024,
            patch_size=16,
        )
        self.assertEqual(backbone.embed_dim, 1536)
        self.assertEqual(backbone.depth, 40)
        self.assertEqual(backbone.num_heads, 24)

    def test_custom_params(self):
        """测试自定义参数构建"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_base',
            img_size=512,
            patch_size=8,
            ffn_ratio=4.0,
            norm_layer='rmsnorm',
            ffn_layer='swiglu',
            qkv_bias=False,
            drop_path_rate=0.1,
            layerscale_init=1e-5,
        )
        self.assertEqual(backbone.img_size, 512)
        self.assertEqual(backbone.patch_size, 8)
        self.assertEqual(backbone.num_patches, (512 // 8) ** 2)

    def test_forward_vit_small_512(self):
        """测试 vit_small 前向传播 (img_size=512)"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=512,
            patch_size=16,
            out_indices=(2, 5, 8, 11),
        )
        backbone.eval()
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            feats = backbone(x)

        # 验证输出数量
        self.assertEqual(len(feats), 4)

        # 512 / 16 = 32, 特征图为 32x32
        expected_hw = 32
        for i, f in enumerate(feats):
            self.assertEqual(f.shape[0], 2, f'batch_size 应为 2')
            self.assertEqual(f.shape[1], 384, f'vit_small embed_dim=384')
            self.assertEqual(f.shape[2], expected_hw)
            self.assertEqual(f.shape[3], expected_hw)

        # 验证所有层通道数一致
        channels = [f.shape[1] for f in feats]
        self.assertEqual(len(set(channels)), 1,
                         'DINOv3 所有层输出通道应相同')

    def test_forward_vit_base_1024(self):
        """测试 vit_base 前向传播 (img_size=1024)"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_base',
            img_size=1024,
            patch_size=16,
            out_indices=(2, 5, 8, 11),
        )
        backbone.eval()
        x = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            feats = backbone(x)

        # 验证输出数量
        self.assertEqual(len(feats), 4)

        # 1024 / 16 = 64
        for f in feats:
            self.assertEqual(f.shape, (1, 768, 64, 64))

        # 验证通道数一致
        channels = [f.shape[1] for f in feats]
        self.assertEqual(len(set(channels)), 1)

    def test_forward_vit_base_800(self):
        """测试 vit_base 前向传播 (非正方形输入 img_size=800)"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_base',
            img_size=800,
            patch_size=16,
            out_indices=(2, 5, 8, 11),
        )
        backbone.eval()
        x = torch.randn(1, 3, 800, 800)
        with torch.no_grad():
            feats = backbone(x)

        # 800 / 16 = 50
        for f in feats:
            self.assertEqual(f.shape, (1, 768, 50, 50))

    def test_forward_single_output(self):
        """测试单尺度输出 (out_indices 只有一个值)"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=256,
            patch_size=16,
            out_indices=(-1,),  # 仅最后一层
        )
        backbone.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feat = backbone(x)

        # 单输出时返回单个 Tensor（非 tuple）
        self.assertEqual(feat.shape, (1, 384, 16, 16))

    def test_frozen_stages_minus_one(self):
        """测试 frozen_stages=-1: 不冻结任何参数"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=256,
            patch_size=16,
            frozen_stages=-1,
        )
        # 所有参数应可训练
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in backbone.parameters())
        self.assertEqual(trainable_params, total_params,
                         'frozen_stages=-1 时所有参数应可训练')

    def test_frozen_stages_zero(self):
        """测试 frozen_stages=0: 冻结 patch_embed"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=256,
            patch_size=16,
            frozen_stages=0,
        )
        # patch_embed 应被冻结
        for param in backbone.vit.patch_embed.parameters():
            self.assertFalse(param.requires_grad,
                             'frozen_stages=0 时 patch_embed 参数应被冻结')

    def test_frozen_stages_one(self):
        """测试 frozen_stages=1: 冻结 patch_embed + 第0层 block"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=256,
            patch_size=16,
            frozen_stages=1,
        )
        # patch_embed 应被冻结
        for param in backbone.vit.patch_embed.parameters():
            self.assertFalse(param.requires_grad)
        # 第0层 block 应被冻结
        for param in backbone.vit.blocks[0].parameters():
            self.assertFalse(param.requires_grad)
        # 第1层 block 应可训练
        for param in backbone.vit.blocks[1].parameters():
            self.assertTrue(param.requires_grad)

    def test_train_eval_mode(self):
        """测试 train/eval 模式切换"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',
            img_size=256,
            patch_size=16,
        )
        backbone.train()
        self.assertTrue(backbone.training)
        backbone.eval()
        self.assertFalse(backbone.training)
        backbone.train()
        self.assertTrue(backbone.training)

    def test_uniform_out_indices(self):
        """测试 _uniform_out_indices 静态方法"""
        # 12层 -> 4个输出
        indices = DinoVisionTransformerBackbone._uniform_out_indices(12, 4)
        self.assertEqual(indices, (2, 5, 8, 11))
        # 24层 -> 4个输出
        indices = DinoVisionTransformerBackbone._uniform_out_indices(24, 4)
        self.assertEqual(indices, (5, 11, 17, 23))

    def test_negative_out_indices(self):
        """测试负索引: -1 应转换为最后一层"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_small',  # depth=12
            img_size=256,
            patch_size=16,
            out_indices=(-1, -2, -3, -4),
        )
        # -1 -> 11, -2 -> 10, -3 -> 9, -4 -> 8
        self.assertEqual(backbone.out_indices, (11, 10, 9, 8))

    def test_params_count(self):
        """测试参数统计"""
        backbone = DinoVisionTransformerBackbone(
            arch='vit_base',
            img_size=1024,
            patch_size=16,
        )
        total = sum(p.numel() for p in backbone.parameters())
        # vit_base 约 85M~86M 参数
        self.assertGreater(total, 80_000_000)
        self.assertLess(total, 90_000_000)


if __name__ == '__main__':
    import unittest

    # 运行所有测试
    # exit=False 避免在 IDE 中触发 SystemExit 异常
    unittest.main(verbosity=2, exit=False)
