# Copyright (c) OpenMMLab. All rights reserved.
"""
DINOv3 骨干网络适配层

将 DINOv3 (DinoVisionTransformer) 适配为 mmrotate 的骨干网络（backbone），
输出多尺度特征图，供 FPN 等 Neck 模块使用。

DINOv3 原始 forward 返回字典: {"x_norm_clstoken", "x_norm_patchtokens", ...}
本适配层从多个中间 Transformer Block 提取 patch tokens，
并 reshape 为 2D 特征图，返回 tuple of Tensors（多尺度特征）。

使用方式（配置文件中）:
    backbone=dict(
        type='DinoVisionTransformerBackbone',
        arch='vit_base',          # vit_small / vit_base / vit_large / vit_giant2 / vit_7b
        img_size=1024,            # 输入图像尺寸
        patch_size=16,            # patch 大小
        out_indices=(3, 5, 7, 11),  # 从哪些 block 索引输出特征（0-based）
        norm_layer='layernorm',
        ffn_layer='swiglu',
    )

依赖: 需要 DINOv3 源码路径可通过 sys.path 访问
    sys.path.insert(0, '/path/to/dinov3-main')
"""
import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from mmengine.model import BaseModule

from mmrotate.registry import MODELS

# ========== 导入 DINOv3 模块 ==========
# 通过 sys.path 动态导入，无需 pip install
_dinov3_path = '/mnt/ht2-nas2/00-model/00-limx/Codes/dinov3-main'
if _dinov3_path not in sys.path:
    sys.path.insert(0, _dinov3_path)

from dinov3.models.vision_transformer import DinoVisionTransformer  # noqa: E402, F811


# ========== 预设的 ViT 架构参数 ==========
# 不同规模模型的 embed_dim 和 num_heads 对照表
ARCH_SETTINGS = {
    'vit_small':     dict(embed_dim=384,  depth=12, num_heads=6),
    'vit_base':      dict(embed_dim=768,  depth=12, num_heads=12),
    'vit_large':     dict(embed_dim=1024, depth=24, num_heads=16),
    'vit_so400m':    dict(embed_dim=1152, depth=27, num_heads=18),
    'vit_huge2':     dict(embed_dim=1280, depth=32, num_heads=20),
    'vit_giant2':    dict(embed_dim=1536, depth=40, num_heads=24),
    'vit_7b':        dict(embed_dim=4096, depth=40, num_heads=32),
}

# 不同架构的默认 out_indices（将深度均分为4个阶段）
DEFAULT_OUT_INDICES = {
    'vit_small':     (2, 5, 8, 11),     # 12层: 0-based
    'vit_base':      (2, 5, 8, 11),     # 12层
    'vit_large':     (5, 11, 17, 23),   # 24层
    'vit_so400m':    (6, 13, 20, 26),   # 27层
    'vit_huge2':     (7, 15, 23, 31),   # 32层
    'vit_giant2':    (9, 19, 29, 39),   # 40层
    'vit_7b':        (9, 19, 29, 39),   # 40层
}


@MODELS.register_module()
class DinoVisionTransformerBackbone(BaseModule):
    """
    DINOv3 骨干网络适配器

    将 DinoVisionTransformer 包装为 mmrotate 可用的骨干网络，
    从指定 Transformer Block 层提取 patch tokens 并输出为多尺度 2D 特征图。

    Args:
        arch (str): 预设架构名称，可选 'vit_small', 'vit_base', 'vit_large',
            'vit_so400m', 'vit_huge2', 'vit_giant2', 'vit_7b'。
            当指定 arch 时，embed_dim/depth/num_heads 等参数自动填充。
            默认为 'vit_base'。
        img_size (int): 输入图像尺寸（宽高相同）。默认 1024。
        patch_size (int): Patch 大小。默认 16。
        in_chans (int): 输入图像通道数。默认 3。
        embed_dim (int): 嵌入维度。若指定 arch 则自动填充。
        depth (int): Transformer 层数。若指定 arch 则自动填充。
        num_heads (int): 注意力头数。若指定 arch 则自动填充。
        ffn_ratio (float): FFN 扩展比例。默认 4.0。
        out_indices (Sequence[int]): 输出特征的 Transformer Block 索引（0-based）。
            若未指定，根据 arch 自动选择均匀分布的 4 个层级。
            例如 vit_base 默认 (2, 5, 8, 11)。
            -1 表示最后一层输出（仅单尺度）。
        norm_layer (str): 归一化层类型，可选 'layernorm', 'rmsnorm'。默认 'layernorm'。
        ffn_layer (str): FFN 层类型，可选 'mlp', 'swiglu'。默认 'swiglu'。
        qkv_bias (bool): QKV 投影是否使用 bias。默认 True。
        drop_path_rate (float): DropPath 概率。默认 0.0。
        layerscale_init (float | None): LayerScale 初始值。默认 None。
        pos_embed_rope_base (float): RoPE 位置编码的 base 参数。默认 100.0。
        pos_embed_rope_dtype (str): RoPE 的数据类型，'fp16', 'bf16', 'fp32'。默认 'bf16'。
        n_storage_tokens (int): 额外的存储 token 数量。默认 0。
        frozen_stages (int): 冻结前 frozen_stages 个阶段的参数。
            -1 表示不冻结。默认 -1。
        init_cfg (dict, optional): 初始化配置。默认 None。
        pretrained (str, optional): 预训练权重路径（.pth 文件）。
            如果提供，将在 init_weights 中加载。
        **kwargs: 传递给 DinoVisionTransformer 的其他参数。
    """

    def __init__(
        self,
        arch: str = 'vit_base',
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: Optional[int] = None,
        depth: Optional[int] = None,
        num_heads: Optional[int] = None,
        ffn_ratio: float = 4.0,
        out_indices: Optional[Sequence[int]] = None,
        norm_layer: str = 'layernorm',
        ffn_layer: str = 'swiglu',
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_dtype: str = 'bf16',
        n_storage_tokens: int = 0,
        frozen_stages: int = -1,
        init_cfg: Optional[dict] = None,
        pretrained: Optional[str] = None,
        forward_force_fp32: bool=False,
        replace_nonfinite: bool=True,
        max_missing_keys_ratio: Optional[float]=None,
        check_pretrained_finite: bool=True,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)

        # 根据预设架构填充参数
        if arch in ARCH_SETTINGS:
            arch_cfg = ARCH_SETTINGS[arch]
            embed_dim = embed_dim or arch_cfg['embed_dim']
            depth = depth or arch_cfg['depth']
            num_heads = num_heads or arch_cfg['num_heads']
        else:
            assert embed_dim is not None, '必须指定 embed_dim 或有效的 arch'
            assert depth is not None, '必须指定 depth 或有效的 arch'
            assert num_heads is not None, '必须指定 num_heads 或有效的 arch'

        self.arch = arch
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained
        self.forward_force_fp32 = forward_force_fp32
        self.frozen_stages = frozen_stages
        self.replace_nonfinite = replace_nonfinite
        self.max_missing_keys_ratio = max_missing_keys_ratio
        self.check_pretrained_finite = check_pretrained_finite

        # 设置输出特征的 block 索引
        if out_indices is None:
            # 如果未指定，从预设中获取，否则均匀采样4个层级
            out_indices = DEFAULT_OUT_INDICES.get(
                arch, self._uniform_out_indices(depth))
        # 将 -1 转换为最后一层索引
        out_indices = tuple(
            depth + idx if idx < 0 else idx for idx in out_indices
        )
        self.out_indices = out_indices

        # 构建 DINOv3 的 DinoVisionTransformer
        self.vit = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            drop_path_rate=drop_path_rate,
            layerscale_init=layerscale_init,
            norm_layer=norm_layer,
            ffn_layer=ffn_layer,
            pos_embed_rope_base=pos_embed_rope_base,
            pos_embed_rope_dtype=pos_embed_rope_dtype,
            n_storage_tokens=n_storage_tokens,
            **kwargs,
        )

        # 初始化权重
        self.vit.init_weights()

        # 如果指定了预训练权重，加载之
        if pretrained is not None:
            self._load_pretrained(pretrained)

        # mask_token 仅用于预训练的 MIM 任务，下游检测不需要
        # 设为 non-trainable 避免 DDP 报 unused parameter 和浪费优化器状态
        self.vit.mask_token.requires_grad = False

        # 标记已初始化，防止 runner 的 _init_model_weights() 再次调用
        # BaseModule.init_weights() 从而覆盖已加载的预训练权重
        self._is_init = True

        # 冻结指定阶段
        self._freeze_stages()

    @staticmethod
    def _uniform_out_indices(depth: int, num_outs: int = 4) -> Tuple[int, ...]:
        """
        均匀采样 num_outs 个输出层索引（0-based）

        参数:
            depth (int): 总层数
            num_outs (int): 期望输出层数

        返回:
            Tuple[int, ...]: 采样得到的索引列表
        """
        step = depth / num_outs
        indices = [int(step * (i + 1)) - 1 for i in range(num_outs)]
        return tuple(indices)

    def _load_pretrained(self, pretrained: str) -> None:
        """
        加载预训练权重

        参数:
            pretrained (str): 预训练权重文件路径 (.pth)
        """
        checkpoint = torch.load(pretrained, map_location='cpu')
        # 兼容不同的 checkpoint 格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint

        # 移除可能的前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            # 移除 'module.' 前缀（DDP 包装）
            new_key = key.replace('module.', '')
            # 移除 'backbone.' 前缀（如果是检测器 checkpoint）
            new_key = new_key.replace('backbone.', '')
            # 移除 'encoder.' 前缀（DINOv3 checkpoint 常用）
            new_key = new_key.replace('encoder.', '')
            new_state_dict[new_key] = value

        # 从 checkpoint 键名推断其使用的 ffn 层类型
        # 若 checkpoint 使用 'mlp.fc1'/'mlp.fc2'（标准MLP），
        # 但当前模型使用 'mlp.w1'/'mlp.w2'/'mlp.w3'（SwiGLU），
        # 则 MLP 权重维度不兼容，需要跳过并给出警告
        # new_state_dict = self._remap_checkpoint_keys(new_state_dict)

        # 加载到 vit 中，允许缺失/不匹配的键
        missing, unexpected = self.vit.load_state_dict(
            new_state_dict, strict=False)
        total_keys = len(self.vit.state_dict())
        loaded_keys = total_keys - len(missing)
        print(f'[DinoVisionTransformerBackbone] loaded keys: {loaded_keys}/{total_keys}')
        if missing:
            print(f'[DinoVisionTransformerBackbone] 缺少的键: {missing[:5]}...')
        if unexpected:
            print(f'[DinoVisionTransformerBackbone] 未预期的键: {unexpected[:5]}...')

        if self.max_missing_keys_ratio is not None:
            missing_ratio = len(missing) / max(total_keys, 1)
            if missing_ratio > self.max_missing_keys_ratio:
                raise RuntimeError(
                    '[DinoVisionTransformerBackbone] Too many missing keys when loading'
                    f'pretrained weights ({missing_ratio:.2%}) >'
                    f'{self.max_missing_keys_ratio}).'
                    'This usually indicates a checkpoint/model mismatch'
                    '(e.g., arch/ffn_layer/storage_tokens/bias settings).'
                )
        
        if self.check_pretrained_finite:
            self._assert_finite_weights()

    def _remap_checkpoint_keys(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ckpt_keys = list(state_dict.keys())
        model_keys = list(self.vit.state_dict().keys())

        ckpt_has_fc = any('mlp.fc1.' in k for k in ckpt_keys)
        ckpt_has_swiglu = any('mlp.w1.' in k for k in ckpt_keys)
        model_has_fc = any('mlp.fc1.' in k for k in model_keys)
        model_has_swiglu = any('mlp.w1.' in k for k in model_keys)
        has_bias_mask = any('attn.qkv.bias_mask' in k for k in ckpt_keys)

        mlp_mismatch = (ckpt_has_fc and model_has_swiglu) or (ckpt_has_swiglu and model_has_fc)

        remapped = {}
        num_skipped_mlp = 0
        num_skipped_other = 0
        num_remapped_bias = 0

        for key, value in state_dict.items():
            if key.startswith('storage_tokens'):
                num_skipped_other += 1
                continue
            if key.endswith('ls1.gamma') or key.endswith('ls2.gamma'):
                num_skipped_other += 1
                continue

            if mlp_mismatch and ('.mlp.fc1.' in key or '.mlp.fc2.' in key or
                                 '.mlp.w1.' in key or '.mlp.w2.' in key or
                                 '.mlp.w3.' in key):
                num_skipped_mlp += 1
                continue

            if has_bias_mask and 'attn.qkv.bias_mask' in key:
                new_key = key.replace('attn.qkv.bias_mask', 'attn.qkv.bias')
                if new_key in self.vit.state_dict():
                    remapped[new_key] = value
                    num_remapped_bias += 1
                    continue

            remapped[key] = value

        if mlp_mismatch and num_skipped_mlp > 0:
            print(f'[DinoVisionTransformerBackbone] 模型 MLP 与 checkpoint 不匹配，'
                  f'已跳过 {num_skipped_mlp} 个 MLP 参数')
        elif not mlp_mismatch:
            print(f'[DinoVisionTransformerBackbone] 模型 MLP 与 checkpoint 一致，正常加载')
        if num_remapped_bias > 0:
            print(f'[DinoVisionTransformerBackbone] 已将 {num_remapped_bias} '
                  f'个 bias_mask 映射为 bias')
        if num_skipped_other > 0:
            print(f'[DinoVisionTransformerBackbone] 已跳过 {num_skipped_other} '
                  f'个不需要的参数 (storage_tokens/LayerScale)')

        return remapped
    
    def _assert_finite_weights(self) -> None:
        bad_params: List[str] = []
        bad_buffers: List[str] = []

        for name, param in self.vit.named_parameters():
            if not torch.isfinite(param).all():
                bad_params.append(name)
                if len(bad_params) >= 5:
                    break

        for name, buffer in self.vit.named_buffers():
            if not torch.isfinite(buffer).all():
                bad_buffers.append(name)
                if len(bad_buffers) >= 5:
                    break

        if bad_params or bad_buffers:
            raise RuntimeError(
                '[DinoVisionTransformerBackbone] Non-finite values found in '
                f'loaded pretrained weights. bad_params={bad_params}, '
                f'bad_buffers={bad_buffers}.')

    def _freeze_stages(self) -> None:
        """
        冻结指定阶段的参数

        frozen_stages 含义：
            - -1: 不冻结任何参数
            - 0:  仅冻结 patch_embed
            - 1:  冻结 patch_embed + 第0层 block
            - 2:  冻结 patch_embed + 前2层 block, etc.
        """
        if self.frozen_stages < 0:
            return

        # 冻结 patch_embed
        if self.frozen_stages >= 0:
            self.vit.patch_embed.eval()
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False

        # 冻结指定数量的 transformer blocks
        for i in range(min(self.frozen_stages, self.depth)):
            block = self.vit.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        if not torch.isfinite(x).all():
            num_bad = (~torch.isfinite(x)).sum().item()
            raise RuntimeError(
                f'[DinoBackbone] Non-finite values detected in input. '
                f'count={num_bad}, shape={tuple(x.shape)}, dtype={x.dtype}')

        if self.forward_force_fp32:
            x = x.float()

        autocast_ctx = nullcontext()
        if self.forward_force_fp32 and torch.is_autocast_enabled():
            autocast_ctx = torch.autocast(device_type=x.device.type, enabled=False)

        with autocast_ctx:
            outs = self.vit.get_intermediate_layers(
                x, n=list(self.out_indices), reshape=True, norm=True)

        outs = tuple(o.float() for o in outs)
        out_indices = list(self.out_indices)
        outs_list = None

        for i, o in enumerate(outs):
            nonfinite_mask = ~torch.isfinite(o)
            if nonfinite_mask.any():
                layer_idx = out_indices[i]
                num_nonfinite = nonfinite_mask.sum().item()
                message = (f'[DinoBackbone] Non-finite values in output {i} '
                           f'(layer {layer_idx}). count={num_nonfinite}')

                if not self.training or not self.replace_nonfinite:
                    raise RuntimeError(message)

                if outs_list is None:
                    outs_list = list(outs)
                # 全部非有限 → 权重已被污染，返回零张量（梯度为0，不进一步恶化）
                # 部分非有限 → 仅替换坏值，保留有效特征
                if num_nonfinite >= o.numel():
                    outs_list[i] = torch.zeros_like(o)
                    print(f'{message}, ALL non-finite → replacing with zeros.')
                else:
                    outs_list[i] = torch.nan_to_num(
                        o, nan=0.0, posinf=0.0, neginf=0.0)
                    print(f'{message}, replacing bad values with zeros.')

        if outs_list is not None:
            outs = tuple(outs_list)
        return outs

    def train(self, mode: bool = True) -> None:
        """
        切换训练/评估模式，同时重新应用冻结逻辑

        参数:
            mode (bool): True 为训练模式，False 为评估模式
        """
        super().train(mode=mode)
        self._freeze_stages()
        return self