# Copyright (c) OpenMMLab. All rights reserved.
"""分布式安全的 NaN/Inf 梯度跳步 OptimWrapper。

关键点（避免 DDP 死锁）：
  - **所有 rank 都照常 backward**，保证梯度 all-reduce 同步完成；
  - all-reduce 之后 NaN 会同步到所有 rank，各 rank 对梯度有限性的判断一致；
  - 仅跳过本地的 optimizer.step()（纯本地操作，无通信），
    所以要么全部 step、要么全部跳过，不会发散，也不会死锁。
"""
import logging

import torch
from mmengine.logging import print_log
from mmengine.optim import OptimWrapper
from mmengine.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class NaNSafeOptimWrapper(OptimWrapper):
    """当本 step 的梯度出现 NaN/Inf 时，跳过 optimizer.step() 并清零梯度。"""

    def __init__(self, *args, max_skips: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        # max_skips: 累计跳过次数超过该值则抛错中止（-1 表示不限制）
        self._max_skips = max_skips
        self._skip_count = 0

    def update_params(self, loss, step_kwargs=None, zero_kwargs=None) -> None:
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}

        loss = self.scale_loss(loss)
        # 始终 backward：即使 loss 非有限，也要让所有 rank 参与 all-reduce
        self.backward(loss)

        if self.should_update():
            if self._grads_finite():
                self.step(**step_kwargs)          # step() 内部会做 clip_grad
            else:
                self._skip_count += 1
                print_log(
                    f'[NaNSafeOptimWrapper] 检测到非有限梯度，'
                    f'跳过第 {self._skip_count} 个 step（loss={float(loss):.4g}）',
                    logger='current',
                    level=logging.WARNING)
                if 0 <= self._max_skips < self._skip_count:
                    raise RuntimeError(
                        f'[NaNSafeOptimWrapper] 累计跳过 {self._skip_count} 次 '
                        f'超过上限 {self._max_skips}，训练已发散，主动中止。')
            # 无论是否 step，都清零梯度，避免坏梯度残留到下一步
            self.zero_grad(**zero_kwargs)

    def _grads_finite(self) -> bool:
        """检查 optimizer 管理的所有梯度是否均为有限值。"""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    return False
        return True