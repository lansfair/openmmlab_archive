# Copyright (c) OpenMMLab. All rights reserved.
"""
mmrotate 环境初始化与模块注册工具

该模块负责将 mmrotate 中的所有子模块（数据集、评估、模型、可视化等）
注册到 mmengine 的全局注册表中，并设置默认作用域（DefaultScope），
确保 mmrotate 的组件能够被正确地构建和使用。
"""

import datetime
import warnings

# 从 mmengine 导入 DefaultScope，用于管理全局默认注册表作用域
from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """
    注册 mmrotate 中的所有模块到注册表，并可选地设置默认作用域

    该函数通过导入 mmrotate 的子模块（datasets、evaluation、models、visualization），
    触发各子模块中的注册装饰器，从而将各类组件（如模型、数据集、评估指标、
    可视化器等）注册到 mmengine 的全局注册表中。

    同时，该函数还负责管理 DefaultScope（默认作用域），确保后续通过配置文件
    构建组件时，优先从 mmrotate 的注册表节点中查找对应的模块。

    参数:
        init_default_scope (bool): 是否初始化 mmrotate 的默认作用域。
            当设为 True 时，全局默认作用域将被设置为 'mmrotate'，
            所有注册表将从 mmrotate 的注册表节点中构建模块。
            关于注册表的更多信息，请参考：
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            默认值为 True。

    作用域管理逻辑:
        1. 如果当前没有 DefaultScope 实例，创建一个名为 'mmrotate' 的默认作用域
        2. 如果已存在 'mmrotate' 作用域，不做任何更改
        3. 如果存在其他作用域（如 'mmdet'），发出警告并创建新的 mmrotate 作用域，
           使用时间戳避免命名冲突
    """
    # ========== 导入 mmrotate 子模块，触发注册 ==========
    # 以下导入语句通过触发各子模块中的 @MODELS.register_module() 等装饰器，
    # 将 mmrotate 的所有组件自动注册到对应的全局注册表中

    # 导入数据集模块（如 DOTA、HRSC 等数据集类）
    import mmrotate.datasets  # noqa: F401,F403
    # 导入评估模块（如旋转框 mAP 等评估指标）
    import mmrotate.evaluation  # noqa: F401,F403
    # 导入模型模块（如 Oriented R-CNN、Rotated RetinaNet 等检测器）
    import mmrotate.models  # noqa: F401,F403
    # 导入可视化模块（如旋转框可视化器等）
    import mmrotate.visualization  # noqa: F401,F403

    # ========== 设置默认作用域 ==========
    # 仅当 init_default_scope 为 True 时才执行作用域初始化
    if init_default_scope:
        # 检查是否需要创建新的作用域实例：
        # 条件1：当前没有任何 DefaultScope 实例存在
        # 条件2：或者名为 'mmrotate' 的实例尚未被创建
        # 如果当前作用域已经是 'mmrotate'，则 never_created 为 False，不需要创建
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmrotate')
        if never_created:
            # 创建名为 'mmrotate' 的默认作用域实例
            # 此后所有注册表查找都将优先在 mmrotate 的作用域中进行
            DefaultScope.get_instance('mmrotate', scope_name='mmrotate')
            return

        # 获取当前的默认作用域实例
        current_scope = DefaultScope.get_current_instance()
        # 如果当前作用域不是 'mmrotate'（例如是 'mmdet'），发出警告
        if current_scope.scope_name != 'mmrotate':
            # 警告用户：当前默认作用域将被强制更改为 'mmrotate'
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmrotate", '
                          '`register_all_modules` will force the current'
                          'default scope to be "mmrotate". If this is not '
                          'expected, please set `init_default_scope=False`.')
            # 为避免与已有作用域的名称冲突，
            # 使用带时间戳的唯一名称创建新的作用域实例
            new_instance_name = f'mmrotate-{datetime.datetime.now()}'
            DefaultScope.get_instance(new_instance_name, scope_name='mmrotate')