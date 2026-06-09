# Copernicus-MMDet

本项目将 Copernicus 模型集成到 [MMDetection](https://github.com/open-mmlab/mmdetection) 框架中，支持在多个遥感/自然图像数据集上进行训练与推理。

|序号|数据集|骨干网络|任务指标|维护人| 项目 |
| -- | -- | -- | -- | -- | -- |
| 1 | m-cashew-plant  | copernicus  | LP(冻结): mIoU 25.76% </br> LP(非冻结): mIoU 75.20%  | 郑谊峰 | [m-cashew-plant](./projects/copernicus/m-cashew-plant)|

## 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [1. 添加数据集工程](#1-添加数据集工程)
  - [2. 配置训练与测试脚本](#2-配置训练与测试脚本)
  - [3. 运行训练与测试](#3-运行训练与测试)
- [日志与输出](#日志与输出)

## 项目结构

```bash
v1
├── mmdet/
│   ├── configs/
│   ├── projects/
│   │   └── copernicus/
│   │       ├── potsdam/                # 示例数据集：Potsdam
│   │       │   ├── checkpoints         # 存放指标最好的权重文件及训练日志，名称与配置文件保持一致。
│   │       │   ├── test.sh             # 测试启动脚本
│   │       │   ├── train.sh            # 训练启动脚本
│   │       │   └── ...                 # 数据集相关代码与配置
│   │       └── your_dataset/           # 用户自定义数据集（按需创建）
│   ├── tools/
│   └── ...
└────── ...
```

## 快速开始
### 1. 添加数据集工程
将您的数据集工程代码放置在 mmdet/projects/copernicus/ 目录下，工程文件夹以数据集名称命名。例如：
```
# 示例：为 Potsdam 数据集创建工程目录
cp -r /path/to/your/code mmdet/projects/copernicus/potsdam
```
### 2. 配置训练与测试脚本
```
# 复制模板脚本（以 Potsdam 为例）

cp mmdet/projects/copernicus/potsdam/train.sh mmdet/projects/copernicus/your_dataset/
cp mmdet/projects/copernicus/potsdam/test.sh  mmdet/projects/copernicus/your_dataset/
```
编辑 train.sh 和 test.sh，根据您的数据集和实验设置调整以下内容（已在脚本中用注释标出）

### 3. 运行训练与测试
```
# 训练
cd mmdet/projects/copernicus/your_dataset
bash train.sh

# 测试（训练完成后执行）
bash test.sh
```

## 日志与输出
所有运行日志、模型权重文件及可视化结果均保存在：
```
/tmp/work_dir/mmdet/${DATASET_NAME}/
```