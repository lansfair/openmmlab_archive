# Copernicus-MMSeg

本项目将 Copernicus 模型集成到 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 框架中，支持在多个遥感/自然图像数据集上进行训练与推理。

|序号|数据集|骨干网络|任务指标|维护人| 项目 |
| -- | -- | -- | -- | -- | -- |
| 1 | m-cashew-plant  | copernicus  | LP(冻结): mIoU 25.76% </br> LP(非冻结): mIoU 56.35%  | 郑谊峰 | [源码](./projects/copernicus/m-cashew-plant)|
| 2 | potsdam         | copernicus  | LP(冻结): mIoU 60.72% </br> Upernet(冻结): mIoU 82.38%  | 张浩晨 | [源码](./projects/copernicus/potsdam)|
| 3 | m-sa-crop-type  | copernicus  | LP(冻结): mIoU 29.00% </br> LP(非冻结): mIoU 35.81%  | 福辉 | [源码](./projects/copernicus/m-sa-crop-type)|
| 4 | SVDT            | copernicus  | LP(冻结): mIoU 71.69% </br> Upernet(冻结): mIoU 85.98%  | 张浩晨 | [源码](./projects/copernicus/SVDT)|
| 5 | m-cashew-plant  | olmoearth  | LP(冻结): mIoU 11.19% </br> LP(非冻结): mIoU 85.88%  | 郑谊峰 | [源码](./projects/olmoearth/m-cashew-plant)|
| 6 | potsdam         | olmoearth  | LP(冻结): mIoU 56.24% </br> Upernet(冻结): mIoU 73.38% </br> LP(非冻结+p8): mIoU 55.26% </br> upernet(非冻结+p8): mIoU 70.72% </br> LP(非冻结+p16): mIoU 63.61% </br> upernet(非冻结+p16): mIoU 74.75%| 张浩晨 | [源码](./projects/olmoearth/potsdam)|
| 7 | potsdam         | dinov3     | LP(冻结): mIoU 77.16% </br> Upernet(冻结): mIoU 86.10%  | 张浩晨 | [源码](./projects/dinov3/potsdam)|
| 8 | SVDT            | dinov3     | LP(冻结): mIoU 80.91% </br> Upernet(冻结): mIoU 86.76%  | 张浩晨 | [源码](./projects/dinov3/SVDT)|
| 9 | m-cashew-plant  | DOFA2  | Upernet(非冻结): mIoU 64.66%  | 郑谊峰 | [源码](./projects/DOFA2/m-cashew-plant)|
| 10 | m-cashew-plant  | dinov3  | Upernet(冻结): mIoU 87.78%  | 郑谊峰 | [源码](./projects/dinov3/m-cashew-plant)|
| 11 | m-cashew-plant  | self-copernicus  | LP(冻结): mIoU 21.10% </br> LP(非冻结): mIoU 32.46%  | 郑谊峰 | [源码](./projects/copernicus/m-cashew-plant)|
| 12 | m-cashew-plant-b12  | self-copernicus  | LP(冻结): mIoU 21.59% </br> LP(非冻结): mIoU 29.95%  | 郑谊峰 | [源码](./projects/copernicus/m-cashew-plant)|
| 12 | m-cashew-plant  | self-olmoearth  | LP(非冻结): mIoU 85.70%  | 郑谊峰 | [源码](./projects/olmoearth/m-cashew-plant)|

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
├── mmseg/
│   ├── configs/
│   ├── pretrained/
│   ├── projects/
│   │   └── copernicus/
│   │       ├── potsdam/                # 示例数据集：Potsdam
│   │       │   ├── checkpoints         # 存放指标最好的权重文件及训练日志，名称与配置文件保持一致。
│   │       │   ├── test.sh             # 测试启动脚本
│   │       │   ├── train.sh            # 训练启动脚本
│   │       │   └── ...                 # 数据集相关代码与配置
│   │       └── your_dataset/           # 用户自定义数据集（按需创建）
│   ├── tools/
│   ├── work_dir/
│   │   └── ${DATASET_NAME}/            # 训练与测试日志、权重文件存放目录
│   └── ...
└────── ...
```

## 快速开始
### 1. 添加数据集工程
将您的数据集工程代码放置在 mmseg/projects/copernicus/ 目录下，工程文件夹以数据集名称命名。例如：
```
# 示例：为 Potsdam 数据集创建工程目录
cp -r /path/to/your/code mmseg/projects/copernicus/potsdam
```
### 2. 配置训练与测试脚本
```
# 复制模板脚本（以 Potsdam 为例）

cp mmseg/projects/copernicus/potsdam/train.sh mmseg/projects/copernicus/your_dataset/
cp mmseg/projects/copernicus/potsdam/test.sh  mmseg/projects/copernicus/your_dataset/
```
编辑 train.sh 和 test.sh，根据您的数据集和实验设置调整以下内容（已在脚本中用注释标出）

### 3. 运行训练与测试
```
# 训练
cd mmseg/projects/copernicus/your_dataset
bash train.sh

# 测试（训练完成后执行）
bash test.sh
```

## 日志与输出
所有运行日志、模型权重文件及可视化结果均保存在：
```
mmseg/work_dir/${DATASET_NAME}/
```