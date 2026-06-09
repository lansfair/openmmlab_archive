# Copernicus-open-cd

本项目将 Copernicus 模型集成到 [OPEN-CD](https://github.com/likyoo/open-cd) 框架中，支持在多个遥感/自然图像数据集上进行训练与推理。

|序号|数据集|骨干网络|任务指标|维护人| 项目 |
| -- | -- | -- | -- | -- | -- |
| 1 | oscd            | copernicus  | upernet： mIou 64.77%                               | 王若宇 |  [oscd](../open-cd/projects/copernicus/oscd)


## 目录

- [Copernicus-open-cd](#copernicus-open-cd)
  - [目录](#目录)
  - [项目结构](#项目结构)
  - [快速开始](#快速开始)
    - [1. 添加数据集工程](#1-添加数据集工程)
    - [2. 配置训练与测试脚本](#2-配置训练与测试脚本)
    - [3. 运行训练与测试](#3-运行训练与测试)
  - [日志与输出](#日志与输出)

## 项目结构

```bash
v1
├── open-cd/
│   ├── configs/
│   ├── pretrained/
│   ├── projects/
│   │   └── copernicus/
│   │       ├── oscd/                # 示例数据集：oscd
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
将您的数据集工程代码放置在 open-cd/projects/copernicus/ 目录下，工程文件夹以数据集名称命名。例如：
```
# 示例：为 oscd 数据集创建工程目录
cp -r /path/to/your/code open-cd/projects/copernicus/oscd
```
### 2. 配置训练与测试脚本
```
# 复制模板脚本（以 oscd 为例）

cp open-cd/projects/copernicus/oscd/train.sh open-cd/projects/copernicus/your_dataset/
cp open-cd/projects/copernicus/oscd/test.sh  open-cd/projects/copernicus/your_dataset/
```
编辑 train.sh 和 test.sh，根据您的数据集和实验设置调整以下内容（已在脚本中用注释标出）

### 3. 运行训练与测试
```
# 训练
cd open-cd/projects/copernicus/your_dataset
bash train.sh

# 测试（训练完成后执行）
bash test.sh
```

## 日志与输出
所有运行日志、模型权重文件及可视化结果均保存在：
```
open-cd/work_dir/${DATASET_NAME}/
```