#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# 自动读取归档目录绝对路径，非必要不修改。  
export MM_ARCHIVE_HOME=$(echo "$PWD" | cut -d '/' -f 1-5)

# 自动读取归档数据目录路径，非必要不修改。
export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/pretrained"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="$HOME/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME='mmdinov3'

# 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CUDA_VISIBLE_DEVICES='1'

# 训练启动配置名称，权重文件要求相同命名。
export CONFIG_NAME='uper_dinov3_dfc2020_b8'


SRCDIR=$PWD
cd "$PWD/../../../"
export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="/mnt/ht2-nas2/EO_test/zhc/mmsegmentation/work_dirs/"$(basename $SRCDIR)"/$CONFIG_NAME"
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/test.py" "$SRCDIR/configs/$CONFIG_NAME.py" "/mnt/ht2-nas2/EO_test/openmmlab-archive/src/v1/mmseg/projects/dinov3/DFC2020s2/work_dirs/DFC2020s2/uper_dinov3_dfc2020_b8/best_mIoU_epoch_45.pth" --show-dir "/mnt/ht2-nas2/EO_test/cyz/Copernicus-FM/mmsegmentation/show3_uper8_dino/"