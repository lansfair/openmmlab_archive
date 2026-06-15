#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

export CONDA_HOME="/mnt/ht2-nas2/EO_test/miniconda3"                                    # miniconda3 安装路径，非必要不修改。
export CONDA_ENV_NAME='dfcmmseg'                                         # miniconda 虚拟环境名称，确保此环境可以正常运行你的项目。
export CUDA_VISIBLE_DEVICES='0'                                         # 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CONFIG_NAME='lp_olmoearth_dfc2020s2'           # 训练启动配置名称，权重文件要求相同命名。

SRCDIR=$PWD
DATASET=$(basename $SRCDIR)
cd "$PWD/../../../"
export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="./work_dirs/$DATASET"
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/train.py" "$SRCDIR/configs/$CONFIG_NAME.py" --work-dir "$WORK_DIR"
