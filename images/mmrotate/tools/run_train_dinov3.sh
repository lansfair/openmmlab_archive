#!/bin/bash
# =============================================================================
# 启动脚本: Oriented R-CNN + DINOv3 (ViT-Base) 训练
#
# 用法:
#   bash tools/run_train_dinov3.sh              # 单卡训练
#   bash tools/run_train_dinov3.sh 0,1,2,3      # 4卡分布式训练
#   bash tools/run_train_dinov3.sh 0,1,2,3,4    # 5卡分布式训练
# =============================================================================

set -e

# 项目根目录
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/dinov3/oriented-rcnn-nanhu.py"
WORK_DIR="work_dirs/$(basename ${CONFIG%.py})"

# GPU 配置: 参数传入或默认单卡 CUDA_VISIBLE_DEVICES
GPUS="${1:-0}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=========================================="
echo " Config:   $CONFIG"
echo " Work Dir: $WORK_DIR"
echo " GPUs:     $GPUS  (共 ${NUM_GPUS} 张)"
echo "=========================================="

if [ "$NUM_GPUS" -le 1 ]; then
    # 单卡训练
    python tools/train.py "$CONFIG" --work-dir "$WORK_DIR"
else
    # 多卡分布式训练
    # 注意: lr 需要线性缩放，schedule_1x 已内置 auto_scale_lr
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --launcher pytorch
fi
