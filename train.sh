#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs
mkdir -p dp15_model

export PYTHONUNBUFFERED=1

DROPOUT_P=0.15
EPOCHS=4000
BATCH_SIZE=16
NUM_WORKERS=0

LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

python -u train_args.py \
  --train-dir "./data/MIX/train" \
  --val-dir "./data/MIX/val" \
  --save-dir "./dp15_model" \
  --model "srcnn_mc" \
  --dropout-p "${DROPOUT_P}" \
  --zoom-factor 4 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --val-save-every 10 \
  --lr-conv1 0.0001 \
  --lr-conv2 0.0001 \
  --lr-conv3 0.00001 \
  --num-workers 12 \
  --resume "./dp15_model/SRCNN_epoch_60.pth" \
  --device auto \
  2>&1 | tee "${LOG_FILE}"

echo "训练完成。"
echo "best 权重：./dp15_model/best_model_SRCNN.pth"
echo "已复制到：./dp15_model/best_model_SRCNN.pth"
echo "日志文件：${LOG_FILE}"