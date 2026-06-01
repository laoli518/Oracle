#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/fine_tuning/lying_posture
nohup python fine_tuning/lying_posture_adapter_finetune_v11.py \
  --data_root /path/to/lying_images \
  --annotation /path/to/lying_annotations.xlsx \
  --model outputs/training/best_direct_contrastive_model.pth \
  --output_dir outputs/fine_tuning/lying_posture \
  --seed 42 \
  --shots_max 10 \
  --shots_min 5 \
  > outputs/fine_tuning/lying_posture/run.log 2>&1 &

echo "Started. Log: outputs/fine_tuning/lying_posture/run.log"
