#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/fine_tuning/fight_nofight
nohup python fine_tuning/fight_nofight_adapter_finetune_v12.py \
  --data_root /path/to/fight_media \
  --annotation /path/to/fight_annotations.xlsx \
  --model outputs/training/best_direct_contrastive_model.pth \
  --output_dir outputs/fine_tuning/fight_nofight \
  --n_frames 25 \
  --shots_max 10 \
  --shots_min 5 \
  > outputs/fine_tuning/fight_nofight/run.log 2>&1 &

echo "Started. Log: outputs/fine_tuning/fight_nofight/run.log"
