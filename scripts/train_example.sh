#!/usr/bin/env bash
set -euo pipefail

oracle-train \
  --train-data data/train-mp4.json \
  --val-data data/val.json \
  --output-dir outputs/training \
  --feature-cache-train outputs/training/train_features.pt \
  --feature-cache-val outputs/training/val_features.pt \
  --sample-cache-train outputs/training/train_samples.json \
  --sample-cache-val outputs/training/val_samples.json \
  --num-frames 25 \
  --motion-alpha 0.5 \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 1e-5 \
  --temperature 0.07 \
  --negative-weight 0.5 \
  --negative-ratio 0.5
