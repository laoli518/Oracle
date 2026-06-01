#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/training
nohup bash scripts/train_example.sh > outputs/training/train.log 2>&1 &
echo "Training PID: $!"
