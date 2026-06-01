#!/usr/bin/env bash
set -euo pipefail

# test1: test samples evaluated with descriptions already seen during training.
oracle-test1 \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --test-data data/test.json \
  --output-dir outputs/test1_seen_descriptions \
  --feature-cache outputs/test1_seen_descriptions/test_features.pt \
  --batch-size 256
