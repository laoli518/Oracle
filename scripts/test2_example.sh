#!/usr/bin/env bash
set -euo pipefail

# test2: test samples evaluated with descriptions unseen during training.
oracle-test2 \
  --model-path outputs/training/best_direct_contrastive_model.pth \
  --test-data data/test.json \
  --desc-file data/descriptions/test2_unseen_descriptions_example.json \
  --output-dir outputs/test2_unseen_descriptions \
  --feature-cache outputs/test2_unseen_descriptions/test_features.pt \
  --batch-size 256
