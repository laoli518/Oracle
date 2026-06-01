#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/test2_unseen_descriptions
nohup bash scripts/test2_example.sh > outputs/test2_unseen_descriptions/test2.log 2>&1 &
echo "Test2 PID: $!"
