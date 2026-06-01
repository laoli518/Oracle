#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/test1_seen_descriptions
nohup bash scripts/test1_example.sh > outputs/test1_seen_descriptions/test1.log 2>&1 &
echo "Test1 PID: $!"
