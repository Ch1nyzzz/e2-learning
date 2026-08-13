#!/usr/bin/env bash
set -euo pipefail

uv run --extra train --extra alfworld accelerate launch \
  --config_file configs/accelerate/fsdp_2xa100_80gb.yaml \
  -m experience_learning.cli train \
  --config configs/alfworld_qwen3_8b.yaml \
  "$@"
