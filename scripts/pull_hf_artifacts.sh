#!/usr/bin/env bash
# Pull the trained artifacts uploaded by scripts/upload_hf_artifacts.sh onto a
# new training machine. Everything is configurable via env vars; defaults
# mirror the upload side and this repository's layout.
#
#   HF_NAMESPACE   Hub user or org (default: erv1n)
#   Q25_CKPT       local destination for the Qwen2.5 cold-start checkpoint
#   Q3_CKPT        local destination for the Qwen3 cold-start checkpoint
#   RWML_DATA_DIR  local data/ dir to receive the RWML files
#   Q25_REPO / Q3_REPO / RWML_DATA_REPO   Hub repo names
#
# Usage:  bash scripts/pull_hf_artifacts.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_NAMESPACE=${HF_NAMESPACE:-erv1n}
Q25_CKPT=${Q25_CKPT:-"$REPO_ROOT/outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200"}
Q3_CKPT=${Q3_CKPT:-"$REPO_ROOT/outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220"}
RWML_DATA_DIR=${RWML_DATA_DIR:-"$REPO_ROOT/data"}
Q25_REPO=${Q25_REPO:-e2l-alfworld-qwen25-7b-coldstart}
Q3_REPO=${Q3_REPO:-e2l-alfworld-qwen3-8b-coldstart}
RWML_DATA_REPO=${RWML_DATA_REPO:-e2l-rwml-alfworld-data}

echo "=== [1/3] $HF_NAMESPACE/$Q25_REPO -> $Q25_CKPT"
hf download "$HF_NAMESPACE/$Q25_REPO" --local-dir "$Q25_CKPT"

echo "=== [2/3] $HF_NAMESPACE/$Q3_REPO -> $Q3_CKPT"
hf download "$HF_NAMESPACE/$Q3_REPO" --local-dir "$Q3_CKPT"

echo "=== [3/3] $HF_NAMESPACE/$RWML_DATA_REPO (dataset) -> $RWML_DATA_DIR"
hf download "$HF_NAMESPACE/$RWML_DATA_REPO" --repo-type dataset \
  --local-dir "$RWML_DATA_DIR"

cat <<EOF
=== PULL_DONE ===
Remaining setup on the new machine (not on the Hub):
  1. Base models are pulled from the Hub on first use: Qwen/Qwen2.5-7B-Instruct,
     Qwen/Qwen3-8B, Qwen/Qwen3-Embedding-0.6B.
  2. ALFWorld data: set ALFWORLD_DATA, then run \`alfworld-download\`.
  3. verl-agent checkout (branch stage2-dual-reward) at VERL_AGENT_DIR.
  4. See docs/handoff_experiments.md for the full checklist.
EOF
