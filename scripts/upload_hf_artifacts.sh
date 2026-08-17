#!/usr/bin/env bash
# Upload the trained artifacts needed on a new training machine to the Hugging
# Face Hub: the two Stage-1 cold-start checkpoints and the RWML data files
# (the "随包提供" list in docs/handoff_experiments.md §1).
#
# Everything is configurable via env vars; there are no machine-specific
# paths. Defaults match this repository's layout and the erv1n namespace.
#
#   HF_NAMESPACE   Hub user or org (default: erv1n)
#   HF_PRIVATE     "true" to create private repos (default: public)
#   Q25_CKPT       local Qwen2.5 cold-start HF checkpoint dir
#   Q3_CKPT        local Qwen3 cold-start HF checkpoint dir
#   RWML_DATA_DIR  local data/ dir holding the RWML files
#   Q25_REPO / Q3_REPO / RWML_DATA_REPO   Hub repo names
#
# Usage:  bash scripts/upload_hf_artifacts.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HF_NAMESPACE=${HF_NAMESPACE:-erv1n}
HF_PRIVATE=${HF_PRIVATE:-false}
Q25_CKPT=${Q25_CKPT:-"$REPO_ROOT/outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200"}
Q3_CKPT=${Q3_CKPT:-"$REPO_ROOT/outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220"}
RWML_DATA_DIR=${RWML_DATA_DIR:-"$REPO_ROOT/data"}
Q25_REPO=${Q25_REPO:-e2l-alfworld-qwen25-7b-coldstart}
Q3_REPO=${Q3_REPO:-e2l-alfworld-qwen3-8b-coldstart}
RWML_DATA_REPO=${RWML_DATA_REPO:-e2l-rwml-alfworld-data}

PRIVATE_FLAG=()
if [[ "$HF_PRIVATE" == "true" ]]; then
  PRIVATE_FLAG=(--private)
fi

# RWML files the remote machine needs (docs/handoff_experiments.md §1).
RWML_FILES=(
  rwml_grpo/train.parquet
  rwml_grpo/val.parquet
  rwml_tau_calibration.json
  rwml_alfworld_qwen25_7b_validation_merged10k.jsonl
)

for f in "$Q25_CKPT" "$Q3_CKPT"; do
  [[ -f "$f/config.json" && -f "$f/model.safetensors.index.json" && -f "$f/tokenizer.json" ]] \
    || { echo "incomplete HF checkpoint: $f"; exit 1; }
done

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

model_card() {
  # $1 = base model id, $2 = source checkpoint descriptor
  cat <<EOF
---
library_name: transformers
license: apache-2.0
---

# E2L ALFWorld cold-start checkpoint ($1)

Stage-1 cold-start checkpoint from
[Ch1nyzzz/e2-learning](https://github.com/Ch1nyzzz/e2-learning): full-parameter
online mistake-driven experience learning (next-observation SFT gated by a
semantic judge) on ALFWorld \`AlfredTWEnv\`, initialized from
\`$1\`. Source run artifact: \`$2\`.

Intended use: initialization ("cold start") for Stage-2 policy GRPO training
with [verl-agent](https://github.com/langfengQ/verl-agent). Load as a regular
Hugging Face checkpoint (\`AutoModelForCausalLM.from_pretrained\`).
\`controller_state.json\` is experiment-controller provenance metadata and can
be ignored by inference/training stacks.
EOF
}

echo "=== [1/3] $HF_NAMESPACE/$Q25_REPO <- $Q25_CKPT"
model_card "Qwen/Qwen2.5-7B-Instruct" \
  "outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200" > "$STAGING/README.q25.md"
hf upload "$HF_NAMESPACE/$Q25_REPO" "$Q25_CKPT" . "${PRIVATE_FLAG[@]}" \
  --commit-message "Upload ALFWorld Qwen2.5-7B Stage-1 cold-start checkpoint"
hf upload "$HF_NAMESPACE/$Q25_REPO" "$STAGING/README.q25.md" README.md \
  --commit-message "Add model card"

echo "=== [2/3] $HF_NAMESPACE/$Q3_REPO <- $Q3_CKPT"
model_card "Qwen/Qwen3-8B" \
  "outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220" > "$STAGING/README.q3.md"
hf upload "$HF_NAMESPACE/$Q3_REPO" "$Q3_CKPT" . "${PRIVATE_FLAG[@]}" \
  --commit-message "Upload ALFWorld Qwen3-8B Stage-1 cold-start checkpoint"
hf upload "$HF_NAMESPACE/$Q3_REPO" "$STAGING/README.q3.md" README.md \
  --commit-message "Add model card"

echo "=== [3/3] $HF_NAMESPACE/$RWML_DATA_REPO (dataset) <- RWML files"
DATA_STAGE="$STAGING/dataset"
for rel in "${RWML_FILES[@]}"; do
  src="$RWML_DATA_DIR/$rel"
  [[ -f "$src" ]] || { echo "missing RWML file: $src"; exit 1; }
  dst="$DATA_STAGE/$rel"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
done
cat > "$DATA_STAGE/README.md" <<'EOF'
---
license: apache-2.0
---

# E2L RWML ALFWorld data

RWML (arXiv:2602.05842) reproduction data from
[Ch1nyzzz/e2-learning](https://github.com/Ch1nyzzz/e2-learning), collected with
`Qwen/Qwen2.5-7B-Instruct` rollouts on ALFWorld:

- `rwml_grpo/train.parquet`, `rwml_grpo/val.parquet` — prompts for RWML GRPO
  training (filtered merged-10k corpus);
- `rwml_tau_calibration.json` — calibrated embedding-distance threshold tau_d;
- `rwml_alfworld_qwen25_7b_validation_merged10k.jsonl` — validation transitions.

Paths mirror the source repository's `data/` layout; pull them back with
`scripts/pull_hf_artifacts.sh`.
EOF
hf upload "$HF_NAMESPACE/$RWML_DATA_REPO" "$DATA_STAGE" . --repo-type dataset \
  "${PRIVATE_FLAG[@]}" --commit-message "Upload RWML GRPO data"

echo "=== UPLOAD_DONE ==="
