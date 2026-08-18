#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the six Stage-2 arms, with the per-hardware
# parameter profile baked in so nobody has to remember the env-var combos:
#
#   Qwen2.5 arms -> 8x A100 80GB single node (official verl-agent 8-GPU
#     recipe: one vLLM engine per GPU, no FSDP offload, CUDA graphs on).
#   Qwen3 arms   -> 4x 80GB GPUs (TP=2 so the two engines land on NVLink
#     pairs 0-1 / 2-3, FSDP offload on, conservative vLLM memory so a
#     co-tenant process cannot OOM the run, prompt cap 5120 because the
#     Qwen3 tokenizer produces up to ~4.2k-token validation prompts).
#
# Usage:
#   ARM=q25_dual bash scripts/launch_stage2_arm.sh [extra hydra overrides]
#   ARM one of: q25_plain q25_pure q25_dual q3_plain q3_pure q3_dual
#     plain = base model, no penalty     (E5)
#     pure  = cold-start ckpt, no penalty (E2)
#     dual  = cold-start ckpt + futile penalty (E1)
#   All arms use the stage2 prompt and history_length=50.
#
# Overridable: GPUS, EXPERIMENT_NAME, Q25_COLDSTART, Q3_COLDSTART, plus
# everything train_policy_grpo_stage2.sh accepts. Defaults for the cold-start
# checkpoints point at this machine's outputs/ tree; on the remote box set
# them to the HF-downloaded paths.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM=${ARM:?Set ARM=q25_plain|q25_pure|q25_dual|q3_plain|q3_pure|q3_dual}

Q25_COLDSTART=${Q25_COLDSTART:-$REPO_ROOT/outputs/alfworld_qwen25_7b_active_10000_hf/checkpoints/env_step_001200}
Q3_COLDSTART=${Q3_COLDSTART:-$REPO_ROOT/outputs/alfworld_qwen3_8b_active_10000_hf/checkpoints/env_step_006220}

case "$ARM" in
  q25_plain) MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}; USE_FUTILE_PENALTY=false; family=q25 ;;
  q25_pure)  MODEL_PATH=${MODEL_PATH:-$Q25_COLDSTART};           USE_FUTILE_PENALTY=false; family=q25 ;;
  q25_dual)  MODEL_PATH=${MODEL_PATH:-$Q25_COLDSTART};           USE_FUTILE_PENALTY=true;  family=q25 ;;
  q3_plain)  MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B};            USE_FUTILE_PENALTY=false; family=q3 ;;
  q3_pure)   MODEL_PATH=${MODEL_PATH:-$Q3_COLDSTART};            USE_FUTILE_PENALTY=false; family=q3 ;;
  q3_dual)   MODEL_PATH=${MODEL_PATH:-$Q3_COLDSTART};            USE_FUTILE_PENALTY=true;  family=q3 ;;
  *) echo "error: unknown ARM=$ARM" >&2; exit 1 ;;
esac

if [[ "$family" == q25 ]]; then
  # 8x A100 80GB profile (official verl-agent 8-GPU ALFWorld recipe).
  GPUS=${GPUS:-0,1,2,3,4,5,6,7}
  ROLLOUT_TP=${ROLLOUT_TP:-1}
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
  FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-false}
  FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-false}
  ENFORCE_EAGER=${ENFORCE_EAGER:-false}
  FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-false}
  # Qwen2.5 prompts never exceeded the recipe cap.
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
else
  # 4x 80GB profile (this machine: H100 PCIe, NVLink pairs 0-1 / 2-3).
  GPUS=${GPUS:-0,1,2,3}
  ROLLOUT_TP=${ROLLOUT_TP:-2}
  # 0.35 survives a co-tenant process on shared GPUs; on an exclusive node
  # GPU_MEM_UTIL=0.6 PPO_MAX_TOKEN_LEN=16384 LOGPROB_MAX_TOKEN_LEN=32768
  # is the measured H100 speed profile (~1.9x per-step).
  GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}
  FSDP_PARAM_OFFLOAD=${FSDP_PARAM_OFFLOAD:-true}
  FSDP_OPTIMIZER_OFFLOAD=${FSDP_OPTIMIZER_OFFLOAD:-true}
  ENFORCE_EAGER=${ENFORCE_EAGER:-true}
  FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-true}
  # Qwen3 tokenizer: longest valid_seen prompt is ~4.2k tokens; 4096 aborts
  # with truncation=error. PPO/LOGPROB token budgets scale automatically.
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-${family}_stage2_${ARM##*_}}

# Checkpoints saved by the stage-1 online pipeline carry an
# 'extra_special_tokens' list that newer transformers rejects
# (AttributeError: 'list' object has no attribute 'keys'). Rename it to the
# standard 'additional_special_tokens' in place (keeping a .bak) before the
# trainer touches the tokenizer.
if [[ -f "$MODEL_PATH/tokenizer_config.json" ]]; then
  python3 - "$MODEL_PATH/tokenizer_config.json" <<'PY'
import json, shutil, sys
path = sys.argv[1]
cfg = json.load(open(path))
if isinstance(cfg.get("extra_special_tokens"), list):
    shutil.copyfile(path, path + ".bak")
    cfg["additional_special_tokens"] = cfg.pop("extra_special_tokens")
    json.dump(cfg, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[launch_stage2_arm] fixed extra_special_tokens in {path}")
PY
fi

echo "[launch_stage2_arm] ARM=$ARM MODEL_PATH=$MODEL_PATH GPUS=$GPUS TP=$ROLLOUT_TP" \
     "MEM_UTIL=$GPU_MEM_UTIL PROMPT_CAP=$MAX_PROMPT_LENGTH PENALTY=$USE_FUTILE_PENALTY"
echo "[launch_stage2_arm] check 'nvidia-smi' first if you are unsure the GPUs are free."

MODEL_PATH="$MODEL_PATH" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
GPUS="$GPUS" \
ROLLOUT_TP="$ROLLOUT_TP" \
GPU_MEM_UTIL="$GPU_MEM_UTIL" \
FSDP_PARAM_OFFLOAD="$FSDP_PARAM_OFFLOAD" \
FSDP_OPTIMIZER_OFFLOAD="$FSDP_OPTIMIZER_OFFLOAD" \
ENFORCE_EAGER="$ENFORCE_EAGER" \
FREE_CACHE_ENGINE="$FREE_CACHE_ENGINE" \
MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH" \
USE_FUTILE_PENALTY="$USE_FUTILE_PENALTY" \
exec bash "$REPO_ROOT/scripts/train_policy_grpo_stage2.sh" "$@"
