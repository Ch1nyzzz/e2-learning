#!/usr/bin/env bash
set -euo pipefail

# Stage 2 dual-reward GRPO on ALFWorld: env success reward (+10) plus a futile
# repetition penalty with streak escalation and SR-annealed coefficient.
# Derived from train_policy_grpo_4xa100_nothink.sh; one script serves all 6
# arms via env-var combos (dual = futile penalty on, pure = penalty off; both
# use the stage2 prompt; plain-from-base arms start from the base HF model
# instead of a cold-start checkpoint).
#
# Every arm additionally requires an explicit GPUS=... (N_GPUS is derived
# from it); the script refuses to guess so it cannot collide with a live run.
#
# The 6 arms:
#   # 1) Q2.5 dual  (cold-start ckpt + futile penalty + stage2 prompt)
#   MODEL_PATH=/path/to/q25_coldstart_hf EXPERIMENT_NAME=q25_stage2_dual \
#     bash scripts/train_policy_grpo_stage2.sh
#   # 2) Q2.5 pure  (cold-start ckpt, no penalty, stage2 prompt)
#   MODEL_PATH=/path/to/q25_coldstart_hf EXPERIMENT_NAME=q25_stage2_pure \
#     USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh
#   # 3) Q3 dual
#   MODEL_PATH=/path/to/q3_coldstart_hf EXPERIMENT_NAME=q3_stage2_dual \
#     bash scripts/train_policy_grpo_stage2.sh
#   # 4) Q3 pure
#   MODEL_PATH=/path/to/q3_coldstart_hf EXPERIMENT_NAME=q3_stage2_pure \
#     USE_FUTILE_PENALTY=false bash scripts/train_policy_grpo_stage2.sh
#   # 5) Q2.5 plain-from-base (no penalty, stage2 prompt, base model)
#   MODEL_PATH=Qwen/Qwen2.5-7B-Instruct EXPERIMENT_NAME=q25_stage2_plain \
#     USE_FUTILE_PENALTY=false STAGE2_PROMPT=true \
#     bash scripts/train_policy_grpo_stage2.sh
#   # 6) Q3 plain-from-base
#   MODEL_PATH=Qwen/Qwen3-8B EXPERIMENT_NAME=q3_stage2_plain \
#     USE_FUTILE_PENALTY=false STAGE2_PROMPT=true \
#     bash scripts/train_policy_grpo_stage2.sh
#
# 8x A100 80GB single node: set GPUS=0,1,2,3,4,5,6,7 and use the 8-GPU
# overrides below, which track verl-agent's official ALFWorld GRPO recipe and
# other 8-GPU ALFWorld GRPO releases (EP-R1, CoEvoSkill; see
# docs/handoff_experiments.md):
#   ROLLOUT_TP=1 GPU_MEM_UTIL=0.6 FSDP_PARAM_OFFLOAD=false \
#   FSDP_OPTIMIZER_OFFLOAD=false ENFORCE_EAGER=false FREE_CACHE_ENGINE=false
# Keep TRAIN_TASKS_PER_UPDATE=16 x GROUP_SIZE=8 (128 trajectories/update): that
# is the paper-standard update size and preserves comparability across arms --
# 8 GPUs buy wall-clock speed, not a larger batch.
#
# W&B: default logger is console+wandb. Key lives in .env.example (shared
# with the remote machine). Offline: WANDB_MODE=offline. Disable:
# LOGGER="['console']".
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH (cold-start HF checkpoint or base model)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
# Machine-specific locations; all overridable, defaults follow common
# conventions ($HOME-based) rather than any particular host.
VERL_AGENT_DIR=${VERL_AGENT_DIR:-$HOME/verl-agent}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
CKPT_ROOT=${CKPT_ROOT:-$REPO_ROOT/checkpoints/e2l_policy_grpo}
# Where verl-agent's data_preprocess.prepare writes/reads the text parquet.
VERL_DATA_DIR=${VERL_DATA_DIR:-$HOME/data/verl-agent}
# Interpreter for verl-agent commands: the checkout's .venv locally; container
# images set VERL_PYTHON to their system interpreter instead.
VERL_PYTHON=${VERL_PYTHON:-.venv/bin/python}
# No default on purpose: a reflexive launch must not land on GPUs a live run
# already owns (e.g. the stage-1 baseline on 0,1,2). Check nvidia-smi first.
GPUS=${GPUS:?Set GPUS explicitly, e.g. GPUS=0,1,2,3 (check nvidia-smi for free devices)}
N_GPUS=${N_GPUS:-$(($(tr -cd ',' <<<"$GPUS" | wc -c) + 1))}

# Stage 2 knobs (defaults = the dual arm; see design spec).
USE_FUTILE_PENALTY=${USE_FUTILE_PENALTY:-true}
FUTILE_PENALTY_COEF=${FUTILE_PENALTY_COEF:-0.25}
FUTILE_PENALTY_CAP_UNITS=${FUTILE_PENALTY_CAP_UNITS:-12}
FUTILE_SR_TARGET=${FUTILE_SR_TARGET:-0.7}
STAGE2_PROMPT=${STAGE2_PROMPT:-true}
# Stage 2 needs the full episode in context for history review, hence the long
# history window and larger prompt budget than the stage-1 script (50/4096 vs
# 2/2048).
HISTORY_LENGTH=${HISTORY_LENGTH:-50}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Token-budget scaling for dynamic batching: the stage-1 script paired
# 2048+512-token samples with ppo_max_token_len_per_gpu=4096 (~1.6 samples)
# and log_prob_max_token_len_per_gpu=8192 (~3.2 samples). Keep the same ratio
# by scaling both linearly with MAX_PROMPT_LENGTH: at the default 4096-token
# prompts (4608/sample) that gives 8192 and 16384 respectively.
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((4096 * MAX_PROMPT_LENGTH / 2048))}
LOGPROB_MAX_TOKEN_LEN=${LOGPROB_MAX_TOKEN_LEN:-$((8192 * MAX_PROMPT_LENGTH / 2048))}

# Horizon, budget, and validation cadence follow verl-agent's official
# examples/grpo_trainer/run_alfworld.sh (max_steps=50, 150 epochs, val on 128
# tasks every 5 steps). Checkpointing deviates deliberately: the official
# script never saves (save_freq=-1); we save periodically and a pruner keeps
# only the best-by-val checkpoint, HF format.
TRAIN_STEPS=${TRAIN_STEPS:-150}
TRAIN_TASKS_PER_UPDATE=${TRAIN_TASKS_PER_UPDATE:-16}
GROUP_SIZE=${GROUP_SIZE:-8}
VAL_TASKS=${VAL_TASKS:-128}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-50}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}
EVAL_DATASET=${EVAL_DATASET:-eval_in_distribution}
# Full rollout dump (train + val), gzipped jsonl per collection call.
# Set ROLLOUT_LOG_DIR=null to disable.
ROLLOUT_LOG_DIR=${ROLLOUT_LOG_DIR:-$CKPT_ROOT/$EXPERIMENT_NAME/rollouts}

for env_file in "$REPO_ROOT/.env" "$REPO_ROOT/.env.example"; do
  [[ -f "$env_file" ]] || continue
  while IFS='=' read -r key value; do
    [[ "$key" == WANDB_* ]] || continue
    [[ -z "$value" ]] && continue
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < <(grep -E '^WANDB_[A-Z0-9_]+=' "$env_file" || true)
done
[[ -z "${WANDB_ENTITY:-}" ]] && unset WANDB_ENTITY
# wandb files land on the mounted repo, not inside /opt/verl-agent.
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT}"
PROJECT_NAME=${WANDB_PROJECT:-e2l_policy_grpo}
LOGGER=${LOGGER:-"['console','wandb']"}
if [[ "$LOGGER" == *wandb* && -z "${WANDB_API_KEY:-}" && "${WANDB_MODE:-}" != "offline" ]]; then
  echo "error: wandb is on but WANDB_API_KEY is empty (expected in .env.example)." >&2
  echo "  or WANDB_MODE=offline, or LOGGER=\"['console']\" to skip." >&2
  exit 1
fi

export ALFWORLD_DATA
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=true
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
# Another user keeps a long-lived Ray cluster registered under /tmp/ray; force
# an isolated local instance so we neither join nor clobber it.
export RAY_ADDRESS=local
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_e2l_$USER}
# Both local disks sit above Ray's default 95% disk-usage alarm; text-only
# rollouts never spill the object store, so raise the threshold to stop the
# raylet warning spam that otherwise floods the training log every 10s.
export RAY_local_fs_capacity_threshold=0.995

cd "$VERL_AGENT_DIR"

"$VERL_PYTHON" -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size "$TRAIN_TASKS_PER_UPDATE" \
  --val_data_size "$VAL_TASKS"

"$VERL_PYTHON" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.gamma=1.0 \
  algorithm.use_kl_in_reward=False \
  data.train_files="$VERL_DATA_DIR/text/train.parquet" \
  data.val_files="$VERL_DATA_DIR/text/test.parquet" \
  data.train_batch_size="$TRAIN_TASKS_PER_UPDATE" \
  data.val_batch_size="$VAL_TASKS" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  data.tokenizer="$TOKENIZER_PATH" \
  '+data.apply_chat_template_kwargs.enable_thinking=False' \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  "actor_rollout_ref.actor.checkpoint.contents=['model','optimizer','extra','hf_model']" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_PARAM_OFFLOAD:-true} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OPTIMIZER_OFFLOAD:-true} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.35} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-true} \
  actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE:-true} \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN" \
  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD:-true} \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.use_futile_penalty="$USE_FUTILE_PENALTY" \
  actor_rollout_ref.actor.futile_penalty_coef="$FUTILE_PENALTY_COEF" \
  actor_rollout_ref.actor.futile_penalty_cap_units="$FUTILE_PENALTY_CAP_UNITS" \
  actor_rollout_ref.actor.futile_sr_target="$FUTILE_SR_TARGET" \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps="$MAX_EPISODE_STEPS" \
  env.history_length="$HISTORY_LENGTH" \
  env.stage2_prompt="$STAGE2_PROMPT" \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset="$EVAL_DATASET" \
  env.rollout_log_dir="$ROLLOUT_LOG_DIR" \
  trainer.critic_warmup=0 \
  "trainer.logger=${LOGGER}" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TRAIN_STEPS" \
  trainer.total_training_steps="$TRAIN_STEPS" \
  trainer.val_before_train=True \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir="$CKPT_ROOT/$EXPERIMENT_NAME" \
  ray_init.num_cpus=${RAY_NUM_CPUS:-96} \
  "$@"
