#!/usr/bin/env bash
set -euo pipefail

# Plain GRPO on ALFWorld with Qwen3-8B in non-thinking mode, adapted for
# 4x A100 80GB from train_policy_grpo_8xa100_80gb.sh. Uses the verl-agent
# checkout at VERL_AGENT_DIR (default $HOME/verl-agent); native thinking is
# disabled via the chat-template kwarg (the prompt-level <think></think> tags
# from verl-agent's ALFWorld prompt remain, matching the GiGPO recipe).
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_PATH}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_AGENT_DIR=${VERL_AGENT_DIR:-$HOME/verl-agent}
ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
CKPT_ROOT=${CKPT_ROOT:-$REPO_ROOT/checkpoints/e2l_policy_grpo}
VERL_PYTHON=${VERL_PYTHON:-.venv/bin/python}
GPUS=${GPUS:-0,1,2,3}
N_GPUS=${N_GPUS:-4}
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
  data.train_files="${VERL_DATA_DIR:-$HOME/data/verl-agent}/text/train.parquet" \
  data.val_files="${VERL_DATA_DIR:-$HOME/data/verl-agent}/text/test.parquet" \
  data.train_batch_size="$TRAIN_TASKS_PER_UPDATE" \
  data.val_batch_size="$VAL_TASKS" \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.35} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps="$MAX_EPISODE_STEPS" \
  env.history_length=2 \
  env.rollout.n="$GROUP_SIZE" \
  env.resources_per_worker.num_cpus=0.1 \
  env.alfworld.eval_dataset="$EVAL_DATASET" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name=e2l_policy_grpo \
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
  ray_init.num_cpus=96 \
  "$@"
