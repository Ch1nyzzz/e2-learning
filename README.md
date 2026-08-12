# Experience Learning

This repository implements the first controlled version of **environment-centric,
mistake-driven experience learning** in the deterministic ALFWorld TextWorld environment.

The training loop is intentionally not GRPO/PPO:

1. enumerate ALFWorld's admissible actions;
2. greedily predict the next observation for every action while measuring the model's mean
   next-token entropy;
3. choose the action with maximum mean token entropy;
4. execute that action in the real environment exactly once;
5. compare the selected action's greedy prediction with the real observation using a constrained
   OpenAI-compatible semantic judge;
6. perform a full-parameter next-observation update only when reality contradicts the model.

The external judge is only a semantic equivalence checker. ALFWorld remains the source of the
training target.

## Current scope

- ALFWorld `AlfredTWEnv`, all six supported task families
- train / valid-seen / valid-unseen split support
- token-entropy, semantic-sample-entropy, and random acquisition strategies
- mistake-only and all-transition update gates for the 2x2 ablation
- full-parameter Hugging Face training; LoRA/PEFT parameters are explicitly rejected
- Accelerate FSDP2 configurations for 2 or 4 x A100 80GB
- sharded model/optimizer/scheduler/RNG checkpoints and complete JSONL transition logs

Phase 1 evaluates world-model learning and interaction efficiency. Task-success claims need a
separately fixed planner, so they are deliberately not conflated with this first experiment.

## Vast.ai setup

Use an Ubuntu image with two or four visible A100 80GB GPUs, enough system RAM for model loading, and a
persistent volume. Python 3.11 is used because it is compatible with ALFWorld/TextWorld.

```bash
export ALFWORLD_DATA=/workspace/data/alfworld
cp .env.example .env
# Fill JUDGE_API_KEY, JUDGE_BASE_URL, and the provider-specific JUDGE_MODEL.

bash scripts/setup_vastai.sh
set -a && source .env && set +a
# Choose the script matching the visible GPU count.
bash scripts/train_2xa100_80gb.sh
# bash scripts/train_4xa100_80gb.sh
```

The provisional model is `Qwen/Qwen3-8B`, pinned to an immutable Hugging Face revision. Override
both fields together if a different base model is chosen:

```bash
bash scripts/train_2xa100_80gb.sh \
  --set model.name=YOUR_MODEL \
  --set model.revision=COMMIT_HASH \
  --set experiment.max_environment_steps=100
```

For a no-API infrastructure smoke run, use `--set judge.provider=exact_match`. It is not a valid
scientific substitute for semantic judging.

The primary comparison is a `2 x 2` factorial under the same environment-action budget and game
seed:

| Acquisition | Update gate | Role |
| --- | --- | --- |
| `random` | `all_transitions` | uniform data + ordinary next-observation SFT |
| `random` | `mistake_only` | isolates mistake gating |
| `token_entropy` | `all_transitions` | isolates active acquisition |
| `token_entropy` | `mistake_only` | proposed method |

Set the two fields with `--set acquisition.strategy=...` and
`--set training.update_gate=...`, and give every run a distinct `experiment.output_dir`. Random
acquisition still generates every candidate so model-generation compute remains comparable, but it
skips acquisition-time semantic clustering and its API cost.

Token-entropy acquisition performs one greedy prediction per candidate action, averages the full
vocabulary entropy over valid generated tokens, and reuses the selected action's prediction for the
reality check. The external judge is therefore called only after the selected action is executed.
Each candidate log also records `generated_tokens` and `hit_token_limit`, making it possible to
measure how often generation reaches `generation.max_new_tokens` without emitting EOS.
The older `semantic_entropy` strategy remains available as a higher-cost ablation; it samples
`generation.samples_per_action` outcomes and uses the judge to cluster every pair before selection.

## Reproducibility and safety boundaries

- Only rank 0 owns ALFWorld and the API judge; all ranks follow synchronized phases.
- The hand-coded ALFWorld expert is disabled. We use the environment's admissible candidate set.
- Candidate sampling never clones, rewinds, or steps the environment.
- A judge `UNCERTAIN` result does not update in the mistake-only condition.
- Prompt/history/action tokens are masked with `-100`; loss is applied only to the real next
  observation and EOS.
- Invalid judge output, timeout exhaustion, or a rank-0 environment error is broadcast as a stop
  instead of being silently treated as a correct prediction.
- ALFWorld game paths are sorted before registration to avoid filesystem-dependent episode order.

Runtime artifacts are written under the configured output directory:

```text
events.jsonl
resolved_config.json
judge_cache.sqlite3
checkpoints/
  env_step_000100/
  final/
```

## Held-out transition evaluation

Create fixed probe sets before training so every method/seed is evaluated on identical
environment transitions:

```bash
uv run --extra alfworld experience-learning collect-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --split eval_in_distribution --episodes 50 \
  --output data/probes_valid_seen.jsonl

uv run --extra alfworld experience-learning collect-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --split eval_out_of_distribution --episodes 50 \
  --output data/probes_valid_unseen.jsonl
```

Evaluate a sharded checkpoint with the same four-GPU launch configuration:

```bash
uv run --extra train --extra alfworld accelerate launch \
  --config_file configs/accelerate/fsdp_4xa100_80gb.yaml \
  -m experience_learning.cli evaluate-probes \
  --config configs/alfworld_qwen3_8b.yaml \
  --checkpoint outputs/alfworld_qwen3_8b/checkpoints/final \
  --probes data/probes_valid_unseen.jsonl \
  --output outputs/alfworld_qwen3_8b/eval_valid_unseen.jsonl
```

The summary reports conservative semantic accuracy (uncertain counts as incorrect), a semantic
accuracy upper bound, definite-decision accuracy, judge coverage, and target-observation token NLL.
Retain the per-transition JSONL to audit judge errors.

## Local checks

The tests do not download a language model or ALFWorld data:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

ALFWorld follows a batch-size-one API: `reset()` returns observations and infos; `step()` accepts
a one-element action list. The adapter exposes an ordinary single-state interface to the
experiment controller.
