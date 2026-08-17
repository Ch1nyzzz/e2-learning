#!/usr/bin/env bash
# After Qwen2.5-7B online training: evaluate all checkpoints (incl. final)
# on GPU 0/1, then start the RWML pipeline. nohup-standalone so it survives
# controller session restarts.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== CHAIN_Q25: eval backlog first ==="
EVAL_GPUS=0,1 EVAL_PORT=29500 bash scripts/run_ckpt_eval_daemon_q25.sh >> runs/online_ckpt_eval_daemon.log 2>&1
echo "=== CHAIN_Q25: EVAL_BACKLOG_DONE, launching RWML pipeline ==="
bash scripts/rwml/run_rwml_pipeline.sh >> runs/rwml_pipeline.log 2>&1
echo "=== CHAIN_Q25_ALL_DONE ==="
