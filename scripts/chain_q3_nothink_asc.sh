#!/usr/bin/env bash
# Wait for the in-flight nothink base in-distribution eval (already running,
# detached from its original driver script) to finish writing its summary,
# give the accelerate processes a moment to release GPU 0/1, then start the
# ascending nothink worker (which skips base-ID via its existence check).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
summary="outputs/sr_eval_10k/online_q3_nothink_base_eval_in_distribution.jsonl.summary.json"
until [ -f "${summary}" ]; do sleep 30; done
sleep 90
exec bash scripts/eval_q3_nothink_gpu01_asc.sh
