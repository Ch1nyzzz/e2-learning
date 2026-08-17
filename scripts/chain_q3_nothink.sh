#!/usr/bin/env bash
# Wait for the running thinking-mode eval worker (pid $1) to exit, then start
# the non-thinking eval script $2 on the same GPU pair.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wait_pid="$1"
next_script="$2"
while kill -0 "${wait_pid}" 2>/dev/null; do sleep 60; done
exec bash "${next_script}"
