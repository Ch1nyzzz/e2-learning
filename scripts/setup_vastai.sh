#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun." >&2
  exit 1
fi

uv python install 3.11
uv sync --python 3.11 --extra train --extra alfworld

if [[ -z "${ALFWORLD_DATA:-}" ]]; then
  echo "Set ALFWORLD_DATA to a persistent volume before downloading ALFWorld." >&2
  exit 1
fi

uv run --extra alfworld alfworld-download --data-dir "${ALFWORLD_DATA}"
