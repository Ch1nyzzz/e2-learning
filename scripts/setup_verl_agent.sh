#!/usr/bin/env bash
# Set up verl-agent for E2L stage-2 training WITHOUT forking or modifying
# upstream: clone langfengQ/verl-agent at a pinned upstream commit, then apply
# the stage-2 patch carried in this repository as an uncommitted working-tree
# diff. The checkout stays AT the upstream commit -- `git diff` inside
# $VERL_AGENT_DIR always shows exactly our delta, and `git checkout . &&
# git clean -fd` restores pristine upstream at any time.
#
# Idempotent: safe to re-run on an already-prepared checkout.
#
# Configurable via env:
#   VERL_AGENT_DIR          target checkout (default: $HOME/verl-agent)
#   VERL_AGENT_REPO         upstream remote (default: langfengQ/verl-agent)
#   VERL_AGENT_BASE_COMMIT  pinned upstream commit the patch applies to
#   STAGE2_PATCH            patch file (default: <this repo>/patches/verl-agent-stage2-dual-reward.patch)
#   INSTALL_EDITABLE        "true" to `pip install --no-deps -e` afterwards (default: false)
#   VERL_PYTHON             interpreter for the editable install (default: python3)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_AGENT_DIR=${VERL_AGENT_DIR:-$HOME/verl-agent}
VERL_AGENT_REPO=${VERL_AGENT_REPO:-https://github.com/langfengQ/verl-agent.git}
VERL_AGENT_BASE_COMMIT=${VERL_AGENT_BASE_COMMIT:-20bd331bdbc9026a5668e11362178e10ab7400c8}
STAGE2_PATCH=${STAGE2_PATCH:-$REPO_ROOT/patches/verl-agent-stage2-dual-reward.patch}
INSTALL_EDITABLE=${INSTALL_EDITABLE:-false}
VERL_PYTHON=${VERL_PYTHON:-python3}
# Commit the patch was taken from (informational; the patch itself is the
# source of truth, since that commit exists on no public remote).
STAGE2_COMMIT=559f9bdf173dfa0c289e5b8c906c2fa44cf6e4c2

[[ -f "$STAGE2_PATCH" ]] || { echo "missing patch: $STAGE2_PATCH" >&2; exit 1; }

if [[ ! -d "$VERL_AGENT_DIR/.git" ]]; then
  echo "=== cloning $VERL_AGENT_REPO -> $VERL_AGENT_DIR"
  git clone "$VERL_AGENT_REPO" "$VERL_AGENT_DIR"
fi

head_sha=$(git -C "$VERL_AGENT_DIR" rev-parse HEAD)

if [[ "$head_sha" == "$STAGE2_COMMIT" ]]; then
  echo "=== already at stage2 commit $STAGE2_COMMIT (dev checkout); nothing to do"
elif git -C "$VERL_AGENT_DIR" apply --reverse --check "$STAGE2_PATCH" 2>/dev/null; then
  echo "=== stage2 patch already applied on upstream base; nothing to do"
else
  echo "=== checking out upstream base $VERL_AGENT_BASE_COMMIT"
  git -C "$VERL_AGENT_DIR" checkout --quiet "$VERL_AGENT_BASE_COMMIT"
  # Refuse to clobber unrelated local changes.
  git -C "$VERL_AGENT_DIR" diff --quiet \
    || { echo "working tree at base commit is dirty; clean it up first" >&2; exit 1; }
  echo "=== applying $STAGE2_PATCH"
  git -C "$VERL_AGENT_DIR" apply "$STAGE2_PATCH"
  # Files newly created by the patch are untracked and would be invisible to
  # `git diff`; register them with intent-to-add so the full delta (and any
  # future `git checkout . && git clean -fd` restore) covers them.
  new_files=()
  while IFS=$'\t' read -r _ _ p; do
    git -C "$VERL_AGENT_DIR" cat-file -e "HEAD:$p" 2>/dev/null || new_files+=("$p")
  done < <(git -C "$VERL_AGENT_DIR" apply --numstat "$STAGE2_PATCH")
  [[ ${#new_files[@]} -eq 0 ]] || git -C "$VERL_AGENT_DIR" add -N "${new_files[@]}"
fi

if [[ "$INSTALL_EDITABLE" == "true" ]]; then
  echo "=== $VERL_PYTHON -m pip install --no-deps -e $VERL_AGENT_DIR"
  "$VERL_PYTHON" -m pip install --no-deps --no-build-isolation -e "$VERL_AGENT_DIR"
fi

echo "=== verl-agent ready: upstream $VERL_AGENT_BASE_COMMIT + stage2 patch (working-tree diff)"
