# syntax=docker/dockerfile:1
# E2L Stage-2 training image (ALFWorld GRPO, 8x A100/H100 80GB single node).
#
#   /opt/verl-agent   langfengQ/verl-agent, pristine checkout at the pinned
#                     upstream commit 20bd331, with patches/verl-agent-stage2-dual-reward.patch
#                     applied as a working-tree diff (= the unpublished
#                     stage2-dual-reward change 559f9bd), installed into the
#                     base image's system python 3.12 (torch 2.8.0 /
#                     vllm 0.11.0 / flash-attn 2.8.3) = "env A" in
#                     docs/playbook.md.
#   /opt/venvs/rwml   isolated python 3.10 env for the RWML GRPO baseline
#                     (verl 0.4.1 / vllm 0.8.3 / torch 2.6.0) = "env B".
#
# Build:  docker build -f docker/Dockerfile -t e2l-train:latest .
# Use:    see docs/playbook.md.
FROM vllm/vllm-openai:v0.11.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=true \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    # Training scripts read these; they point at the baked-in envs.
    VERL_AGENT_DIR=/opt/verl-agent \
    VERL_PYTHON=python3 \
    VERL_VENV_PYTHON=/opt/venvs/rwml/bin/python

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: builds the RWML venv now and lets the remote friend recreate
# e2-learning's own environment (uv sync) at runtime if needed.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# verl-agent: pristine upstream checkout at the pinned commit plus the stage-2
# patch carried in this repository, applied as a working-tree diff by the same
# setup script used outside Docker (scripts/setup_verl_agent.sh).
COPY scripts/setup_verl_agent.sh /tmp/setup_verl_agent.sh
COPY patches/verl-agent-stage2-dual-reward.patch /tmp/patches/verl-agent-stage2-dual-reward.patch
RUN VERL_AGENT_DIR="$VERL_AGENT_DIR" \
    STAGE2_PATCH=/tmp/patches/verl-agent-stage2-dual-reward.patch \
    bash /tmp/setup_verl_agent.sh

# Env A: stage-2 GRPO (base image's system python 3.12).
COPY docker/requirements-stage2.txt /tmp/requirements-stage2.txt
RUN pip install --no-cache-dir -r /tmp/requirements-stage2.txt \
 && pip install --no-cache-dir --no-deps --no-build-isolation -e "$VERL_AGENT_DIR"

# Env B: RWML GRPO baseline (isolated python 3.10 venv).
COPY docker/requirements-rwml.txt /tmp/requirements-rwml.txt
RUN uv venv --python 3.10 /opt/venvs/rwml \
 && uv pip install --python /opt/venvs/rwml/bin/python -r /tmp/requirements-rwml.txt

WORKDIR /workspace
CMD ["bash"]
