#!/bin/bash
#SBATCH --job-name=dr-sft
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1600G
#SBATCH --time=48:00:00
#SBATCH --output=sft/logs/dr-sft-%j.out
#SBATCH --error=sft/logs/dr-sft-%j.err
# ──────────────────────────────────────────────────────────────────────────────
# Fine-tune Qwen3.5 on deep research tool-calling trajectories
# using elastic-serving tools (web_search, open_url, find_text,
# paper_search, pubmed_search, python).
#
# Usage:
#   sbatch sft/train.sh                              # default config
#   sbatch sft/train.sh sft/qwen35-9b-dr-sft.yaml   # custom config
#
# Masking (automatic via multi-turn sharegpt format):
#   system, human, observation → masked (not trained on)
#   function_call, gpt         → trained on
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_FACTORY_DIR="${SCRIPT_DIR}/LLaMA-Factory"

mkdir -p "${SCRIPT_DIR}/logs"

# Activate your conda env (adjust path as needed)
source /opt/conda/etc/profile.d/conda.sh
conda activate llamafactory

# Load .env if present (e.g. for WANDB_API_KEY, HF_TOKEN)
if [ -f "${LLAMA_FACTORY_DIR}/.env" ]; then
    source "${LLAMA_FACTORY_DIR}/.env"
    export $(grep -v '^#' "${LLAMA_FACTORY_DIR}/.env" | xargs)
fi

export WANDB_PROJECT="${WANDB_PROJECT:-dr-sft}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"

# Install flash-attn on the GPU node if not present (needs CUDA for compilation)
python -c "import flash_attn" 2>/dev/null || {
    echo "Installing flash-attn (source build on GPU node)..."
    pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -5
}

CONFIG="${1:-${SCRIPT_DIR}/qwen35-9b-dr-sft.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# Resolve to absolute path before cd into LLaMA-Factory
CONFIG="$(realpath "$CONFIG")"

echo "============================================================"
echo "  LLaMA-Factory SFT — elastic-serving deep research"
echo "  Config: $CONFIG"
echo "  LLaMA-Factory dir: $LLAMA_FACTORY_DIR"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  Node:   ${SLURM_NODELIST:-$(hostname)}"
echo "  GPUs:   ${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
echo "============================================================"

cd "${LLAMA_FACTORY_DIR}"
llamafactory-cli train "$CONFIG"
