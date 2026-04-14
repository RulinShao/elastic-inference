#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Fine-tune Qwen models on deep research tool-calling trajectories
# using LLaMA-Factory with elastic-serving tools.
#
# Usage:
#   # On a SLURM cluster (adapt SBATCH directives to your cluster):
#   sbatch sft/train.sh sft/qwen35-9b-sft.example.yaml
#
#   # Local multi-GPU:
#   bash sft/train.sh sft/qwen35-9b-sft.example.yaml
#
# The training config should use dataset_dir pointing to sft/data/
# with datasets registered in sft/data/dataset_info.json.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_FACTORY_DIR="${SCRIPT_DIR}/LLaMA-Factory"

# Activate your environment (adjust to your setup)
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate llamafactory

# Install flash-attn if needed
python -c "import flash_attn" 2>/dev/null || {
    echo "Installing flash-attn (source build)..."
    pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -5
}

CONFIG="${1:-${SCRIPT_DIR}/qwen35-9b-sft.example.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

CONFIG="$(realpath "$CONFIG")"

echo "============================================================"
echo "  LLaMA-Factory SFT — elastic-serving deep research"
echo "  Config: $CONFIG"
echo "============================================================"

cd "${LLAMA_FACTORY_DIR}"
llamafactory-cli train "$CONFIG"
