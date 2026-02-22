#!/bin/bash
# =============================================================================
# Launch a single worker node manually (normally the scheduler does this
# automatically, but you can also submit manually for testing)
# =============================================================================
#
# Usage:
#   sbatch scripts/launch_worker.slurm <scheduler_url> <model> [engine] [tp_size]
# =============================================================================
#SBATCH --job-name=elastic-worker
#SBATCH --partition=h200
#SBATCH --qos=h200_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=logs/worker_%j.out
#SBATCH --error=logs/worker_%j.err

set -e

SCHEDULER_URL="${1:?Usage: sbatch launch_worker.slurm <scheduler_url> <model> [engine] [tp_size]}"
MODEL="${2:?Provide model name/path}"
ENGINE="${3:-vllm}"
TP_SIZE="${4:-1}"
BASE_PORT="${5:-8001}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda
if [ -n "$CONDA_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

DP_SIZE=$((8 / TP_SIZE))

echo "============================================================"
echo "Elastic Serving Worker"
echo "============================================================"
echo "Hostname:   $(hostname)"
echo "Scheduler:  ${SCHEDULER_URL}"
echo "Model:      ${MODEL}"
echo "Engine:     ${ENGINE}"
echo "TP size:    ${TP_SIZE}"
echo "DP size:    ${DP_SIZE}"
echo "Base port:  ${BASE_PORT}"
echo "SLURM Job:  ${SLURM_JOB_ID}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-all}"
echo "============================================================"

nvidia-smi || true

python -m elastic_serving.worker \
    --scheduler-url "${SCHEDULER_URL}" \
    --engine "${ENGINE}" \
    --model "${MODEL}" \
    --base-port "${BASE_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpus-per-node 8
