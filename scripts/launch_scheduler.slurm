#!/bin/bash
# =============================================================================
# Launch Scheduler as a long-running SLURM job (CPU-only, no GPU needed)
# =============================================================================
#
# Usage:
#   sbatch scripts/launch_scheduler.slurm <model> [tp_size] [max_nodes] [engine]
#
# Examples:
#   sbatch scripts/launch_scheduler.slurm meta-llama/Llama-3-70B-Instruct 4 16 vllm
#   sbatch scripts/launch_scheduler.slurm Qwen/Qwen3-8B 1 8 sglang
# =============================================================================
#SBATCH --job-name=elastic-scheduler
#SBATCH --partition=devlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/scheduler_%j.log

set -e

MODEL="${1:?Usage: sbatch launch_scheduler.slurm <model> [tp_size] [max_nodes] [engine]}"
TP_SIZE="${2:-1}"
MAX_NODES="${3:-16}"
ENGINE="${4:-vllm}"
PORT="${5:-8780}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate conda
if [ -n "$CONDA_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p "${PROJECT_ROOT}/logs"

# Write service discovery file
SERVICE_FILE="${PROJECT_ROOT}/.scheduler_url"
SCHEDULER_URL="http://$(hostname):${PORT}"
echo "${SCHEDULER_URL}" > "${SERVICE_FILE}"

DP_SIZE=$((8 / TP_SIZE))

echo "============================================================"
echo "Elastic Serving Scheduler (SLURM)"
echo "============================================================"
echo "Host:       $(hostname)"
echo "Port:       ${PORT}"
echo "Model:      ${MODEL}"
echo "Engine:     ${ENGINE}"
echo "TP size:    ${TP_SIZE}"
echo "DP/node:    ${DP_SIZE}"
echo "Max nodes:  ${MAX_NODES}"
echo "Max GPUs:   $((MAX_NODES * 8))"
echo "Max workers: $((MAX_NODES * DP_SIZE))"
echo "SLURM Job:  ${SLURM_JOB_ID}"
echo "URL:        ${SCHEDULER_URL}"
echo "============================================================"

python -m elastic_serving.scheduler \
    --model "${MODEL}" \
    --engine "${ENGINE}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-nodes "${MAX_NODES}" \
    --port "${PORT}" \
    --project-root "${PROJECT_ROOT}" \
    --log-dir "${PROJECT_ROOT}/logs"
