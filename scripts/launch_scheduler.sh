#!/bin/bash
# =============================================================================
# Launch the Elastic Serving Scheduler (on login/head node — no GPU needed)
# =============================================================================
#
# Usage:
#   bash scripts/launch_scheduler.sh --model <model> [options]
#
# Examples:
#   # Serve Llama-70B with TP=4 (DP=2 per node), up to 16 H200 nodes
#   bash scripts/launch_scheduler.sh \
#       --model meta-llama/Llama-3-70B-Instruct \
#       --tensor-parallel-size 4
#
#   # Serve smaller model with full DP (TP=1, DP=8 per node)
#   bash scripts/launch_scheduler.sh \
#       --model meta-llama/Llama-3-8B-Instruct \
#       --tensor-parallel-size 1 --max-nodes 8
#
#   # Use SGLang
#   bash scripts/launch_scheduler.sh \
#       --model Qwen/Qwen3-32B --engine sglang \
#       --tensor-parallel-size 4
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate conda if specified
if [ -n "$CONDA_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Elastic Serving Scheduler"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Arguments: $@"
echo "============================================================"

python -m elastic_serving.scheduler \
    --project-root "$PROJECT_ROOT" \
    --log-dir "${PROJECT_ROOT}/logs" \
    "$@"
