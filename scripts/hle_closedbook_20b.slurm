#!/bin/bash
# =============================================================================
# HLE Closed-Book Evaluation — gpt-oss-20b
#
# This script:
#   1. Launches the elastic scheduler (which auto-acquires GPU nodes)
#   2. Waits for workers to become ready
#   3. Runs the closed-book eval on HLE
# =============================================================================
#SBATCH --job-name=hle-cb-20b
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/hle_closedbook_20b/eval_%j.log

set -e

# ---- Config ----
MODEL="/checkpoint/ram/jacklanchantin/pretrained-llms/gpt-oss-20b"
TP_SIZE=2
MAX_NODES=1
PORT=8781
OUTPUT_DIR="/checkpoint/dream/rulin/elastic-serving/results/hle_closedbook_20b"
DATASET="rl-rag/hle_text_only"
SPLIT="test"
NUM_TRAJECTORIES=1
CONCURRENCY=64
TEMPERATURE=0.7
MAX_GEN_TOKENS=16384

# ---- Setup ----
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen35
cd /checkpoint/dream/rulin/elastic-serving
export PYTHONPATH=".:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"

SCHED_HOST=$(hostname -I | awk '{print $1}')
SCHEDULER_URL="http://${SCHED_HOST}:${PORT}"
echo "============================================================"
echo "HLE Closed-Book Eval: gpt-oss-20b"
echo "============================================================"
echo "Scheduler:  ${SCHEDULER_URL}"
echo "Model:      ${MODEL}"
echo "TP:         ${TP_SIZE}"
echo "Max nodes:  ${MAX_NODES}"
echo "Output:     ${OUTPUT_DIR}"
echo "SLURM Job:  ${SLURM_JOB_ID}"
echo "============================================================"

# ---- Launch scheduler in background ----
python -m elastic_serving.scheduler \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --engine vllm \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-nodes "${MAX_NODES}" \
    --qos h200_dream_high --partition h200 --account dream \
    --conda-env qwen35 \
    &
SCHED_PID=$!
echo "Scheduler PID: ${SCHED_PID}"

# Wait for scheduler to start
sleep 30

# ---- Run closed-book eval ----
echo ""
echo "Starting closed-book evaluation..."
python scripts/eval_closedbook.py \
    --scheduler-url "${SCHEDULER_URL}" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --concurrency "${CONCURRENCY}" \
    --temperature "${TEMPERATURE}" \
    --max-gen-tokens "${MAX_GEN_TOKENS}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume

echo ""
echo "Evaluation complete! Results in ${OUTPUT_DIR}"
cat "${OUTPUT_DIR}/results.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
k = d.get('num_trajectories_per_q', 1)
print(f\"pass@{k}: {d.get(f'pass@{k}', 0):.1%}\")
print(f\"avg@{k}: {d.get(f'avg@{k}', 0):.1%}\")
print(f\"traj acc: {d.get('trajectory_accuracy', 0):.1%}\")
"

# Cleanup
kill $SCHED_PID 2>/dev/null || true
wait $SCHED_PID 2>/dev/null || true

