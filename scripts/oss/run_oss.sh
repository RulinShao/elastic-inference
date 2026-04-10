#!/bin/bash

PROJECT_ROOT="/checkpoint/dream/rulin/elastic-serving"

# Copy the URL printed by scripts/oss/launch_scheduler_oss.slurm.
SCHEDULER_URL="http://REPLACE_ME:8780"

# Pick the dataset/output pair you want to run.
DATASET="rl-rag/drtulu_v2_bc_synthetic_v3_0318"
OUTPUT_DIR="${PROJECT_ROOT}/runs/oss_high_bc_syn_v3"
SPLIT="train"

NUM_SAMPLES=-1
NUM_TRAJECTORIES=1
CONCURRENCY=32
MAX_RETRIES=5
REASONING_EFFORT="high"

if [[ "${SCHEDULER_URL}" == "http://REPLACE_ME:8780" ]]; then
    echo "Set SCHEDULER_URL at the top of scripts/oss/run_oss.sh before running." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

python scripts/run_oss.py \
    --scheduler-url "${SCHEDULER_URL}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --concurrency "${CONCURRENCY}" \
    --max-retries "${MAX_RETRIES}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume
