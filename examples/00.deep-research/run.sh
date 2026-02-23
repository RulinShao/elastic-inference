#!/bin/bash
# =============================================================================
# Deep Research Agent — end-to-end example
#
# Usage:
#   bash examples/00.deep-research/run.sh
#
# Prerequisites:
#   - conda env: rl_verl (with vLLM, transformers, httpx)
#   - .env file with SERPER_API_KEY (and optionally JINA_API_KEY, S2_API_KEY)
#   - SLURM cluster access (h200 partition)
# =============================================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate rl_verl

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

MODEL="/checkpoint/maestro/models/gpt-oss-120b"
SCHEDULER_URL="http://localhost:8780"
OUTDIR="examples/00.deep-research/results"

echo "============================================================"
echo "  Deep Research Agent — Example"
echo "============================================================"

# --- Step 1: Start scheduler (skip if already running) ---
if curl -s "${SCHEDULER_URL}/health" >/dev/null 2>&1; then
    echo "[1/4] Scheduler already running."
else
    echo "[1/4] Starting scheduler..."
    python -m elastic_serving.scheduler \
        --model "$MODEL" \
        --engine vllm \
        --tensor-parallel-size 8 \
        --max-nodes 2 \
        --qos h200_lowest \
        --partition h200 \
        --account dream \
        --conda-env rl_verl \
        --port 8780 &
    SCHED_PID=$!
    echo "  Scheduler PID=$SCHED_PID"

    # Wait for scheduler
    for i in $(seq 1 30); do
        if curl -s "${SCHEDULER_URL}/health" >/dev/null 2>&1; then break; fi
        sleep 2
    done
fi

# --- Step 2: Wait for workers ---
echo "[2/4] Waiting for workers..."
for i in $(seq 1 120); do
    READY=$(curl -s "${SCHEDULER_URL}/cluster_status" 2>/dev/null | \
        python3 -c "import sys,json; print(json.load(sys.stdin).get('ready_workers',0))" 2>/dev/null || echo 0)
    if [ "$READY" -ge 1 ] 2>/dev/null; then
        echo "  $READY worker(s) ready."
        break
    fi
    echo "  Waiting... (${i}0s)"
    sleep 10
done

# --- Step 3: Verify tools ---
echo "[3/4] Verifying tools..."
python tests/test_tools.py 2>&1 | head -20
echo "  ... (tools OK)"

# --- Step 4: Generate trajectories ---
echo "[4/4] Generating trajectories on sample questions..."
mkdir -p "$OUTDIR"
python scripts/generate_trajectories.py \
    --scheduler-url "$SCHEDULER_URL" \
    --dataset sample \
    --num-samples 3 \
    --max-tool-calls 15 \
    --temperature 0.7 \
    --output-dir "$OUTDIR"

echo ""
echo "============================================================"
echo "  Done! Results in: $OUTDIR"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  # Interactive chat"
echo "  python scripts/chat.py --scheduler-url $SCHEDULER_URL"
echo ""
echo "  # WebShaper evaluation"
echo "  python scripts/eval_webshaper.py --num-samples 5 --num-trajectories 1"


