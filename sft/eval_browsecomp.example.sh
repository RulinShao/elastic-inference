#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Serve a model with vLLM, run BrowseComp eval (pass@1), then shut down.
#
# Usage:
#   bash sft/eval_browsecomp.example.sh Qwen/Qwen3.5-9B results/bc_base
#   bash sft/eval_browsecomp.example.sh sft/ckpts/qwen35-9b-sft results/bc_sft
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:?Usage: eval_browsecomp.sh MODEL_PATH OUTPUT_DIR}"
OUTPUT_DIR="${2:?Usage: eval_browsecomp.sh MODEL_PATH OUTPUT_DIR}"
PORT=8001

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${PROJ_DIR}/${OUTPUT_DIR}"

# Activate your environment
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate qwen35

HOST_IP=$(hostname -I | awk '{print $1}')
echo "============================================================"
echo "  BrowseComp Eval (pass@1)"
echo "  Model: $MODEL"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"

# Start vLLM server
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve "$MODEL" \
    --port $PORT \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 &
VLLM_PID=$!

echo "Waiting for vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1 && { echo "Ready"; break; }
    sleep 10
done

if ! curl -s "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null; exit 1
fi

cd "$PROJ_DIR"
python scripts/eval_generic.py \
    --scheduler-url "http://localhost:${PORT}" \
    --model "$MODEL" \
    --model-format qwen3 \
    --dataset data/browsecomp.jsonl \
    --split train \
    --num-trajectories 1 \
    --concurrency 16 \
    --max-tool-calls 150 \
    --temperature 0.7 \
    --output-dir "$OUTPUT_DIR" \
    --save-full-trajectories \
    --blocked-domains huggingface.co \
    --resume

echo "Done. Results in $OUTPUT_DIR"
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null || true
