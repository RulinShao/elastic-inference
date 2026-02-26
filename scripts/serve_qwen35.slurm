#!/bin/bash
#SBATCH --job-name=qwen35-vllm
#SBATCH --partition=h200
#SBATCH --qos=h200_dream_high
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/qwen35_serve_%j.log

source /opt/conda/etc/profile.d/conda.sh
conda activate qwen35

HOST_IP=$(hostname -I | awk '{print $1}')
echo "======================================"
echo "vLLM serving Qwen3.5-27B on $HOST_IP:8001"
echo "======================================"

vllm serve Qwen/Qwen3.5-27B \
    --port 8001 \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --language-model-only \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0

