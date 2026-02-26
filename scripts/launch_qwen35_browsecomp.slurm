#!/bin/bash
#SBATCH --job-name=qwen35-sched
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/qwen35_scheduler_%j.log

source /opt/conda/etc/profile.d/conda.sh
conda activate qwen35
cd /checkpoint/dream/rulin/elastic-serving
export PYTHONPATH=".:$PYTHONPATH"

SCHED_HOST=$(hostname -I | awk '{print $1}')
echo "Scheduler running on $SCHED_HOST:8780"

# Serve Qwen3.5-27B (dense, text-only, 128K context)
# Workers are 1 H200 node with TP=8
python -m elastic_serving.scheduler \
    --model Qwen/Qwen3.5-27B \
    --port 8780 \
    --host 0.0.0.0 \
    --engine vllm \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-nodes 2 \
    --qos h200_dream_high --partition h200 --account dream \
    --conda-env qwen35 \
    --engine-extra-args "--language-model-only --gpu-memory-utilization 0.9"

