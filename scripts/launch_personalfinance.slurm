#!/bin/bash
#SBATCH --job-name=pf-sched
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/pf_scheduler_%j.log

source /opt/conda/etc/profile.d/conda.sh
conda activate rl_verl
cd /checkpoint/dream/rulin/elastic-serving
export PYTHONPATH=".:$PYTHONPATH"

SCHED_HOST=$(hostname -I | awk '{print $1}')
echo "Scheduler running on $SCHED_HOST:8780"

python -m elastic_serving.scheduler \
    --model /checkpoint/maestro/models/gpt-oss-120b \
    --port 8780 \
    --host 0.0.0.0 \
    --engine vllm \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-nodes 2 \
    --qos h200_dream_high --partition h200 --account dream \
    --conda-env rl_verl

