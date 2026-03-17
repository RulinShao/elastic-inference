#!/bin/bash
#SBATCH --job-name=oss-env-sched
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/oss_env_scheduler_%j.log

OSS_VENV=/checkpoint/dream/rulin/elastic-serving/envs/oss_env/.venv

cd /checkpoint/dream/rulin/elastic-serving
export PYTHONPATH=".:$PYTHONPATH"
source $OSS_VENV/bin/activate

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
    --venv-path $OSS_VENV
