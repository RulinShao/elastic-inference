#!/bin/bash
#SBATCH --job-name=oss-sched
#SBATCH --partition=cpu
#SBATCH --qos=cpu_lowest
#SBATCH --account=dream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/checkpoint/dream/rulin/elastic-serving/results/oss_scheduler_%j.log

PROJECT_ROOT="/checkpoint/dream/rulin/elastic-serving"
OSS_VENV="${PROJECT_ROOT}/envs/oss_env/.venv"
MODEL="/checkpoint/maestro/models/gpt-oss-120b"
PORT=8780
TP_SIZE=8
MAX_NODES=2
GPU_MEMORY_UTILIZATION=0.9
REASONING_EFFORT=high
WORKER_QOS="h200_dream_high"
WORKER_PARTITION="h200"
WORKER_ACCOUNT="dream"

cd "${PROJECT_ROOT}"
export PYTHONPATH=".:${PYTHONPATH:-}"
source "${OSS_VENV}/bin/activate"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/results"

SCHED_HOST=$(hostname -I | awk '{print $1}')
SCHEDULER_URL="http://${SCHED_HOST}:${PORT}"

echo "============================================================"
echo "Elastic OSS Scheduler (dream cluster)"
echo "============================================================"
echo "Scheduler URL: ${SCHEDULER_URL}"
echo "Model: ${MODEL}"
echo "Engine: oss"
echo "Tensor parallel size: ${TP_SIZE}"
echo "Max nodes: ${MAX_NODES}"
echo "GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Reasoning effort: ${REASONING_EFFORT}"
echo "Worker QoS: ${WORKER_QOS}"
echo "Worker partition: ${WORKER_PARTITION}"
echo "Worker account: ${WORKER_ACCOUNT}"
echo "Worker venv: ${OSS_VENV}"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo
echo "Then edit scripts/oss/run_oss.sh and set:"
echo "  SCHEDULER_URL=\"${SCHEDULER_URL}\""
echo "Then run:"
echo "  bash scripts/oss/run_oss.sh"
echo "============================================================"

python -m elastic_serving.scheduler \
    --model "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --engine oss \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-nodes "${MAX_NODES}" \
    --qos "${WORKER_QOS}" \
    --partition "${WORKER_PARTITION}" \
    --account "${WORKER_ACCOUNT}" \
    --venv-path "${OSS_VENV}" \
    --project-root "${PROJECT_ROOT}" \
    --log-dir "${PROJECT_ROOT}/logs" \
    --reasoning-effort "${REASONING_EFFORT}"
