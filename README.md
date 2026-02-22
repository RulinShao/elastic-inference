# Elastic Serving

Run vLLM/SGLang inference servers elastically on low-priority SLURM resources. The scheduler greedily acquires nodes up to a user-defined cap and automatically replaces preempted ones.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Scheduler  (login node, no GPU)                                 │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ Node        │  │ Health       │  │ OpenAI-compatible      │  │
│  │ Acquirer    │  │ Monitor      │  │ Proxy (/v1/...)        │  │
│  │ (sbatch     │  │ (heartbeat   │  │ (round-robin across    │  │
│  │  greedy)    │  │  timeout →   │  │  all ready workers)    │  │
│  │             │  │  replace)    │  │                        │  │
│  └─────────────┘  └──────────────┘  └────────────────────────┘  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ HTTP
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Node 0       │   │ Node 1       │   │ Node N       │
│ 8× H200 GPU  │   │ 8× H200 GPU  │   │ 8× H200 GPU  │
│              │   │              │   │              │
│ DP0: vLLM    │   │ DP0: vLLM    │   │ DP0: vLLM    │
│  (GPU 0-3)   │   │  (GPU 0-3)   │   │  (GPU 0-3)   │
│ DP1: vLLM    │   │ DP1: vLLM    │   │ DP1: vLLM    │
│  (GPU 4-7)   │   │  (GPU 4-7)   │   │  (GPU 4-7)   │
└──────────────┘   └──────────────┘   └──────────────┘
     TP=4, DP=2 per node (configurable)
```

**Key idea:** a background thread continuously runs `sbatch` to fill up to `max_nodes`. Each SLURM job starts a worker that launches one or more vLLM/SGLang instances (DP shards) with disjoint GPU slices, registers with the scheduler, and sends heartbeats. If a node is preempted (heartbeat timeout), the scheduler cleans it up and the acquirer thread submits a replacement.

Clients see a single OpenAI-compatible endpoint at the scheduler URL. Requests are round-robin'd across all ready workers.

## Quick Start

```bash
# Start scheduler on login node — auto-acquires up to 16 H200 nodes
python -m elastic_serving.scheduler \
    --model meta-llama/Llama-3-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-nodes 16

# Use from Python (standard OpenAI client)
from openai import OpenAI
client = OpenAI(base_url="http://SCHEDULER_HOST:8780/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3-70B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Check cluster status
python -m elastic_serving.client status
```

## Parallelism: DP + TP

Each node gets 8 GPUs exclusively and runs `gpus_per_node / tensor_parallel_size` independent server instances:

| TP | DP/node | Workers/node | 16 nodes total |
|----|---------|-------------|----------------|
| 1  | 8       | 8 instances | 128 workers    |
| 2  | 4       | 4 instances | 64 workers     |
| 4  | 2       | 2 instances | 32 workers     |
| 8  | 1       | 1 instance  | 16 workers     |

## Configuration

Defaults are set for FAIR cluster H200 nodes (`h200_lowest` QoS, `dream` account). Override via CLI:

```bash
python -m elastic_serving.scheduler \
    --model /path/to/model \
    --engine vllm \          # or sglang
    --qos h200_lowest \
    --partition h200 \
    --account dream \
    --max-nodes 16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --conda-env rl_verl \
    --port 8780
```

Or load from JSON: `--config config.json`.

## Trajectory Generation

Included script for generating deep-research SFT trajectories using gpt-oss-120b with web search (Serper) and URL reading (Jina):

```bash
# Set API keys in .env
echo "SERPER_API_KEY=..." >> .env
echo "JINA_API_KEY=..." >> .env

python scripts/generate_trajectories.py \
    --scheduler-url http://localhost:8780 \
    --dataset sample \
    --num-samples 10 \
    --concurrency 4
```

The script uses `tokenizer.apply_chat_template` with native Harmony tokens to format tool definitions and tool calls, and generates via the `/v1/completions` endpoint. Tool calls are intercepted at `<|call|>` stop tokens, executed, and fed back through the chat template.

## Project Structure

```
elastic_serving/
├── scheduler.py    # FastAPI scheduler + SLURM node acquirer + OpenAI proxy
├── worker.py       # Per-node daemon: starts DP vLLM/SGLang instances
├── client.py       # CLI + Python client helpers
└── config.py       # Data models and SchedulerConfig
scripts/
├── generate_trajectories.py   # SFT trajectory generation with tool use
├── test_load.py               # Load testing
├── launch_scheduler.sh        # Shell launcher
├── launch_scheduler.slurm     # SLURM launcher for scheduler
└── launch_worker.slurm        # Manual worker launcher
```

## Install

```bash
pip install -e ".[client]"   # for openai client helper
# vllm/sglang should already be in your conda env
```

