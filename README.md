# Elastic Inference

Run vLLM/SGLang inference servers elastically on low-priority SLURM resources, with built-in deep research agent tools (web search, paper search, PubMed).

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

A background thread continuously `sbatch`es up to `max_nodes`. Each SLURM job launches vLLM/SGLang instances with disjoint GPU slices, registers with the scheduler, and sends heartbeats. Preempted nodes are detected and replaced automatically. Clients see a single OpenAI-compatible endpoint.

## Quick Start

```bash
# Start scheduler — auto-acquires up to 2 H200 nodes on low priority
python -m elastic_serving.scheduler \
    --model /path/to/model \
    --tensor-parallel-size 8 --max-nodes 2

# Use from Python
from openai import OpenAI
client = OpenAI(base_url="http://SCHEDULER_HOST:8780/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="/path/to/model",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Check cluster status
python -m elastic_serving.client status
```

## Deep Research Agent

Built-in agentic tool-use framework using the model's native Harmony token format (`builtin_tools=["browser"]`). The agent can search the web, read pages, search academic papers, and query biomedical literature — all via tool calls within a single generation loop.

### Tools

| Tool | Backend | Namespace | Use for |
|------|---------|-----------|---------|
| `browser.search` | Serper | built-in | Web search |
| `browser.open` | Jina Reader | built-in | Read web pages |
| `browser.find` | local | built-in | Find text in opened page |
| `paper_search` | Semantic Scholar | custom | Academic papers (metadata or body snippets) |
| `pubmed_search` | PubMed/NCBI | custom | Biomedical literature |

See [`elastic_serving/dr_utils/README.md`](elastic_serving/dr_utils/README.md) for detailed tool reference.

### Interactive Chat

```bash
# Chat with tool use
python scripts/chat.py --scheduler-url http://localhost:8780

# With verbose mode (shows token counts, timing)
python scripts/chat.py --verbose --max-tool-calls 20
```

### Evaluation (WebShaper)

```bash
# Set API keys
echo "SERPER_API_KEY=..." >> .env
echo "JINA_API_KEY=..." >> .env
echo "S2_API_KEY=..." >> .env

# Test with 1 question
python scripts/eval_webshaper.py --num-samples 1 --num-trajectories 1

# Full eval: 500 questions × 4 trajectories, pass@4 with LLM judge
python scripts/eval_webshaper.py \
    --num-samples 500 --num-trajectories 4 \
    --max-tool-calls 50 --concurrency 8
```

### Trajectory Generation

```bash
python scripts/generate_trajectories.py \
    --dataset sample --num-samples 5 --max-tool-calls 15
```

## Parallelism: DP + TP

Each node gets 8 GPUs exclusively and runs `gpus_per_node / tp_size` independent server instances:

| TP | DP/node | 16 nodes total |
|----|---------|----------------|
| 1  | 8       | 128 workers    |
| 4  | 2       | 32 workers     |
| 8  | 1       | 16 workers     |

vLLM prefix caching is enabled by default for efficient multi-round agentic conversations.

## Configuration

```bash
python -m elastic_serving.scheduler \
    --model /path/to/model \
    --engine vllm \
    --qos h200_lowest --partition h200 --account dream \
    --max-nodes 16 --tensor-parallel-size 4 \
    --conda-env rl_verl --port 8780
```

Or load from JSON: `--config config.json`.

## Dashboard

Live monitoring of cluster status and eval job progress:

```bash
python scripts/dashboard.py                  # auto-refresh with colors
python scripts/dashboard.py --once           # print once
python scripts/dashboard.py --interval 5     # faster refresh
```

![Live Dashboard](assets/dashboard.png)

## Project Structure

```
elastic_serving/
├── scheduler.py          # FastAPI scheduler + SLURM acquirer + OpenAI proxy
├── worker.py             # Per-node daemon: starts DP vLLM/SGLang instances
├── client.py             # CLI + Python client helpers
├── config.py             # Data models and SchedulerConfig
├── tools.py              # Harmony format parsing, prompt building, re-exports
└── dr_utils/             # Deep research tools and prompts
    ├── tools.py           # BrowserSession, paper_search, pubmed_search
    ├── prompts.py         # System prompt, model identity
    └── README.md          # Tool reference
scripts/
├── chat.py               # Interactive CLI chat with tool use
├── eval_webshaper.py     # WebShaper evaluation (pass@k with LLM judge)
├── generate_trajectories.py  # Trajectory generation
├── test_load.py          # Load testing
└── launch_scheduler.sh   # Shell launcher
tests/
├── test_tools.py         # Tool integration tests
└── test_prefix_caching.py  # Prefix caching benchmark
```

## Install

```bash
pip install -e ".[client]"   # for openai client helper
# vllm/sglang should already be in your conda env
```
