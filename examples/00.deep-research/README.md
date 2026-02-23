# Example: Deep Research Agent

This example walks through setting up the elastic inference cluster and running a deep research agent that can search the web, read pages, and query academic papers.

## Prerequisites

```bash
# Conda environment with vLLM
conda activate rl_verl

# Project on PYTHONPATH
cd /checkpoint/dream/rulin/elastic-serving
export PYTHONPATH="$(pwd):$PYTHONPATH"

# API keys in .env (at project root)
cat .env
# SERPER_API_KEY=...       (required: web search)
# JINA_API_KEY=...         (optional: better URL reading)
# S2_API_KEY=...           (optional: higher Semantic Scholar rate limit)
# OPENAI_API_KEY=...       (optional: LLM judge for eval)
```

## Step 1: Launch the Serving Cluster

```bash
# Start scheduler — greedily acquires H200 nodes on low priority
python -m elastic_serving.scheduler \
    --model /checkpoint/maestro/models/gpt-oss-120b \
    --engine vllm \
    --tensor-parallel-size 8 \
    --max-nodes 2 \
    --qos h200_lowest \
    --partition h200 \
    --account dream \
    --conda-env rl_verl \
    --port 8780

# Check status (in another terminal)
curl http://localhost:8780/cluster_status | python -m json.tool
```

Wait until `ready_workers > 0`.

## Step 2: Interactive Chat

```bash
python scripts/chat.py --scheduler-url http://localhost:8780 --verbose
```

Example session:
```
You ❯ What is DR Tulu?
────────────────────────────────────────────────────────────
  🔧 browser.search({"query": "DR Tulu deep research model"})
  │ [1] Searched for "DR Tulu deep research model"
  │ ...
  🔧 browser.open({"id": 1})
  │ [2]
  │ L1: DR Tulu: Reinforcement Learning with Evolving Rubrics ...
  │ ...
  📝 Answer:

  DR Tulu (Deep Research Tulu) is an open-source deep research
  agent trained with reinforcement learning ...
────────────────────────────────────────────────────────────
```

Commands: `/clear`, `/verbose`, `/system`, `/quit`

## Step 3: Generate Trajectories

Generate research trajectories on sample questions:

```bash
python scripts/generate_trajectories.py \
    --scheduler-url http://localhost:8780 \
    --dataset sample \
    --num-samples 5 \
    --max-tool-calls 15 \
    --temperature 0.7 \
    --output-dir examples/00.deep-research/results
```

Output: `examples/00.deep-research/results/trajectories_sample.jsonl`

Each line is a JSON object:
```json
{
  "qid": 1,
  "question": "What were the key findings of the most recent IPCC report?",
  "answer": "The IPCC AR6 report found that ...",
  "reasoning": "I need to search for the latest IPCC report ...",
  "num_tool_calls": 5,
  "tool_calls": [
    {"round": 1, "tool": "browser.search", "args": {"query": "IPCC AR6 key findings"}, "result_len": 3200},
    {"round": 2, "tool": "browser.open", "args": {"id": 1}, "result_len": 28000},
    ...
  ],
  "latency_s": 45.2,
  "status": "success"
}
```

## Step 4: Evaluate on WebShaper

Run the full evaluation benchmark (500 multi-hop questions):

```bash
# Quick test (1 question, 1 trajectory)
python scripts/eval_webshaper.py \
    --scheduler-url http://localhost:8780 \
    --num-samples 1 --num-trajectories 1 \
    --output-dir examples/00.deep-research/eval_test

# Full run (500 questions × 4 trajectories, pass@4)
python scripts/eval_webshaper.py \
    --scheduler-url http://localhost:8780 \
    --num-samples 500 --num-trajectories 4 \
    --max-tool-calls 50 --concurrency 8 \
    --output-dir examples/00.deep-research/eval_full
```

Output files:
- `trajectories.jsonl` — all trajectories with tool calls
- `results.json` — pass@k, per-question judgments, aggregate stats

## Tools Available

The agent has 5 tools across two namespaces:

| Tool | What it does |
|------|-------------|
| `browser.search(query)` | Web search via Serper |
| `browser.open(id)` | Read a page via Jina Reader |
| `browser.find(pattern)` | Find text in opened page |
| `paper_search(query)` | Academic papers via Semantic Scholar (`mode="snippets"` for content, `mode="papers"` for metadata) |
| `pubmed_search(query)` | Biomedical literature via PubMed |

## Verify Tools Work (No LLM)

```bash
python tests/test_tools.py
```

This calls each tool directly and prints the raw output.

