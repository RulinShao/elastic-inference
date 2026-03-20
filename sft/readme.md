# SFT Pipeline for Deep Research Tool Calling (Qwen3.5)

Fine-tune Qwen3.5 on multi-turn tool-calling trajectories using
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and the
search tools defined in `elastic-serving`:

| Tool | Backend | Description |
|------|---------|-------------|
| `web_search` | Serper API | Web search with titles, URLs, snippets |
| `open_url` | Jina Reader | Open and read webpages |
| `find_text` | local | Find text patterns in opened pages |
| `paper_search` | Semantic Scholar | Academic paper search (metadata + snippets) |
| `pubmed_search` | NCBI PubMed | Biomedical literature search |
| `python` | Jupyter kernel | Stateful Python code execution |

## Quick Start

```bash
# 1. Set up LLaMA-Factory (one-time)
git submodule add https://github.com/hiyouga/LLaMA-Factory.git sft/LLaMA-Factory
conda create -n llamafactory python=3.11 -y
conda activate llamafactory
cd sft/LLaMA-Factory && pip install -e . && cd ../..
pip install deepspeed wandb flash-attn --no-build-isolation

# 2. Convert DR Tulu v1 data to Qwen3.5 format
python sft/convert_drtulu.py

# 3. Train
sbatch sft/train.sh
```

### Alternative: generate your own trajectories

```bash
python sft/generate_trajectories.py \
    --base-url http://GPU_NODE:8001 \
    --model Qwen/Qwen3.5-27B \
    --dataset data/browsecomp.jsonl
python sft/reformat_sft_data.py \
    --input sft/trajectories/trajectories.jsonl
# Then update qwen35-27b-dr-sft.yaml: dataset: dr_sft_multiturn
sbatch sft/train.sh
```

## Environment Setup

### 1. Initialize the LLaMA-Factory submodule

```bash
cd /checkpoint/dream/rulin/elastic-serving
git submodule add https://github.com/hiyouga/LLaMA-Factory.git sft/LLaMA-Factory
```

Or if already added:

```bash
git submodule update --init --recursive
```

### 2. Create the `llamafactory` conda environment

```bash
conda create -n llamafactory python=3.11 -y
conda activate llamafactory

cd /checkpoint/dream/rulin/elastic-serving/sft/LLaMA-Factory
pip install -e .

pip install deepspeed wandb flash-attn --no-build-isolation
```

### 3. API keys for trajectory generation

The trajectory generation script requires API keys for the tool backends.
Create a `.env` file in the repo root:

```bash
SERPER_API_KEY=your_serper_key      # required for web_search
JINA_API_KEY=your_jina_key          # optional, improves open_url quality
S2_API_KEY=your_s2_key              # optional, higher rate limits for paper_search
```

### 4. HuggingFace authentication (if using private datasets)

```bash
huggingface-cli login
```

## Step 1: Generate Trajectories

Use a running Qwen3.5 vLLM server (via `elastic-serving` or standalone) to
generate tool-calling trajectories:

```bash
# Start vLLM server (if not already running)
sbatch scripts/serve_qwen35.slurm

# Generate with sample questions
python sft/generate_trajectories.py \
    --base-url http://GPU_NODE:8001 \
    --model Qwen/Qwen3.5-27B \
    --dataset sample --num-samples 5

# Generate from a JSONL dataset
python sft/generate_trajectories.py \
    --base-url http://GPU_NODE:8001 \
    --model Qwen/Qwen3.5-27B \
    --dataset data/browsecomp.jsonl \
    --concurrency 8 --output-dir sft/trajectories

# Generate with Python code execution enabled
python sft/generate_trajectories.py \
    --base-url http://GPU_NODE:8001 \
    --enable-python --dataset sample
```

The script saves full trajectories (including raw model outputs with
`<think>` reasoning and `<tool_call>` blocks, plus tool responses) to
`sft/trajectories/trajectories.jsonl`.

The script supports resumption — re-running skips already-completed IDs.

### Input dataset format

JSONL with at least `id` and `question` fields:

```json
{"id": "q001", "question": "What is the Mamba state space model?"}
{"id": "q002", "question": "Latest GLP-1 agonist clinical trials?"}
```

## Step 2: Reformat for LLaMA-Factory

Convert the raw trajectories to LLaMA-Factory's multi-turn ShareGPT format:

```bash
python sft/reformat_sft_data.py \
    --input sft/trajectories/trajectories.jsonl \
    --min-tool-calls 1
```

This produces `sft/data/dr-sft-trajectories-multiturn.json` with the
following role structure:

| Role | Side | Trained on? |
|------|------|-------------|
| `system` | prompt | No (masked) |
| `human` | prompt | No (masked) |
| `observation` | prompt | No (masked) |
| `function_call` | response | **Yes** |
| `gpt` | response | **Yes** |

Each `function_call` turn contains the model's reasoning (`<think>` block)
and tool call (JSON in `<tool_call>` tags). Each `observation` turn contains
the raw tool response. The model learns to generate tool calls and final
answers while the tool outputs are provided as context.

## Step 3: Train

### Submit SLURM job (8×H200)

```bash
cd /checkpoint/dream/rulin/elastic-serving
sbatch sft/train.sh
```

### Custom config

```bash
sbatch sft/train.sh sft/my_custom_config.yaml
```

### Monitor training

```bash
# SLURM logs
tail -f /checkpoint/dream/rulin/elastic-serving/sft/logs/dr-sft-<JOB_ID>.out

# wandb (project: dr-sft)
```

### Default training config (`qwen35-27b-dr-sft.yaml`)

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3.5-27B` |
| Method | Full fine-tuning, DeepSpeed ZeRO-3 |
| Template | `qwen3` (ChatML with tool calling + reasoning) |
| Sequence length | 16,384 |
| Batch size | 1 per device × 16 grad accum × 8 GPUs = 128 effective |
| Learning rate | 1e-5, cosine schedule, 10% warmup |
| Epochs | 3 |
| Checkpoints | `/checkpoint/dream/rulin/elastic-serving/sft/ckpts/qwen35-27b-dr-sft` |

## Using DR Tulu v1 Data

The default training config uses the DR Tulu v1 dataset
([rl-rag/dr-tulu-sft-unified](https://huggingface.co/datasets/rl-rag/dr-tulu-sft-unified)),
converted from its original tool format to Qwen3.5's native tool-calling protocol.

### Conversion details

The `convert_drtulu.py` script handles:

| DR Tulu v1 Tool | Qwen3.5 Tool | Calls |
|---|---|---|
| `serper_google_webpage_search` | `web_search` | 18,380 |
| `serper_fetch_webpage_content` | `open_url` | 3,274 |
| `semantic_scholar_snippet_search` | `paper_search` | 20,049 |
| `search_papers_by_relevance` | `paper_search` | 13 |

- **Tool calls**: converted from `tool_name(key=val)` to Qwen3.5 JSON `<tool_call>[{"name": "...", "arguments": {...}}]</tool_call>`
- **Tool outputs**: reformatted from raw Serper/S2 JSON to human-readable text (preserving `snippet_id` for `<cite>` references)
- **Citations**: `<cite id="S_xxx,S_yyy">claim text</cite>` tags preserved as-is
- **System prompt**: replaced with elastic-serving's research assistant prompt
- **Think blocks**: `<think>reasoning</think>` preserved in both `function_call` and `gpt` turns

### Filter by type

```bash
# Only long-form and short-form (exclude exact_answer)
python sft/convert_drtulu.py --types long_form,short_form

# With custom token length limit
python sft/convert_drtulu.py --max-token-length 8192
```

### Datasets available in `dataset_info.json`

| Name | Source | Description |
|------|--------|-------------|
| `drtulu_qwen35` | DR Tulu v1 converted | 11K examples with web_search, open_url, paper_search |
| `dr_sft_multiturn` | Self-generated | Trajectories from `generate_trajectories.py` |

To use both, set in the YAML: `dataset: drtulu_qwen35,dr_sft_multiturn`

## File Structure

```
sft/
├── LLaMA-Factory/              # Git submodule (LLaMA-Factory framework)
├── data/
│   ├── dataset_info.json       # Dataset registration for LLaMA-Factory
│   ├── drtulu-qwen35-multiturn.json  # DR Tulu v1 converted to Qwen3.5
│   └── dr-tulu-sft-source.jsonl      # Cached source data from HF
├── trajectories/               # Generated trajectory data (gitignored)
│   └── trajectories.jsonl      # Raw trajectories from generate step
├── ckpts/                      # Training checkpoints (gitignored)
├── logs/                       # SLURM job logs (gitignored)
├── convert_drtulu.py           # Convert DR Tulu v1 → Qwen3.5 format
├── generate_trajectories.py    # Generate new trajectories with tools
├── reformat_sft_data.py        # Convert generated trajectories for LLaMA-Factory
├── qwen35-27b-dr-sft.yaml     # Training config
├── train.sh                    # SLURM launch script
└── readme.md                   # This file
```

## Tool Format Details

During trajectory generation, the Qwen3 adapter produces tool calls in
XML format:

```xml
<think>
I need to search for information about the Mamba model.
</think>

<tool_call>
<function=web_search>
<parameter=query>Mamba state space model architecture</parameter>
</function>
</tool_call>
```

The reformat script converts these to JSON format for LLaMA-Factory:

```
<think>
I need to search for information about the Mamba model.
</think>

<tool_call>
[{"name": "web_search", "arguments": {"query": "Mamba state space model architecture"}}]
</tool_call>
```

Tool responses are stored as `observation` turns. The Qwen3 template
automatically wraps them in `<tool_response>` tags during tokenization.
