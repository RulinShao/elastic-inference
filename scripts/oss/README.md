# OSS Engine Workflow

The `oss` engine is different from the normal `vllm` / `sglang` serving path. Instead of exposing OpenAI-style `/v1/completions`, an OSS worker exposes [`/v1/oss/run_one`](../../elastic_serving/oss/server.py#L57).

`run_one` is a single-question trajectory API for the OSS runtime.

Request fields:
- `question`: the user question to answer
- `qid`: optional question id
- `reasoning_effort`: optional override for `high`, `medium`, or `low`

Response fields:
- `qid`: the resolved question id
- `messages`: the full trajectory messages produced by the OSS runtime

The worker-local OSS service is implemented in [`elastic_serving/oss/server.py`](../../elastic_serving/oss/server.py). That service calls [`OSSEngineRuntime.run_one()`](../../elastic_serving/oss/runtime.py#L125), which:
- creates a browser tool session keyed by `qid`
- injects the Harmony system prompt and tool configuration
- runs the model/tool loop until it reaches an assistant `final` message
- returns the full conversation as `messages`

By default, OSS browser search blocks substrings `huggingface` and `browsecomp` when collecting search results ([`elastic_serving/oss/server.py`](../../elastic_serving/oss/server.py#L13)).

## Scheduler using OSS

When the elastic scheduler is launched with `--engine oss`, worker jobs start [`elastic_serving/oss/server.py`](../../elastic_serving/oss/server.py) instead of a normal vLLM-compatible OpenAI server. The scheduler also exposes [`POST /v1/oss/run_one`](../../elastic_serving/scheduler.py#L617) and proxies that request to a ready OSS worker.

[`run_oss.py`](../run_oss.py) works with the scheduler at endpoint `http://<scheduler-host>:8780/v1/oss/run_one`, and the scheduler fans requests out to the worker pool as before.

## Scripts

[`launch_scheduler_oss.slurm`](./launch_scheduler_oss.slurm) submits the scheduler itself as a CPU SLURM job, assuming:
- project root: `/checkpoint/dream/rulin/elastic-serving`
- venv: `/checkpoint/dream/rulin/elastic-serving/envs/oss_env/.venv`
- model: `/checkpoint/maestro/models/gpt-oss-120b`
- worker QoS / partition / account: `h200_dream_high` / `h200` / `dream`

[`run_oss.sh`](./run_oss.sh) is the paired client-side runner. It is also hardcoded to use those SLURM configs above. You need to edit a few variables at the top of the file after scheduler runs, then do `run_oss.sh`.

[`eval_bc.py`](./eval_bc.py) evaluates a trajectory JSONL file with an OpenAI judge model. It matches runs to ground truth by question text, supports either a local JSONL/JSON ground-truth file or a Hugging Face dataset id, and writes results directly under the `--eval_dir` you provide.

## Run generation

1. Submit the scheduler job:

```bash
sbatch scripts/oss/launch_scheduler_oss.slurm
```

2. Wait for the scheduler job log to print a line like:

```bash
Scheduler URL: http://<scheduler-host>:8780
```

3. Open [`run_oss.sh`](./run_oss.sh) and set:

```bash
SCHEDULER_URL="http://<scheduler-host>:8780"
```

4. In the same file, choose the dataset and output directory you want. For example:

```bash
DATASET="rl-rag/drtulu_v2_bc_synthetic_v3_0318"
OUTPUT_DIR="${PROJECT_ROOT}/runs/oss_high_bc_syn_v3"
SPLIT="train"
```

5. Run the trajectory job:

```bash
bash scripts/oss/run_oss.sh
```

This runs [`run_oss.py`](../run_oss.py), which writes `trajectories.jsonl` in the output directory.

## Evaluation

After generation, run:

```bash
python scripts/oss/eval_bc.py \
  --input /checkpoint/dream/rulin/elastic-serving/runs/oss_high_bc_syn_v3/trajectories.jsonl \
  --eval_dir /checkpoint/dream/rulin/elastic-serving/evals/oss_high_bc_syn_v3 \
  --ground_truth rl-rag/drtulu_v2_bc_synthetic_v3_0318 \
  --ground_truth_split train \
  --num-threads 50
```

You can pass in ground truth with either a local file or another huggingface dataset.

The evaluator writes:
- `evaluation_summary.json` under `--eval_dir`
- `detailed_judge_results.csv` under `--eval_dir`
- one `*_eval.json` per trajectory row under `--eval_dir/eval_files/`

Evaluation uses GPT-4.1 by default, following BrowseComp evaluation, except confidence scores and calibration errors are not computed.
