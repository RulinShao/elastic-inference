#!/usr/bin/env python3
"""Push eval results to HuggingFace Hub with metrics README."""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CONTAM_MARKERS = ["Alibaba-NLP/WebShaper", "huggingface.co/datasets"]


def build_readme(results, rows, repo_id):
    """Generate a dataset card README with metrics."""
    model = results.get("model", "unknown").split("/")[-1]
    dataset = results.get("dataset", results.get("split", "unknown"))
    split = results.get("split", "")
    k = results.get("num_trajectories_per_q", 4)
    n_q = results.get("num_questions", 0)
    n_traj = results.get("total_trajectories", len(rows))
    judge = results.get("judge_model", "gpt-4o")
    blocked = results.get("blocked_domains") or []

    pass_k = results.get(f"pass@{k}", 0)
    avg_k = results.get(f"avg@{k}", 0)
    traj_acc = results.get("trajectory_accuracy", 0)
    correct = results.get("correct_trajectories", 0)
    avg_tools = results.get("avg_tool_calls", 0)

    # Tool distribution from rows (only valid tools, not hallucinated)
    VALID_TOOLS = {
        "browser.search", "browser.open", "browser.find",
        "functions.paper_search", "functions.pubmed_search",
    }
    tool_counts = Counter()
    for r in rows:
        for tc in json.loads(r.get("tool_calls", "[]")):
            tool = tc.get("tool", "")
            if tool in VALID_TOOLS:
                tool_counts[tool] += 1
    total_tools = sum(tool_counts.values())

    tool_table = ""
    for tool, cnt in tool_counts.most_common():
        pct = cnt / max(total_tools, 1) * 100
        tool_table += f"| `{tool}` | {cnt:,} | {pct:.0f}% |\n"

    # Contamination
    n_contam = sum(1 for r in rows if r.get("contaminated", False))
    contam_note = ""
    if n_contam:
        contam_note = f"\n⚠️ **Contamination**: {n_contam}/{n_traj} trajectories flagged (search results contained evaluation dataset pages). Filter with `ds.filter(lambda x: not x['contaminated'])`.\n"

    has_conv = sum(1 for r in rows if r.get("conversation"))

    readme = f"""---
tags:
- deep-research
- tool-use
- evaluation
---

# {repo_id.split('/')[-1]}

Deep research agent evaluation on **{dataset}**{f' ({split} split)' if split else ''}.

## Results

| Metric | Value |
|--------|-------|
| **pass@{k}** | **{pass_k:.1%}** |
| **avg@{k}** | **{avg_k:.1%}** |
| Trajectory accuracy | {traj_acc:.1%} ({correct}/{n_traj}) |
| Questions | {n_q} |
| Trajectories | {n_traj} ({k} per question) |
| Avg tool calls | {avg_tools:.1f} |
| Full conversations | {'✅' if has_conv else '❌'} |

## Model & Setup

| | |
|---|---|
| **Model** | `{model}` |
| **Judge** | `{judge}` |
| **Max tool calls** | {results.get('max_tool_calls', 50)} |
| **Temperature** | {results.get('temperature', 0.7)} |
| **Blocked domains** | {', '.join(blocked) if blocked else 'None'} |

## Tool Usage

| Tool | Calls | % |
|------|------:|--:|
{tool_table}
**Total: {total_tools:,} tool calls** ({avg_tools:.1f} per trajectory)
{contam_note}
## Columns

| Column | Description |
|--------|-------------|
| `qid` | Question ID |
| `traj_idx` | Trajectory index (0-{k-1}) |
| `question` | Input question |
| `reference_answer` | Ground truth answer |
| `boxed_answer` | Model's extracted `\\boxed{{}}` answer |
| `correct` | GPT-4o judge verdict |
| `judge_explanation` | Judge's reasoning |
| `question_accuracy` | Fraction of trajectories correct for this question (difficulty: 0=hardest, 1=easiest) |
| `num_tool_calls` | Number of tool calls in trajectory |
| `tool_calls` | Tool call log (JSON) |
| `conversation` | Full trajectory with reasoning + tool responses (JSON) |
| `contaminated` | Whether search results contained evaluation dataset pages |
| `latency_s` | Generation time in seconds |
"""
    return readme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/webshaper_full")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    import datasets
    from huggingface_hub import HfApi

    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    results_file = os.path.join(args.output_dir, "results.json")

    with open(traj_file) as f:
        trajs = [json.loads(line) for line in f]
    with open(results_file) as f:
        results = json.load(f)

    # Build lookups
    correctness = {}
    question_accuracy = {}
    for q in results.get("per_question", []):
        qid = q["qid"]
        question_accuracy[qid] = q.get("accuracy", 0.0)
        for a in q.get("answers", []):
            correctness[(qid, a.get("traj_idx", 0))] = {
                "correct": a.get("correct", False),
                "judge_explanation": a.get("explanation", ""),
            }

    rows = []
    for t in trajs:
        qid = t.get("qid", "")
        traj_idx = t.get("traj_idx", 0)
        judge = correctness.get(
            (qid, traj_idx), {"correct": False, "judge_explanation": ""}
        )

        # Check contamination
        contaminated = False
        conv = t.get("conversation") or []
        for turn in (conv if isinstance(conv, list) else []):
            if turn.get("role") == "tool":
                content = turn.get("content", "")
                if any(m in content for m in CONTAM_MARKERS):
                    contaminated = True
                    break

        row = {
            "qid": qid,
            "traj_idx": traj_idx,
            "question": t.get("question", ""),
            "reference_answer": t.get("reference_answer", ""),
            "boxed_answer": t.get("boxed_answer", ""),
            "correct": judge["correct"],
            "judge_explanation": judge["judge_explanation"],
            "question_accuracy": question_accuracy.get(qid, 0.0),
            "contaminated": contaminated,
            "num_tool_calls": t.get("num_tool_calls", 0),
            "tool_calls": json.dumps(t.get("tool_calls", [])),
            "latency_s": t.get("latency_s", 0),
            "status": t.get("status", ""),
        }

        conv_data = t.get("conversation")
        if conv_data:
            row["conversation"] = json.dumps(conv_data)

        rows.append(row)

    print(f"Rows: {len(rows)}")
    print(f"Correct: {sum(1 for r in rows if r['correct'])}/{len(rows)}")
    print(f"With conversation: {sum(1 for r in rows if 'conversation' in r)}/{len(rows)}")
    print(f"Contaminated: {sum(1 for r in rows if r['contaminated'])}/{len(rows)}")

    # Push dataset
    ds = datasets.Dataset.from_list(rows)
    print(f"\nPushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id, private=args.private)

    # Push README
    readme = build_readme(results, rows, args.repo_id)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    print("Done!")


if __name__ == "__main__":
    main()
