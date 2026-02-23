#!/usr/bin/env python3
"""Push eval results to HuggingFace Hub with per-question difficulty."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/webshaper_full")
    parser.add_argument("--repo-id", default="rl-rag/webshaper-gpt-oss-120b-260222")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    import datasets

    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    results_file = os.path.join(args.output_dir, "results.json")

    with open(traj_file) as f:
        trajs = [json.loads(line) for line in f]
    with open(results_file) as f:
        results = json.load(f)

    # Build lookups from results.json
    # (qid, traj_idx) → correct/explanation
    correctness = {}
    # qid → question-level accuracy (avg@k = difficulty measure)
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

        row = {
            "qid": qid,
            "traj_idx": traj_idx,
            "question": t.get("question", ""),
            "reference_answer": t.get("reference_answer", ""),
            "boxed_answer": t.get("boxed_answer", ""),
            "correct": judge["correct"],
            "judge_explanation": judge["judge_explanation"],
            "question_accuracy": question_accuracy.get(qid, 0.0),
            "num_tool_calls": t.get("num_tool_calls", 0),
            "tool_calls": json.dumps(t.get("tool_calls", [])),
            "latency_s": t.get("latency_s", 0),
            "status": t.get("status", ""),
        }

        # Include conversation if saved
        conv = t.get("conversation")
        if conv:
            row["conversation"] = json.dumps(conv)

        rows.append(row)

    print(f"Rows: {len(rows)}")
    print(f"Correct: {sum(1 for r in rows if r['correct'])}/{len(rows)}")
    print(f"With conversation: {sum(1 for r in rows if 'conversation' in r)}/{len(rows)}")
    print(f"Question accuracy distribution:")
    accs = [r["question_accuracy"] for r in rows]
    for bucket in [0.0, 0.25, 0.5, 0.75, 1.0]:
        count = sum(1 for a in accs if a == bucket)
        print(f"  {bucket:.0%}: {count} trajectories")

    ds = datasets.Dataset.from_list(rows)
    print(f"\nPushing to {args.repo_id}...")
    ds.push_to_hub(args.repo_id, private=args.private)
    print("Done!")


if __name__ == "__main__":
    main()

