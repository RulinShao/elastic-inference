import argparse
import csv
import datasets
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
""".strip()


def load_ground_truth(
    source: str,
    split: str,
) -> Dict[str, Dict[str, str]]:
    gt_by_question: Dict[str, Dict[str, str]] = {}

    if source.endswith(".json") or source.endswith(".jsonl"):
        with Path(source).open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = list(datasets.load_dataset(source, split=split))

    for obj in rows:
        question = obj.get("query", obj.get("question"))
        if question is None:
            raise KeyError("Ground truth rows must contain query/question")

        record = {
            "question": str(question),
            "answer": str(obj["answer"]),
        }
        gt_by_question[record["question"].strip()] = record

    return gt_by_question


def parse_judge_response(judge_response: str) -> dict:
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "parse_error": False,
    }

    if not judge_response:
        result["parse_error"] = True
        return result

    answer_match = re.search(
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not answer_match:
        answer_match = re.search(
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not answer_match:
        answer_match = re.search(
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if answer_match:
        result["extracted_final_answer"] = answer_match.group(1).strip()

    reasoning_match = re.search(
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        judge_response,
        re.IGNORECASE | re.DOTALL,
    )
    if not reasoning_match:
        reasoning_match = re.search(
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if not reasoning_match:
        reasoning_match = re.search(
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
            judge_response,
            re.IGNORECASE | re.DOTALL,
        )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_match = re.search(
        r"\*\*correct:\*\*\s*(yes|no)",
        judge_response,
        re.IGNORECASE,
    )
    if not correct_match:
        correct_match = re.search(
            r"\*\*correct\*\*:\s*(yes|no)",
            judge_response,
            re.IGNORECASE,
        )
    if not correct_match:
        correct_match = re.search(
            r"correct:\s*(yes|no)",
            judge_response,
            re.IGNORECASE,
        )
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    return "\n".join(part for part in text_parts if part)


def extract_response_from_messages(messages: List[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("channel") == "final":
            return extract_text_content(msg.get("content"))

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return extract_text_content(msg.get("content"))

    return ""


def extract_tool_call_counts_from_messages(messages: List[dict]) -> Dict[str, int]:
    tool_counts: Dict[str, int] = defaultdict(int)

    for msg in messages:
        if msg.get("role") == "tool" and msg.get("name"):
            tool_counts[str(msg["name"])] += 1

    return dict(tool_counts)


def load_input_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.is_file() or input_path.suffix != ".jsonl":
        raise ValueError("Input path must be a trajectory .jsonl file")

    records: List[Dict[str, Any]] = []
    seen_source_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            run_data = json.loads(line)
            if "messages" not in run_data:
                raise ValueError(
                    f"{input_path}:{line_idx} is missing 'messages'"
                )

            qid = run_data.get("qid", run_data.get("query_id"))
            if qid is None:
                raise ValueError(
                    f"{input_path}:{line_idx} is missing qid/query_id"
                )

            traj_idx = run_data.get("traj_idx")
            source_id = (
                f"{qid}__traj{traj_idx}" if traj_idx is not None else str(qid)
            )
            if source_id in seen_source_ids:
                raise ValueError(
                    f"Duplicate trajectory id {source_id} found at {input_path}:{line_idx}"
                )

            seen_source_ids.add(source_id)
            records.append(
                {
                    "source_id": source_id,
                    "run_data": run_data,
                }
            )

    return records


def save_detailed_csv(all_results: List[dict], output_dir: Path) -> Path:
    csv_path = output_dir / "detailed_judge_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "query_id",
            "predicted_answer",
            "correct_answer",
            "judge_correct",
            "is_completed",
            "parse_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            judge_result = result.get("judge_result", {})
            predicted_answer = judge_result.get("extracted_final_answer") or ""
            if not predicted_answer:
                response = result.get("response", "")
                predicted_answer = (
                    response[:200] + "..." if len(response) > 200 else response
                )

            writer.writerow(
                {
                    "query_id": result.get("query_id", ""),
                    "predicted_answer": predicted_answer,
                    "correct_answer": result.get("correct_answer", ""),
                    "judge_correct": judge_result.get("correct", ""),
                    "is_completed": result.get("is_completed", ""),
                    "parse_error": judge_result.get("parse_error", False),
                }
            )

    print(f"Detailed CSV results saved to {csv_path}")
    return csv_path


def evaluate_record(
    record: Dict[str, Any],
    eval_files_dir: Path,
    args: argparse.Namespace,
    client: openai.OpenAI,
    ground_truth_by_question: Dict[str, Dict[str, str]],
) -> tuple[str, Optional[dict]]:
    source_id = str(record["source_id"])
    run_data = record["run_data"]
    eval_path = eval_files_dir / f"{source_id}_eval.json"

    if eval_path.exists() and not args.force:
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                return "skipped", json.load(f)
        except Exception:
            pass

    question = run_data.get("question")
    if not isinstance(question, str):
        print(f"Missing question for {source_id}")
        return "unmatched", None

    gt_record = ground_truth_by_question.get(question.strip())
    if gt_record is None:
        print(f"No ground truth match for {source_id}")
        return "unmatched", None

    query_id = source_id

    response = extract_response_from_messages(run_data.get("messages", []))
    is_completed = run_data.get("status") == "success"
    tool_call_counts = extract_tool_call_counts_from_messages(
        run_data.get("messages", [])
    )

    if not response or not is_completed:
        result = {
            "query_id": query_id,
            "question": gt_record["question"],
            "response": response,
            "correct_answer": gt_record["answer"],
            "is_completed": is_completed,
            "judge_prompt": None,
            "judge_response": None,
            "judge_result": {
                "parse_error": True,
                "error": "Response incomplete or cannot be parsed",
            },
            "tool_call_counts": tool_call_counts,
            "model_info": {
                "judge_model": args.model,
                "max_output_tokens": args.max_output_tokens,
            },
        }
        with eval_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return "evaluated", result

    judge_prompt = GRADER_TEMPLATE.format(
        question=gt_record["question"],
        response=response,
        correct_answer=gt_record["answer"],
    )

    try:
        judge_response = client.responses.create(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
            input=judge_prompt,
        )
    except Exception as e:
        print(f"Error calling judge model for {source_id}: {e}")
        return "error", None

    judge_text = judge_response.output_text if hasattr(judge_response, "output_text") else ""
    result = {
        "query_id": query_id,
        "question": gt_record["question"],
        "response": response,
        "correct_answer": gt_record["answer"],
        "is_completed": is_completed,
        "judge_prompt": judge_prompt,
        "judge_response": judge_text,
        "judge_result": parse_judge_response(judge_text),
        "tool_call_counts": tool_call_counts,
        "model_info": {
            "judge_model": args.model,
            "max_output_tokens": args.max_output_tokens,
        },
    }

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return "evaluated", result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory JSONL responses using the OpenAI judge model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a trajectory JSONL file containing message-based OSS runs",
    )
    parser.add_argument(
        "--ground_truth",
        default="rl-rag/drtulu_v2_bc_synthetic_v3_0318",
        help="Ground truth source: local JSONL/JSON path or Hugging Face dataset id",
    )
    parser.add_argument(
        "--ground_truth_split",
        default="train",
        help="Split to use when --ground_truth is a Hugging Face dataset id",
    )
    parser.add_argument(
        "--eval_dir",
        default="./evals",
        help="Directory to store evaluation results",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="OpenAI model for judging",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=1024,
        help="Maximum output tokens for the judge model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation of existing files",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel judge threads to use",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    eval_dir = Path(args.eval_dir)

    if not input_path.exists():
        raise ValueError(f"Input path {input_path} does not exist")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1")

    if (
        args.ground_truth.endswith(".json")
        or args.ground_truth.endswith(".jsonl")
    ) and not Path(args.ground_truth).is_file():
        raise ValueError(f"Ground truth file {args.ground_truth} does not exist")

    ground_truth_by_question = load_ground_truth(
        args.ground_truth,
        args.ground_truth_split,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_files_dir = eval_dir / "eval_files"
    eval_files_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluations will be saved to {eval_dir}")

    input_records = load_input_records(input_path)
    if not input_records:
        print(f"No evaluable records found in {input_path}")
        return

    print(f"Found {len(input_records)} trajectory records to evaluate")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment")
    client = openai.OpenAI(api_key=api_key)

    all_results: List[dict] = []
    skipped = 0

    if args.num_threads == 1:
        for record in tqdm(input_records, desc="Evaluating"):
            status, result = evaluate_record(
                record,
                eval_files_dir,
                args,
                client,
                ground_truth_by_question,
            )
            if status == "skipped":
                skipped += 1
            if result is not None:
                all_results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = [
                executor.submit(
                    evaluate_record,
                    record,
                    eval_files_dir,
                    args,
                    client,
                    ground_truth_by_question,
                )
                for record in input_records
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating",
            ):
                status, result = future.result()
                if status == "skipped":
                    skipped += 1
                if result is not None:
                    all_results.append(result)

    print(f"\nProcessed {len(all_results)} evaluations ({skipped} skipped)")
    if not all_results:
        print("No results to analyze")
        return

    total = len(all_results)
    correct_count = sum(
        1 for r in all_results if r.get("judge_result", {}).get("correct", False)
    )
    accuracy_percent = round((correct_count / total) * 100.0, 2)

    all_tool_counts: Dict[str, float] = defaultdict(float)
    for result in all_results:
        for tool_name, count in result.get("tool_call_counts", {}).items():
            all_tool_counts[tool_name] += count
    for tool_name in list(all_tool_counts.keys()):
        all_tool_counts[tool_name] /= total

    per_query_metrics = [
        {
            "query_id": result.get("query_id"),
            "correct": bool(result.get("judge_result", {}).get("correct", False)),
        }
        for result in all_results
    ]

    summary = {
        "accuracy_percent": accuracy_percent,
        "num_results": total,
        "num_correct": correct_count,
        "avg_tool_stats": dict(all_tool_counts),
        "judge_model": args.model,
        "evaluation_date": datetime.now().date().isoformat(),
        "per_query_metrics": per_query_metrics,
    }

    summary_path = eval_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Evaluated {total} responses:")
    print(f"Accuracy: {accuracy_percent:.2f}%")
    print(f"Average Tool Calls: {dict(all_tool_counts)}")
    print(f"\nSummary saved to {summary_path}")

    save_detailed_csv(all_results, eval_dir)


if __name__ == "__main__":
    main()
