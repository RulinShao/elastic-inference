#!/usr/bin/env python3
"""
WebShaper Evaluation — Browser-Agentic Information Seeking
============================================================

Evaluates the model on Alibaba-NLP/WebShaper (500 multi-hop questions
requiring web browsing).  For each question, generates N trajectories
using Harmony-native browser tools (browser.search, browser.open,
browser.find), then judges correctness with an LLM and computes pass@k.

Usage:
    # Test with one prompt
    python scripts/eval_webshaper.py --scheduler-url http://localhost:8780 \\
        --num-samples 1 --num-trajectories 1

    # Full run: 4 trajectories per prompt, pass@4
    python scripts/eval_webshaper.py --scheduler-url http://localhost:8780 \\
        --num-samples 500 --num-trajectories 4 --concurrency 8

    # Resume from checkpoint
    python scripts/eval_webshaper.py --scheduler-url http://localhost:8780 \\
        --num-samples 500 --num-trajectories 4 --resume
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import dotenv
import httpx

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.tools import (
    STOP_TOKEN_IDS,
    STOP_TOKEN_IDS_NO_CALL,
    SYSTEM_PROMPT,
    BrowserSession,
    append_tool_round,
    build_initial_prompt,
    execute_custom_tool,
    extract_final_answer,
    parse_tool_call,
)

# =============================================================================
# Constants
# =============================================================================

MAX_TOOL_CALLS = 50
MAX_GEN_TOKENS = 8192
MAX_MODEL_LEN = 131072  # gpt-oss-120b max context
TEMPERATURE = 0.7

JUDGE_MODEL = "gpt-4o"
JUDGE_PROMPT = """\
You are an impartial judge evaluating whether a model's answer is correct.

**Question:** {question}

**Reference answer:** {reference}

**Model's answer:** {prediction}

Evaluate whether the model's answer is correct. The model's answer does NOT \
need to match the reference answer word-for-word — it just needs to convey \
the same factual information. Be lenient about formatting differences \
(e.g. "12000" vs "12,000", "Dr. Smith" vs "Smith").

Respond with a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""


# =============================================================================
# Trajectory generation
# =============================================================================


async def generate_trajectory(
    question: str,
    qid: str,
    base_url: str,
    model: str,
    tokenizer,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    traj_idx: int = 0,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_gen_tokens: int = MAX_GEN_TOKENS,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """Generate a single research trajectory for one question."""
    browser = BrowserSession(http_client)
    tool_call_count = 0
    tool_calls_log: List[Dict[str, Any]] = []
    conversation: List[Dict[str, Any]] = []  # full turn-by-turn log

    # Build initial prompt
    prompt = build_initial_prompt(tokenizer, user_message=question)
    prompt_tokens_init = len(tokenizer.encode(prompt))

    tag = f"qid={qid} t={traj_idx}"
    print(f"  [{tag}] Starting ({prompt_tokens_init} init tokens)")

    t0 = time.time()

    while True:
        at_limit = tool_call_count >= max_tool_calls
        stop_ids = STOP_TOKEN_IDS_NO_CALL if at_limit else STOP_TOKEN_IDS

        # Guard against context overflow
        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len + max_gen_tokens > MAX_MODEL_LEN:
            # Reduce gen tokens or force finish
            remaining = MAX_MODEL_LEN - prompt_len - 256
            if remaining < 256:
                print(f"  [{tag}] Context full ({prompt_len} tokens), forcing answer")
                stops = STOP_TOKENS_NO_CALL
                remaining = min(2048, MAX_MODEL_LEN - prompt_len - 64)
                if remaining <= 0:
                    break
            max_gen_tokens_this_round = remaining
        else:
            max_gen_tokens_this_round = max_gen_tokens

        try:
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_gen_tokens_this_round,
                    "temperature": temperature,
                    "stop_token_ids": stop_ids,
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=600,
            )
            if resp.status_code == 503:
                print(f"  [{tag}] 503 — waiting for workers...")
                await asyncio.sleep(10)
                continue
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["text"]
        except Exception as e:
            print(f"  [{tag}] Generation error: {e}")
            break

        # Parse tool call
        tool_call = parse_tool_call(raw_text) if not at_limit else None

        if tool_call:
            ns, tool_name, tool_args = tool_call
            tool_call_count += 1

            short_args = json.dumps(tool_args, ensure_ascii=False)[:100]
            print(f"  [{tag}] Tool {tool_call_count}/{max_tool_calls}: "
                  f"{ns}.{tool_name}({short_args})")

            if ns == "browser":
                result = await browser.execute(tool_name, tool_args)
            else:
                result = await execute_custom_tool(
                    tool_name, tool_args, http_client
                )
            tool_calls_log.append({
                "round": tool_call_count,
                "tool": f"{ns}.{tool_name}",
                "args": tool_args,
                "result_len": len(result),
            })
            conversation.append({
                "role": "assistant",
                "content": raw_text,
                "tool_call": {
                    "tool": f"{ns}.{tool_name}",
                    "args": tool_args,
                },
            })
            # Truncate very large tool results for storage
            result_stored = result[:30000] if len(result) > 30000 else result
            conversation.append({
                "role": "tool",
                "tool": f"{ns}.{tool_name}",
                "content": result_stored,
            })

            prompt = append_tool_round(
                prompt, raw_text, tool_name, result, namespace=ns
            )
            continue
        else:
            # Final answer
            _reasoning, answer = extract_final_answer(raw_text)

            # Extract \boxed{...} if present and clean LaTeX
            boxed_match = re.search(
                r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer
            )
            boxed_answer = ""
            if boxed_match:
                raw_boxed = boxed_match.group(1).strip()
                # Clean LaTeX: \text{X} → X, \, → ,, {,} → ,, etc.
                cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', raw_boxed)
                cleaned = re.sub(r'\\textrm\{([^}]*)\}', r'\1', cleaned)
                cleaned = re.sub(r'\\[,;!\s]', ' ', cleaned)
                cleaned = re.sub(r'\{,\}', ',', cleaned)
                cleaned = re.sub(r'[{}]', '', cleaned)
                cleaned = cleaned.replace('\\', '').strip()
                cleaned = re.sub(r'\s+', ' ', cleaned)
                boxed_answer = cleaned

            conversation.append({
                "role": "assistant",
                "content": raw_text,
                "final_answer": answer,
            })

            elapsed = time.time() - t0
            short = boxed_answer or answer[:100]
            print(f"  [{tag}] Done: {tool_call_count} tools, "
                  f"{elapsed:.1f}s, answer={short}")

            return {
                "qid": qid,
                "traj_idx": traj_idx,
                "question": question,
                "answer": answer,
                "boxed_answer": boxed_answer,
                "raw_output": raw_text,
                "num_tool_calls": tool_call_count,
                "tool_calls": tool_calls_log,
                "conversation": conversation,
                "latency_s": elapsed,
                "status": "success",
            }

    # Fallback if we broke out of the loop
    elapsed = time.time() - t0
    return {
        "qid": qid,
        "traj_idx": traj_idx,
        "question": question,
        "answer": "",
        "raw_output": "",
        "num_tool_calls": tool_call_count,
        "tool_calls": tool_calls_log,
        "conversation": conversation,
        "latency_s": elapsed,
        "status": "context_overflow",
    }


# =============================================================================
# LLM Judge
# =============================================================================


async def judge_answer(
    question: str,
    reference: str,
    prediction: str,
    judge_http: httpx.AsyncClient,
    judge_model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    """Use an LLM to judge if prediction matches reference."""
    if not prediction.strip():
        return {"correct": False, "explanation": "Empty answer"}

    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        # Fallback: exact match
        norm_ref = reference.strip().lower()
        norm_pred = prediction.strip().lower()
        match = norm_ref in norm_pred or norm_pred in norm_ref
        return {"correct": match, "explanation": "fallback exact match"}

    try:
        resp = await judge_http.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 256,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON from response
        # Handle cases where the model wraps JSON in markdown
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        return {
            "correct": bool(result.get("correct", False)),
            "explanation": result.get("explanation", ""),
        }
    except json.JSONDecodeError:
        # Try to extract correct/incorrect from text
        lower = content.lower()
        correct = '"correct": true' in lower or '"correct":true' in lower
        return {"correct": correct, "explanation": content[:200]}
    except Exception as e:
        return {"correct": False, "explanation": f"Judge error: {e}"}


# =============================================================================
# Main evaluation loop
# =============================================================================


async def run_evaluation(
    scheduler_url: str,
    model: str,
    tokenizer,
    num_samples: int,
    num_trajectories: int,
    concurrency: int,
    output_dir: str,
    max_tool_calls: int,
    temperature: float,
    resume: bool,
    judge_model: str,
):
    import datasets

    print(f"Loading WebShaper dataset...")
    ds = datasets.load_dataset("Alibaba-NLP/WebShaper", split="main")
    print(f"Dataset: {len(ds)} questions")

    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    print(f"Evaluating {len(ds)} questions × {num_trajectories} trajectories")

    os.makedirs(output_dir, exist_ok=True)
    traj_file = os.path.join(output_dir, "trajectories.jsonl")
    results_file = os.path.join(output_dir, "results.json")

    # Resume support — track by (qid, traj_idx) to avoid duplicates
    completed = {}  # (qid, traj_idx) -> result
    if resume and os.path.exists(traj_file):
        with open(traj_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r["qid"], r.get("traj_idx", 0))
                    completed[key] = r
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} trajectories already done")

    # Wait for workers
    base_url = scheduler_url.rstrip("/")
    async with httpx.AsyncClient() as tmp:
        for _ in range(120):
            try:
                resp = await tmp.get(f"{base_url}/cluster_status", timeout=5)
                status = resp.json()
                if status.get("ready_workers", 0) > 0:
                    print(f"Cluster: {status['ready_workers']} ready workers")
                    break
            except Exception:
                pass
            print("  Waiting for workers...")
            await asyncio.sleep(10)
        else:
            print("Timed out waiting for workers.")
            return

    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=600)
    judge_http = httpx.AsyncClient(timeout=60)

    sem = asyncio.Semaphore(concurrency)
    total_done = 0

    async def process_one(item, traj_idx):
        nonlocal total_done
        async with sem:
            qid = item["id"]
            # Skip if already done
            key = (qid, traj_idx)
            if key in completed:
                return completed[key]

            try:
                result = await generate_trajectory(
                    question=item["question"],
                    qid=qid[:8],
                    base_url=base_url,
                    model=model,
                    tokenizer=tokenizer,
                    http_client=http_client,
                    openai_http=openai_http,
                    traj_idx=traj_idx,
                    max_tool_calls=max_tool_calls,
                    temperature=temperature,
                )
                result["reference_answer"] = item["answer"]
            except Exception as e:
                result = {
                    "qid": qid,
                    "traj_idx": traj_idx,
                    "question": item["question"],
                    "answer": "",
                    "reference_answer": item["answer"],
                    "error": traceback.format_exc(),
                    "status": "error",
                    "latency_s": 0,
                }
            total_done += 1
            return result

    # Generate all trajectories
    print(f"\n{'='*60}")
    print(f"Generating trajectories (concurrency={concurrency})...")
    print(f"{'='*60}\n")

    all_results: List[Dict[str, Any]] = []
    tasks = []
    for item in ds:
        for t in range(num_trajectories):
            tasks.append((item, t))

    # Process and stream results to file
    async_tasks = [asyncio.create_task(process_one(item, t)) for item, t in tasks]

    with open(traj_file, "a") as writer:
        for coro in asyncio.as_completed(async_tasks):
            result = await coro
            all_results.append(result)
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()

            done_pct = total_done / len(tasks) * 100
            if total_done % 10 == 0 or total_done == len(tasks):
                print(f"[{total_done}/{len(tasks)} ({done_pct:.0f}%)] "
                      f"qid={result.get('qid', '?')[:8]} "
                      f"t={result.get('traj_idx', '?')} "
                      f"status={result.get('status', '?')} "
                      f"tools={result.get('num_tool_calls', '?')} "
                      f"time={result.get('latency_s', 0):.1f}s")

    # ---- Judge answers ----
    print(f"\n{'='*60}")
    print(f"Judging answers with {judge_model}...")
    print(f"{'='*60}\n")

    # Group by qid
    by_qid: Dict[str, List[Dict]] = {}
    for r in all_results:
        qid = r["qid"]
        if qid not in by_qid:
            by_qid[qid] = []
        by_qid[qid].append(r)

    judge_sem = asyncio.Semaphore(10)
    judge_results = []

    async def judge_one(r):
        async with judge_sem:
            if r.get("status") != "success" or not r.get("answer", "").strip():
                return {**r, "judge": {"correct": False, "explanation": "no answer"}}
            # Prefer boxed_answer for judging (more precise)
            prediction = r.get("boxed_answer") or r.get("answer", "")
            verdict = await judge_answer(
                question=r["question"],
                reference=r.get("reference_answer", ""),
                prediction=prediction,
                judge_http=judge_http,
                judge_model=judge_model,
            )
            return {**r, "judge": verdict}

    judge_tasks = [asyncio.create_task(judge_one(r)) for r in all_results]
    judged_results = []
    for coro in asyncio.as_completed(judge_tasks):
        result = await coro
        judged_results.append(result)

    # ---- Compute metrics ----
    print(f"\n{'='*60}")
    print(f"Computing pass@{num_trajectories}...")
    print(f"{'='*60}\n")

    # Group judged results by qid
    judged_by_qid: Dict[str, List[Dict]] = {}
    for r in judged_results:
        qid = r["qid"]
        if qid not in judged_by_qid:
            judged_by_qid[qid] = []
        judged_by_qid[qid].append(r)

    pass_count = 0
    total_questions = 0
    per_question = []

    for qid, trajs in judged_by_qid.items():
        any_correct = any(
            t.get("judge", {}).get("correct", False) for t in trajs
        )
        num_correct = sum(
            1 for t in trajs if t.get("judge", {}).get("correct", False)
        )
        total_questions += 1
        if any_correct:
            pass_count += 1

        accuracy = num_correct / max(len(trajs), 1)
        per_question.append({
            "qid": qid,
            "question": trajs[0].get("question", "")[:200],
            "reference": trajs[0].get("reference_answer", ""),
            "num_trajectories": len(trajs),
            "num_correct": num_correct,
            "accuracy": accuracy,
            "pass": any_correct,
            "answers": [
                {
                    "traj_idx": t.get("traj_idx"),
                    "answer": t.get("answer", "")[:500],
                    "boxed_answer": t.get("boxed_answer", ""),
                    "correct": t.get("judge", {}).get("correct", False),
                    "explanation": t.get("judge", {}).get("explanation", ""),
                    "num_tools": t.get("num_tool_calls", 0),
                    "time_s": t.get("latency_s", 0),
                }
                for t in sorted(trajs, key=lambda x: x.get("traj_idx", 0))
            ],
        })

    pass_at_k = pass_count / max(total_questions, 1)

    # avg@k: average per-question accuracy (num_correct / num_trajectories)
    avg_at_k = (
        sum(q["accuracy"] for q in per_question) / max(len(per_question), 1)
    )

    # Individual trajectory accuracy
    total_trajs = len(judged_results)
    correct_trajs = sum(
        1 for r in judged_results if r.get("judge", {}).get("correct", False)
    )
    traj_acc = correct_trajs / max(total_trajs, 1)

    avg_tools = sum(r.get("num_tool_calls", 0) for r in judged_results) / max(total_trajs, 1)
    avg_time = sum(r.get("latency_s", 0) for r in judged_results) / max(total_trajs, 1)

    summary = {
        "dataset": "Alibaba-NLP/WebShaper",
        "model": model,
        "num_questions": total_questions,
        "num_trajectories_per_q": num_trajectories,
        "total_trajectories": total_trajs,
        "max_tool_calls": max_tool_calls,
        "temperature": temperature,
        "judge_model": judge_model,
        f"pass@{num_trajectories}": pass_at_k,
        f"avg@{num_trajectories}": avg_at_k,
        "trajectory_accuracy": traj_acc,
        "correct_trajectories": correct_trajs,
        "avg_tool_calls": round(avg_tools, 1),
        "avg_latency_s": round(avg_time, 1),
        "per_question": per_question,
    }

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"{'='*60}")
    print(f"  RESULTS: WebShaper Evaluation")
    print(f"{'='*60}")
    print(f"  Model:            {model}")
    print(f"  Questions:        {total_questions}")
    print(f"  Trajectories/Q:   {num_trajectories}")
    print(f"  Max tool calls:   {max_tool_calls}")
    print(f"  Temperature:      {temperature}")
    print(f"  Judge:            {judge_model}")
    print(f"  ────────────────────────────────────────")
    print(f"  pass@{num_trajectories}:          {pass_at_k:.1%} ({pass_count}/{total_questions})")
    print(f"  avg@{num_trajectories}:           {avg_at_k:.1%}")
    print(f"  Traj accuracy:    {traj_acc:.1%} ({correct_trajs}/{total_trajs})")
    print(f"  Avg tool calls:   {avg_tools:.1f}")
    print(f"  Avg latency:      {avg_time:.1f}s")
    print(f"  ────────────────────────────────────────")
    print(f"  Trajectories:     {traj_file}")
    print(f"  Results:          {results_file}")
    print(f"{'='*60}")

    await http_client.aclose()
    await openai_http.aclose()
    await judge_http.aclose()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate on WebShaper with browser-agentic trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Test with 1 prompt, 1 trajectory
  python scripts/eval_webshaper.py --num-samples 1 --num-trajectories 1

  # Full eval: 500 prompts × 4 trajectories
  python scripts/eval_webshaper.py --num-samples 500 --num-trajectories 4 \\
      --concurrency 8 --output-dir results/webshaper

  # Resume interrupted run
  python scripts/eval_webshaper.py --num-samples 500 --num-trajectories 4 --resume
""",
    )
    parser.add_argument(
        "--scheduler-url", type=str,
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of questions to evaluate (default: all 500)")
    parser.add_argument("--num-trajectories", type=int, default=4,
                        help="Trajectories per question (default: 4)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent trajectory generations")
    parser.add_argument("--max-tool-calls", type=int, default=MAX_TOOL_CALLS,
                        help=f"Max browser tool calls per trajectory (default: {MAX_TOOL_CALLS})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-gen-tokens", type=int, default=MAX_GEN_TOKENS)
    parser.add_argument("--output-dir", type=str, default="results/webshaper")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing trajectories file")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help=f"LLM judge model (default: {JUDGE_MODEL})")
    args = parser.parse_args()

    # Auto-detect model
    if not args.model:
        try:
            resp = httpx.get(
                f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5
            )
            args.model = resp.json().get("model", "")
        except Exception:
            pass
        if not args.model:
            print("Error: Could not detect model. Use --model.")
            sys.exit(1)
        print(f"Model: {args.model}")

    # Load tokenizer
    print(f"Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    asyncio.run(run_evaluation(
        scheduler_url=args.scheduler_url,
        model=args.model,
        tokenizer=tokenizer,
        num_samples=args.num_samples,
        num_trajectories=args.num_trajectories,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        max_tool_calls=args.max_tool_calls,
        temperature=args.temperature,
        resume=args.resume,
        judge_model=args.judge_model,
    ))


if __name__ == "__main__":
    main()

