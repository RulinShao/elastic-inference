#!/usr/bin/env python3
"""
SFT Trajectory Generation with Deep Research
==============================================

Generates research trajectories using Harmony-native browser tools
(browser.search, browser.open, browser.find) and paper_search.

Output: JSONL with full trajectories including tool calls and answers.

Usage:
    python scripts/generate_trajectories.py \\
        --scheduler-url http://localhost:8780 \\
        --dataset sample --num-samples 3 \\
        --output-dir results/trajectories
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import dotenv
import httpx

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.tools import (
    STOP_TOKENS,
    STOP_TOKENS_NO_CALL,
    SYSTEM_PROMPT,
    BrowserSession,
    append_tool_round,
    build_initial_prompt,
    execute_custom_tool,
    extract_final_answer,
    parse_tool_call,
)


# =============================================================================
# Trajectory Generation
# =============================================================================


async def generate_one_trajectory(
    question: str,
    qid: Any,
    base_url: str,
    model: str,
    tokenizer,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    max_tool_calls: int = 15,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Generate a single research trajectory."""
    browser = BrowserSession(http_client)
    tool_call_count = 0
    tool_calls_log: List[Dict[str, Any]] = []

    prompt = build_initial_prompt(tokenizer, user_message=question)

    t0 = time.time()
    print(f"  [qid={qid}] Starting")

    while True:
        at_limit = tool_call_count >= max_tool_calls
        stops = STOP_TOKENS_NO_CALL if at_limit else STOP_TOKENS

        try:
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_gen_tokens,
                    "temperature": temperature,
                    "stop": stops,
                    "skip_special_tokens": False,
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=600,
            )
            if resp.status_code == 503:
                await asyncio.sleep(10)
                continue
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["text"]
        except Exception as e:
            print(f"  [qid={qid}] Generation error: {e}")
            break

        tool_call = parse_tool_call(raw_text) if not at_limit else None

        if tool_call:
            ns, tool_name, tool_args = tool_call
            tool_call_count += 1

            short = json.dumps(tool_args, ensure_ascii=False)[:100]
            print(f"  [qid={qid}]   Tool {tool_call_count}: {ns}.{tool_name}({short})")

            if ns == "browser":
                result = await browser.execute(tool_name, tool_args)
            else:
                result = await execute_custom_tool(tool_name, tool_args, http_client)

            tool_calls_log.append({
                "round": tool_call_count,
                "tool": f"{ns}.{tool_name}",
                "args": tool_args,
                "result_len": len(result),
            })

            prompt = append_tool_round(prompt, raw_text, tool_name, result, namespace=ns)
            continue
        else:
            reasoning, answer = extract_final_answer(raw_text)
            elapsed = time.time() - t0
            print(f"  [qid={qid}] Done: {tool_call_count} tools, {elapsed:.1f}s, "
                  f"answer={answer[:80]}")

            return {
                "qid": qid,
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "num_tool_calls": tool_call_count,
                "tool_calls": tool_calls_log,
                "latency_s": elapsed,
                "status": "success",
            }

    elapsed = time.time() - t0
    return {
        "qid": qid,
        "question": question,
        "answer": "",
        "num_tool_calls": tool_call_count,
        "tool_calls": tool_calls_log,
        "latency_s": elapsed,
        "status": "error",
    }


# =============================================================================
# Main
# =============================================================================


SAMPLE_QUESTIONS = [
    {"qid": 1, "question": "What were the key findings of the most recent IPCC report on climate change?"},
    {"qid": 2, "question": "Who is the current CEO of Anthropic, when was the company founded, and what is their stated mission regarding AI safety?"},
    {"qid": 3, "question": "Describe the architecture and key innovations of the Mamba state space model."},
    {"qid": 4, "question": "What is the current state of nuclear fusion energy research? Describe the NIF's ignition achievement."},
    {"qid": 5, "question": "What is DR Tulu and how does it work?"},
]


async def run_generation(
    scheduler_url: str,
    model: str,
    dataset_name: str,
    num_samples: int,
    concurrency: int,
    output_dir: str,
    max_tool_calls: int,
    max_gen_tokens: int,
    temperature: float,
):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=600)

    # Load dataset
    if dataset_name == "sample":
        data = SAMPLE_QUESTIONS
    else:
        import datasets
        ds = datasets.load_dataset(dataset_name, split="main")
        data = [{"qid": row["id"], "question": row["question"]} for row in ds]

    if num_samples > 0:
        data = data[:num_samples]

    # Wait for workers
    base_url = scheduler_url.rstrip("/")
    async with httpx.AsyncClient() as tmp:
        for _ in range(120):
            try:
                resp = await tmp.get(f"{base_url}/cluster_status", timeout=5)
                if resp.json().get("ready_workers", 0) > 0:
                    print(f"Cluster: {resp.json()['ready_workers']} ready workers")
                    break
            except Exception:
                pass
            print("  Waiting for workers...")
            await asyncio.sleep(10)
        else:
            print("Timed out waiting for workers.")
            return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"trajectories_{dataset_name}.jsonl")

    # Resume
    completed_qids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    completed_qids.add(json.loads(line)["qid"])
                except Exception:
                    pass
        print(f"Resuming: {len(completed_qids)} done")

    pending = [d for d in data if d["qid"] not in completed_qids]
    if not pending:
        print("All done!")
        return

    print(f"Processing {len(pending)} samples (concurrency={concurrency})...\n")

    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def process_one(item):
        nonlocal completed
        async with sem:
            try:
                result = await generate_one_trajectory(
                    question=item["question"],
                    qid=item["qid"],
                    base_url=base_url,
                    model=model,
                    tokenizer=tokenizer,
                    http_client=http_client,
                    openai_http=openai_http,
                    max_tool_calls=max_tool_calls,
                    max_gen_tokens=max_gen_tokens,
                    temperature=temperature,
                )
                result["answer_ref"] = item.get("answer", "")
            except Exception:
                result = {
                    "qid": item["qid"],
                    "question": item["question"],
                    "error": traceback.format_exc(),
                    "status": "error",
                }
            completed += 1
            print(f"[{completed}/{len(pending)}] qid={item['qid']} "
                  f"{result.get('status', '?')} "
                  f"tools={result.get('num_tool_calls', 0)} "
                  f"time={result.get('latency_s', 0):.1f}s")
            return result

    tasks = [asyncio.create_task(process_one(item)) for item in pending]
    with open(output_file, "a") as writer:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()

    await http_client.aclose()
    await openai_http.aclose()
    print(f"\nDone! {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate research trajectories")
    parser.add_argument("--scheduler-url", type=str,
                        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sample")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--max-gen-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="results/trajectories")
    args = parser.parse_args()

    if not args.model:
        try:
            resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "default")
        except Exception:
            args.model = "default"

    asyncio.run(run_generation(
        scheduler_url=args.scheduler_url,
        model=args.model,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        max_tool_calls=args.max_tool_calls,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()
