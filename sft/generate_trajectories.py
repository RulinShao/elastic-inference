#!/usr/bin/env python3
"""
Generate SFT trajectories using Qwen3.5 with elastic-serving tools.

Uses the Qwen3Adapter and elastic-serving tool backends (web_search, open_url,
find_text, paper_search, pubmed_search, python) to generate multi-turn
tool-calling trajectories for supervised fine-tuning.

Output: JSONL with full conversations including tool calls and responses,
ready for reformat_sft_data.py to convert into LLaMA-Factory format.

Usage::

    # Using elastic-serving scheduler
    python sft/generate_trajectories.py \\
        --base-url http://localhost:8780 \\
        --model Qwen/Qwen3.5-27B \\
        --dataset data/browsecomp.jsonl \\
        --output-dir sft/trajectories

    # Using standalone vLLM server
    python sft/generate_trajectories.py \\
        --base-url http://GPU_NODE:8001 \\
        --model Qwen/Qwen3.5-27B \\
        --dataset sample --num-samples 5

    # With Python code execution
    python sft/generate_trajectories.py \\
        --base-url http://localhost:8780 \\
        --enable-python --dataset sample
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

from elastic_serving.adapters import Qwen3Adapter
from elastic_serving.dr_utils import (
    SYSTEM_PROMPT,
    BrowserSession,
    PythonSession,
    execute_custom_tool,
)

# Reverse map: (namespace, backend_name) → Qwen3 tool name
_BACKEND_TO_QWEN = {
    ("browser", "search"): "web_search",
    ("browser", "open"): "open_url",
    ("browser", "find"): "find_text",
    ("python", "execute"): "python",
    ("functions", "paper_search"): "paper_search",
    ("functions", "pubmed_search"): "pubmed_search",
}

SAMPLE_QUESTIONS = [
    {
        "id": "sample_1",
        "question": (
            "What were the key findings of the most recent IPCC report "
            "on climate change?"
        ),
    },
    {
        "id": "sample_2",
        "question": (
            "Who is the current CEO of Anthropic, when was the company "
            "founded, and what is their stated mission regarding AI safety?"
        ),
    },
    {
        "id": "sample_3",
        "question": (
            "Describe the architecture and key innovations of the Mamba "
            "state space model."
        ),
    },
    {
        "id": "sample_4",
        "question": (
            "What is the current state of nuclear fusion energy research? "
            "Describe the NIF's ignition achievement."
        ),
    },
    {
        "id": "sample_5",
        "question": (
            "What is the latest progress on GLP-1 receptor agonists for "
            "weight loss? Search both web and PubMed."
        ),
    },
]


async def generate_trajectory(
    question: str,
    qid: str,
    base_url: str,
    model: str,
    adapter: Qwen3Adapter,
    tokenizer,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    max_tool_calls: int = 15,
    max_gen_tokens: int = 16384,
    temperature: float = 0.7,
    enable_python: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Generate a single trajectory with full tool call history.

    Returns a dict with ``turns`` containing the raw assistant outputs
    (including ``<think>`` and ``<tool_call>`` blocks) and tool responses.
    """
    browser = BrowserSession(http_client)
    python_session = None
    if enable_python:
        python_session = PythonSession()

    tool_call_count = 0
    turns: List[Dict[str, Any]] = []
    tools_used: set = set()

    prompt = adapter.build_prompt(
        tokenizer,
        question,
        system_prompt=system_prompt,
        enable_python=enable_python,
    )

    t0 = time.time()
    print(f"  [{qid}] Starting: {question[:80]}...")

    try:
        while True:
            at_limit = tool_call_count >= max_tool_calls
            stops = (
                adapter.stop_tokens_no_call if at_limit else adapter.stop_tokens
            )

            try:
                resp = await openai_http.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": max_gen_tokens,
                        "temperature": temperature,
                        "stop": stops,
                        **adapter.extra_body,
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
                print(f"  [{qid}] Generation error: {e}")
                break

            tool_call = adapter.parse_tool_call(raw_text) if not at_limit else None

            if tool_call:
                ns, tool_name, tool_args = tool_call
                tool_call_count += 1
                qwen_name = _BACKEND_TO_QWEN.get(
                    (ns, tool_name), f"{ns}.{tool_name}"
                )
                tools_used.add(qwen_name)

                short = json.dumps(tool_args, ensure_ascii=False)[:100]
                print(
                    f"  [{qid}]   Tool {tool_call_count}: "
                    f"{qwen_name}({short})"
                )

                if ns == "browser":
                    result = await browser.execute(tool_name, tool_args)
                elif ns == "python" and python_session:
                    result = python_session.execute(tool_args.get("code", ""))
                else:
                    result = await execute_custom_tool(
                        tool_name, tool_args, http_client
                    )

                turns.append(
                    {
                        "role": "assistant",
                        "content": raw_text,
                        "tool_call": {
                            "name": qwen_name,
                            "arguments": tool_args,
                        },
                    }
                )
                turns.append(
                    {"role": "tool", "name": qwen_name, "content": result}
                )

                prompt = adapter.format_tool_response(
                    prompt, raw_text, tool_name, result, namespace=ns
                )
            else:
                reasoning, answer = adapter.extract_final_answer(raw_text)
                turns.append({"role": "assistant", "content": raw_text})
                elapsed = time.time() - t0
                print(
                    f"  [{qid}] Done: {tool_call_count} tools, "
                    f"{elapsed:.1f}s, answer={answer[:80]}..."
                )

                return {
                    "id": qid,
                    "question": question,
                    "system_prompt": system_prompt,
                    "turns": turns,
                    "tools_used": sorted(tools_used),
                    "num_tool_calls": tool_call_count,
                    "answer": answer,
                    "reasoning": reasoning,
                    "status": "success",
                    "latency_s": round(time.time() - t0, 1),
                }
    finally:
        if python_session:
            python_session.close()

    return {
        "id": qid,
        "question": question,
        "system_prompt": system_prompt,
        "turns": turns,
        "tools_used": sorted(tools_used),
        "num_tool_calls": tool_call_count,
        "answer": "",
        "status": "error",
        "latency_s": round(time.time() - t0, 1),
    }


# ── Dataset loading ──────────────────────────────────────────────────────────


def load_dataset_items(dataset_arg: str, num_samples: int) -> List[Dict]:
    if dataset_arg == "sample":
        items = list(SAMPLE_QUESTIONS)
    elif dataset_arg.endswith(".jsonl"):
        items = []
        with open(dataset_arg) as f:
            for line in f:
                row = json.loads(line)
                items.append(
                    {
                        "id": str(row.get("id", row.get("qid", len(items)))),
                        "question": row.get(
                            "question", row.get("input", "")
                        ),
                        "answer": row.get("answer", row.get("target", "")),
                    }
                )
    else:
        import datasets

        ds = datasets.load_dataset(dataset_arg, split="train")
        items = [
            {
                "id": str(row.get("id", i)),
                "question": row["question"],
                "answer": row.get("answer", ""),
            }
            for i, row in enumerate(ds)
        ]

    if num_samples > 0:
        items = items[:num_samples]
    return items


# ── Main generation loop ─────────────────────────────────────────────────────


async def run_generation(args):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    adapter = Qwen3Adapter(enable_thinking=not args.no_think)

    data = load_dataset_items(args.dataset, args.num_samples)

    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=600)
    base_url = args.base_url.rstrip("/")

    print(f"Connecting to {base_url}...")
    for attempt in range(60):
        try:
            resp = await openai_http.get(f"{base_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                print("Server ready.")
                break
        except Exception:
            pass
        try:
            resp = await openai_http.get(
                f"{base_url}/cluster_status", timeout=5
            )
            info = resp.json()
            if info.get("ready_workers", 0) > 0:
                print(f"Cluster ready: {info['ready_workers']} workers")
                break
        except Exception:
            pass
        if attempt == 0:
            print("  Waiting for server...")
        await asyncio.sleep(5)
    else:
        print("Warning: could not verify server status, proceeding anyway")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "trajectories.jsonl")

    completed_ids: set = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resuming: {len(completed_ids)} already done")

    pending = [d for d in data if d["id"] not in completed_ids]
    if not pending:
        print("All trajectories already generated!")
        return

    print(
        f"Generating {len(pending)} trajectories "
        f"(concurrency={args.concurrency})...\n"
    )

    sem = asyncio.Semaphore(args.concurrency)
    completed = 0

    async def process_one(item):
        nonlocal completed
        async with sem:
            try:
                result = await generate_trajectory(
                    question=item["question"],
                    qid=item["id"],
                    base_url=base_url,
                    model=args.model,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    http_client=http_client,
                    openai_http=openai_http,
                    max_tool_calls=args.max_tool_calls,
                    max_gen_tokens=args.max_gen_tokens,
                    temperature=args.temperature,
                    enable_python=args.enable_python,
                )
                if "answer" in item and item["answer"]:
                    result["answer_ref"] = item["answer"]
            except Exception:
                result = {
                    "id": item["id"],
                    "question": item["question"],
                    "error": traceback.format_exc(),
                    "status": "error",
                }
            completed += 1
            print(
                f"[{completed}/{len(pending)}] id={item['id']} "
                f"{result.get('status', '?')} "
                f"tools={result.get('num_tool_calls', 0)}"
            )
            return result

    tasks = [asyncio.create_task(process_one(item)) for item in pending]
    with open(output_file, "a") as f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    await http_client.aclose()
    await openai_http.aclose()

    total = success = 0
    with open(output_file) as f:
        for line in f:
            total += 1
            if json.loads(line).get("status") == "success":
                success += 1
    print(f"\nDone! {output_file}")
    print(f"  Total: {total}  Success: {success}  Error: {total - success}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT trajectories with Qwen3.5 + elastic-serving tools"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
        help="elastic-serving scheduler or vLLM server URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="Model name (must match served model)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample",
        help="'sample', a .jsonl path, or a HuggingFace dataset name",
    )
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tool-calls", type=int, default=15)
    parser.add_argument("--max-gen-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="sft/trajectories")
    parser.add_argument(
        "--enable-python",
        action="store_true",
        help="Enable Python code execution tool",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable <think> reasoning blocks",
    )
    args = parser.parse_args()

    asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
