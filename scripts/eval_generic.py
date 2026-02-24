#!/usr/bin/env python3
"""
Generic evaluation — runs the deep research agent on any HF dataset
with 'question' and 'answer' columns.

Usage:
    python scripts/eval_generic.py \
        --dataset rl-rag/bc_synthetic_v_2 --split normal \
        --num-trajectories 4 --blocked-domains huggingface.co \
        --output-dir results/bc_synthetic_normal
"""

import argparse
import asyncio
import hashlib
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

MAX_TOOL_CALLS = 50
MAX_GEN_TOKENS = 8192
MAX_MODEL_LEN = 131072
TEMPERATURE = 0.7
JUDGE_MODEL = "gpt-4o"

JUDGE_PROMPT = """\
You are an impartial judge evaluating whether a model's answer is correct.

**Question:** {question}

**Reference answer:** {reference}

**Model's answer:** {prediction}

Evaluate whether the model's answer is correct. The model's answer does NOT \
need to match the reference answer word-for-word — it just needs to convey \
the same factual information. Be lenient about formatting differences.

Respond with a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""


async def generate_trajectory(
    question, qid, base_url, model, tokenizer,
    traj_idx=0, max_tool_calls=MAX_TOOL_CALLS,
    max_gen_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE,
    save_conversation=False, api_sem=None, blocked_domains=None,
):
    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=600)
    browser = BrowserSession(http_client, blocked_domains=blocked_domains)
    tool_call_count = 0
    tool_calls_log = []
    conversation = [] if save_conversation else None

    prompt = build_initial_prompt(tokenizer, user_message=question)
    tag = f"qid={qid} t={traj_idx}"
    t0 = time.time()

    while True:
        at_limit = tool_call_count >= max_tool_calls
        stops = STOP_TOKENS_NO_CALL if at_limit else STOP_TOKENS

        prompt_len = len(tokenizer.encode(prompt))
        if prompt_len + max_gen_tokens > MAX_MODEL_LEN:
            remaining = MAX_MODEL_LEN - prompt_len - 256
            if remaining < 256:
                stops = STOP_TOKENS_NO_CALL
                remaining = min(2048, MAX_MODEL_LEN - prompt_len - 64)
                if remaining <= 0:
                    break
            gen_tokens = remaining
        else:
            gen_tokens = max_gen_tokens

        try:
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model, "prompt": prompt,
                    "max_tokens": gen_tokens, "temperature": temperature,
                    "stop": stops, "skip_special_tokens": False,
                },
                headers={"Authorization": "Bearer EMPTY"}, timeout=600,
            )
            if resp.status_code == 503:
                await asyncio.sleep(10)
                continue
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["text"]
        except Exception as e:
            print(f"  [{tag}] Error: {e}")
            break

        tool_call = parse_tool_call(raw_text) if not at_limit else None

        if tool_call:
            ns, tool_name, tool_args = tool_call
            tool_call_count += 1
            short = json.dumps(tool_args, ensure_ascii=False)[:80]
            print(f"  [{tag}] Tool {tool_call_count}/{max_tool_calls}: {ns}.{tool_name}({short})")

            if api_sem:
                async with api_sem:
                    result = await browser.execute(tool_name, tool_args) if ns == "browser" else \
                             await execute_custom_tool(tool_name, tool_args, http_client)
            else:
                result = await browser.execute(tool_name, tool_args) if ns == "browser" else \
                         await execute_custom_tool(tool_name, tool_args, http_client)

            tool_calls_log.append({
                "round": tool_call_count, "tool": f"{ns}.{tool_name}",
                "args": tool_args, "result_len": len(result),
            })
            if save_conversation:
                conversation.append({"role": "assistant", "content": raw_text,
                    "tool_call": {"tool": f"{ns}.{tool_name}", "args": tool_args}})
                conversation.append({"role": "tool", "tool": f"{ns}.{tool_name}",
                    "content": result[:30000]})

            prompt = append_tool_round(prompt, raw_text, tool_name, result, namespace=ns)
        else:
            _r, answer = extract_final_answer(raw_text)
            boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer)
            boxed_answer = ""
            if boxed_match:
                cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_match.group(1).strip())
                cleaned = re.sub(r'\\textrm\{([^}]*)\}', r'\1', cleaned)
                cleaned = re.sub(r'\\[,;!\s]', ' ', cleaned)
                cleaned = re.sub(r'\{,\}', ',', cleaned)
                cleaned = re.sub(r'[{}]', '', cleaned).replace('\\', '').strip()
                boxed_answer = re.sub(r'\s+', ' ', cleaned)

            if save_conversation:
                conversation.append({"role": "assistant", "content": raw_text, "final_answer": answer})

            elapsed = time.time() - t0
            print(f"  [{tag}] Done: {tool_call_count} tools, {elapsed:.1f}s, answer={boxed_answer or answer[:80]}")
            await http_client.aclose()
            await openai_http.aclose()
            return {
                "qid": qid, "traj_idx": traj_idx, "question": question,
                "answer": answer, "boxed_answer": boxed_answer,
                "num_tool_calls": tool_call_count, "tool_calls": tool_calls_log,
                "conversation": conversation, "latency_s": elapsed, "status": "success",
            }

    await http_client.aclose()
    await openai_http.aclose()
    return {
        "qid": qid, "traj_idx": traj_idx, "question": question,
        "answer": "", "boxed_answer": "", "num_tool_calls": tool_call_count,
        "tool_calls": tool_calls_log, "conversation": conversation,
        "latency_s": time.time() - t0, "status": "context_overflow",
    }


async def judge_answer(question, reference, prediction, http, model=JUDGE_MODEL):
    if not prediction.strip():
        return {"correct": False, "explanation": "Empty answer"}
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        norm_r = reference.strip().lower()
        norm_p = prediction.strip().lower()
        return {"correct": norm_r in norm_p or norm_p in norm_r, "explanation": "fallback match"}
    try:
        resp = await http.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content":
                JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)}],
                "temperature": 0, "max_tokens": 256},
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        result = json.loads(content)
        return {"correct": bool(result.get("correct")), "explanation": result.get("explanation", "")}
    except Exception as e:
        return {"correct": False, "explanation": f"Judge error: {e}"}


async def run_eval(args):
    import datasets
    from transformers import AutoTokenizer

    # Load dataset
    print(f"Loading dataset: {args.dataset} split={args.split}")
    ds = datasets.load_dataset(args.dataset, split=args.split)
    print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    # Add IDs if missing
    # Normalize column names: support 'query' as alias for 'question'
    q_col = "question" if "question" in ds.column_names else "query"
    a_col = "answer"

    has_id = "id" in ds.column_names
    def get_id(row, idx):
        if has_id:
            return str(row["id"])
        return hashlib.md5(row[q_col].encode()).hexdigest()[:8]

    # Auto-detect model
    if not args.model:
        resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
        args.model = resp.json().get("model", "")
    print(f"Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Wait for workers
    base_url = args.scheduler_url.rstrip("/")
    async with httpx.AsyncClient() as tmp:
        for _ in range(120):
            try:
                r = await tmp.get(f"{base_url}/cluster_status", timeout=5)
                if r.json().get("ready_workers", 0) > 0:
                    print(f"Cluster: {r.json()['ready_workers']} workers ready")
                    break
            except Exception:
                pass
            await asyncio.sleep(10)

    os.makedirs(args.output_dir, exist_ok=True)
    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    results_file = os.path.join(args.output_dir, "results.json")

    # Resume
    completed = {}
    if args.resume and os.path.exists(traj_file):
        with open(traj_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed[(r["qid"], r.get("traj_idx", 0))] = r
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} trajectories done")

    judge_http = httpx.AsyncClient(timeout=60)
    sem = asyncio.Semaphore(args.concurrency)
    api_sem = asyncio.Semaphore(min(args.concurrency * 3, 200))
    total_done = 0
    all_results = []
    write_buffer = []
    FLUSH_EVERY = 20

    def flush_buffer():
        if write_buffer:
            with open(traj_file, "a") as w:
                w.write("".join(write_buffer))
            write_buffer.clear()

    async def process_one(row, idx, traj_idx):
        nonlocal total_done
        async with sem:
            qid = get_id(row, idx)
            if (qid, traj_idx) in completed:
                return completed[(qid, traj_idx)]
            try:
                result = await generate_trajectory(
                    question=row[q_col], qid=qid, base_url=base_url,
                    model=args.model, tokenizer=tokenizer, traj_idx=traj_idx,
                    max_tool_calls=args.max_tool_calls, temperature=args.temperature,
                    save_conversation=args.save_full_trajectories,
                    api_sem=api_sem, blocked_domains=args.blocked_domains,
                )
                result["reference_answer"] = row.get(a_col, "")
            except Exception:
                result = {"qid": qid, "traj_idx": traj_idx, "question": row[q_col],
                          "answer": "", "reference_answer": row.get(a_col, ""),
                          "error": traceback.format_exc(), "status": "error", "latency_s": 0}
            total_done += 1
            return result

    # Generate
    n_trajs = len(ds) * args.num_trajectories
    print(f"\nGenerating {len(ds)} × {args.num_trajectories} = {n_trajs} trajectories (concurrency={args.concurrency})")

    tasks = []
    for idx, row in enumerate(ds):
        for t in range(args.num_trajectories):
            tasks.append(asyncio.create_task(process_one(row, idx, t)))

    for coro in asyncio.as_completed(tasks):
        result = await coro
        all_results.append(result)
        write_buffer.append(json.dumps(result, ensure_ascii=False) + "\n")
        if len(write_buffer) >= FLUSH_EVERY:
            flush_buffer()
        if total_done % 50 == 0 or total_done == n_trajs:
            print(f"[{total_done}/{n_trajs} ({total_done/n_trajs*100:.0f}%)]")
    flush_buffer()

    # Judge
    print(f"\nJudging with {args.judge_model}...")
    judge_sem = asyncio.Semaphore(10)

    async def judge_one(r):
        async with judge_sem:
            pred = r.get("boxed_answer") or r.get("answer", "")
            if r.get("status") != "success" or not pred.strip():
                return {**r, "judge": {"correct": False, "explanation": "no answer"}}
            v = await judge_answer(r["question"], r.get("reference_answer", ""), pred, judge_http, args.judge_model)
            return {**r, "judge": v}

    judged = []
    for coro in asyncio.as_completed([asyncio.create_task(judge_one(r)) for r in all_results]):
        judged.append(await coro)

    # Metrics
    by_qid = {}
    for r in judged:
        by_qid.setdefault(r["qid"], []).append(r)

    pass_count = 0
    per_question = []
    for qid, trajs in by_qid.items():
        nc = sum(1 for t in trajs if t.get("judge", {}).get("correct"))
        any_c = nc > 0
        if any_c: pass_count += 1
        acc = nc / max(len(trajs), 1)
        per_question.append({"qid": qid, "reference": trajs[0].get("reference_answer", ""),
            "num_correct": nc, "accuracy": acc, "pass": any_c,
            "answers": [{"traj_idx": t.get("traj_idx"), "boxed_answer": t.get("boxed_answer", ""),
                "correct": t.get("judge", {}).get("correct", False),
                "explanation": t.get("judge", {}).get("explanation", "")}
                for t in sorted(trajs, key=lambda x: x.get("traj_idx", 0))]})

    n_q = len(by_qid)
    k = args.num_trajectories
    pass_at_k = pass_count / max(n_q, 1)
    avg_at_k = sum(q["accuracy"] for q in per_question) / max(n_q, 1)
    total_trajs = len(judged)
    correct_trajs = sum(1 for r in judged if r.get("judge", {}).get("correct"))
    avg_tools = sum(r.get("num_tool_calls", 0) for r in judged) / max(total_trajs, 1)

    summary = {
        "dataset": args.dataset, "split": args.split, "model": args.model,
        "num_questions": n_q, "num_trajectories_per_q": k,
        "total_trajectories": total_trajs, "max_tool_calls": args.max_tool_calls,
        "temperature": args.temperature, "judge_model": args.judge_model,
        "blocked_domains": args.blocked_domains,
        f"pass@{k}": pass_at_k, f"avg@{k}": avg_at_k,
        "trajectory_accuracy": correct_trajs / max(total_trajs, 1),
        "correct_trajectories": correct_trajs,
        "avg_tool_calls": round(avg_tools, 1),
        "per_question": per_question,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  {args.dataset} ({args.split})")
    print(f"  pass@{k}: {pass_at_k:.1%}  avg@{k}: {avg_at_k:.1%}")
    print(f"  traj acc: {correct_trajs}/{total_trajs} ({correct_trajs/max(total_trajs,1):.1%})")
    print(f"  avg tools: {avg_tools:.1f}")
    print(f"{'='*60}")
    await judge_http.aclose()


def main():
    p = argparse.ArgumentParser(description="Generic deep research evaluation")
    p.add_argument("--scheduler-url", default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="normal")
    p.add_argument("--num-samples", type=int, default=-1)
    p.add_argument("--num-trajectories", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tool-calls", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--save-full-trajectories", action="store_true")
    p.add_argument("--blocked-domains", nargs="*", default=None)
    args = p.parse_args()

    if not args.model:
        try:
            r = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = r.json().get("model", "")
        except Exception:
            pass

    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()

