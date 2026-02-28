#!/usr/bin/env python3
"""
Closed-book evaluation — runs the model on HLE (or any QA dataset) WITHOUT
any tool use. The model answers purely from its parametric knowledge.

Usage:
    # Evaluate gpt-oss-120b on HLE
    python scripts/eval_closedbook.py \
        --dataset rl-rag/hle_text_only --split test \
        --num-trajectories 1 --concurrency 64 \
        --output-dir results/hle_closedbook_120b

    # Evaluate gpt-oss-20b on HLE
    python scripts/eval_closedbook.py \
        --scheduler-url http://SCHEDULER_HOST:8781 \
        --dataset rl-rag/hle_text_only --split test \
        --num-trajectories 1 --concurrency 64 \
        --output-dir results/hle_closedbook_20b
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

MAX_GEN_TOKENS = 16384
MAX_MODEL_LEN = 131072
TEMPERATURE = 0.7
JUDGE_MODEL = "gpt-4o"

CLOSEDBOOK_SYSTEM_PROMPT = """\
You are a knowledgeable assistant. Answer the following question by \
reasoning step by step. Think carefully before answering.

Provide your final answer in \\boxed{answer}. If the answer is a number, \
put just the number. If it is a short phrase, put the phrase. \
Acknowledge uncertainty when you are not sure."""

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


async def generate_closedbook(
    question: str,
    qid: str,
    base_urls,
    model: str,
    tokenizer: Any,
    traj_idx: int = 0,
    max_gen_tokens: int = MAX_GEN_TOKENS,
    temperature: float = TEMPERATURE,
    system_prompt: str = CLOSEDBOOK_SYSTEM_PROMPT,
):
    """Generate a single closed-book answer (no tools)."""
    openai_http = httpx.AsyncClient(timeout=600)

    # Round-robin across backend URLs
    if isinstance(base_urls, list):
        if not hasattr(generate_closedbook, '_rr_counter'):
            generate_closedbook._rr_counter = 0
        idx = generate_closedbook._rr_counter % len(base_urls)
        generate_closedbook._rr_counter += 1
        base_url = base_urls[idx]
    else:
        base_url = base_urls

    # Build prompt WITHOUT tool definitions
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    # Try to detect if this is a Harmony model (gpt-oss) or standard model
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    is_harmony = "<|channel|>" in str(chat_template) or "builtin_tools" in str(chat_template)

    if is_harmony:
        # Harmony format: use apply_chat_template WITHOUT builtin_tools
        from elastic_serving.dr_utils.prompts import MODEL_IDENTITY
        prompt = tokenizer.apply_chat_template(
            messages,
            model_identity=MODEL_IDENTITY,
            tokenize=False,
            add_generation_prompt=True,
        )
        stop_tokens = ["<|return|>"]
        extra_body = {"skip_special_tokens": False}
    else:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        stop_tokens = ["<|im_end|>"]
        extra_body = {}

    tag = f"qid={qid} t={traj_idx}"
    t0 = time.time()

    # Check prompt length
    prompt_len = len(tokenizer.encode(prompt))
    gen_tokens = min(max_gen_tokens, MAX_MODEL_LEN - prompt_len - 64)
    if gen_tokens <= 0:
        await openai_http.aclose()
        return {
            "qid": qid, "traj_idx": traj_idx, "question": question,
            "answer": "", "boxed_answer": "",
            "latency_s": time.time() - t0, "status": "context_overflow",
        }

    try:
        req_body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": gen_tokens,
            "temperature": temperature,
            "stop": stop_tokens,
            **extra_body,
        }

        for retry in range(60):
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json=req_body,
                headers={"Authorization": "Bearer EMPTY"},
                timeout=600,
            )
            if resp.status_code == 503:
                await asyncio.sleep(15 + min(retry * 5, 60))
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError("Server returned 503 after 60 retries")

        raw_text = resp.json()["choices"][0]["text"]

    except Exception as e:
        print(f"  [{tag}] Error: {e}")
        await openai_http.aclose()
        return {
            "qid": qid, "traj_idx": traj_idx, "question": question,
            "answer": "", "boxed_answer": "",
            "latency_s": time.time() - t0, "status": "error",
            "error": str(e),
        }

    # Extract answer
    answer = raw_text.strip()
    # Clean Harmony special tokens
    answer = re.sub(r"<\|[^|]+\|>", "", answer).strip()
    # Remove "assistant", "analysis", "commentary", "final" prefixes
    answer = re.sub(r"^(assistant\s*)?((analysis|commentary)\s*)*", "", answer).strip()

    # Extract \boxed{...}
    boxed_answer = ""
    boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer)
    if boxed_match:
        cleaned = re.sub(r'\\text\{([^}]*)\}', r'\1', boxed_match.group(1).strip())
        cleaned = re.sub(r'\\textrm\{([^}]*)\}', r'\1', cleaned)
        cleaned = re.sub(r'\\[,;!\s]', ' ', cleaned)
        cleaned = re.sub(r'\{,\}', ',', cleaned)
        cleaned = re.sub(r'[{}]', '', cleaned).replace('\\', '').strip()
        boxed_answer = re.sub(r'\s+', ' ', cleaned)

    elapsed = time.time() - t0
    print(f"  [{tag}] Done: {elapsed:.1f}s, answer={boxed_answer or answer[:80]}")
    await openai_http.aclose()

    return {
        "qid": qid, "traj_idx": traj_idx, "question": question,
        "answer": answer, "boxed_answer": boxed_answer,
        "num_tool_calls": 0, "latency_s": elapsed, "status": "success",
    }


async def judge_answer(question, reference, prediction, http, model=JUDGE_MODEL):
    """Judge if prediction matches reference using LLM judge."""
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
            json={
                "model": model,
                "messages": [{"role": "user", "content":
                    JUDGE_PROMPT.format(question=question, reference=reference, prediction=prediction)}],
                "temperature": 0, "max_tokens": 256,
            },
            headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return {"correct": bool(result.get("correct")), "explanation": result.get("explanation", "")}
    except Exception as e:
        return {"correct": False, "explanation": f"Judge error: {e}"}


async def run_eval(args):
    import datasets
    from transformers import AutoTokenizer

    # Load dataset
    print(f"Loading dataset: {args.dataset} split={args.split}")
    if args.dataset.endswith(".jsonl") or args.dataset.endswith(".json"):
        ds = datasets.load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = datasets.load_dataset(args.dataset, split=args.split)
    print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}")

    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    # Normalize column names
    q_col = "question" if "question" in ds.column_names else "query"
    a_col = "answer"
    has_id = "id" in ds.column_names

    def get_id(row, idx):
        if has_id:
            return str(row["id"])
        return hashlib.md5(row[q_col].encode()).hexdigest()[:8]

    # Auto-detect model
    if not args.model:
        try:
            resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "")
        except Exception:
            pass
    if not args.model:
        print("ERROR: --model is required (or scheduler must be running)")
        sys.exit(1)
    print(f"Model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Parse URLs
    raw_urls = [u.strip().rstrip("/") for u in args.scheduler_url.split(",")]
    base_urls = raw_urls if len(raw_urls) > 1 else raw_urls[0]
    base_url = raw_urls[0]

    # Wait for server to be ready (up to ~30 min for model loading)
    print("Waiting for server to be ready...")
    server_ready = False
    async with httpx.AsyncClient() as tmp:
        for attempt in range(180):
            try:
                r = await tmp.get(f"{base_url}/cluster_status", timeout=5)
                if r.status_code == 200 and r.json().get("ready_workers", 0) > 0:
                    print(f"Cluster: {r.json()['ready_workers']} workers ready")
                    server_ready = True
                    break
            except Exception:
                pass
            try:
                r = await tmp.get(f"{base_url}/v1/models", timeout=5)
                if r.status_code == 200:
                    print("Direct vLLM server ready")
                    server_ready = True
                    break
            except Exception:
                pass
            if attempt % 6 == 0:
                print(f"  ... waiting ({attempt * 10}s)")
            await asyncio.sleep(10)
    if not server_ready:
        print("ERROR: No workers became ready after 30 min. Exiting.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    results_file = os.path.join(args.output_dir, "results.json")

    # Resume support
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

    sem = asyncio.Semaphore(args.concurrency)
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
                result = await generate_closedbook(
                    question=row[q_col],
                    qid=qid,
                    base_urls=base_urls,
                    model=args.model,
                    tokenizer=tokenizer,
                    traj_idx=traj_idx,
                    max_gen_tokens=args.max_gen_tokens,
                    temperature=args.temperature,
                )
                result["reference_answer"] = row.get(a_col, "")
            except Exception:
                result = {
                    "qid": qid, "traj_idx": traj_idx,
                    "question": row[q_col],
                    "answer": "", "boxed_answer": "",
                    "reference_answer": row.get(a_col, ""),
                    "error": traceback.format_exc(),
                    "status": "error", "latency_s": 0,
                }
            total_done += 1
            return result

    # Generate trajectories
    n_trajs = len(ds) * args.num_trajectories
    print(f"\nGenerating {len(ds)} × {args.num_trajectories} = {n_trajs} "
          f"closed-book trajectories (concurrency={args.concurrency})")

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
    judge_http = httpx.AsyncClient(timeout=60)
    judge_sem = asyncio.Semaphore(10)

    async def judge_one(r):
        async with judge_sem:
            pred = r.get("boxed_answer") or r.get("answer", "")
            if r.get("status") != "success" or not pred.strip():
                return {**r, "judge": {"correct": False, "explanation": "no answer"}}
            v = await judge_answer(
                r["question"], r.get("reference_answer", ""),
                pred, judge_http, args.judge_model,
            )
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
        if any_c:
            pass_count += 1
        acc = nc / max(len(trajs), 1)
        per_question.append({
            "qid": qid,
            "reference": trajs[0].get("reference_answer", ""),
            "num_correct": nc, "accuracy": acc, "pass": any_c,
            "answers": [
                {
                    "traj_idx": t.get("traj_idx"),
                    "boxed_answer": t.get("boxed_answer", ""),
                    "correct": t.get("judge", {}).get("correct", False),
                    "explanation": t.get("judge", {}).get("explanation", ""),
                }
                for t in sorted(trajs, key=lambda x: x.get("traj_idx", 0))
            ],
        })

    n_q = len(by_qid)
    k = args.num_trajectories
    pass_at_k = pass_count / max(n_q, 1)
    avg_at_k = sum(q["accuracy"] for q in per_question) / max(n_q, 1)
    total_trajs = len(judged)
    correct_trajs = sum(1 for r in judged if r.get("judge", {}).get("correct"))

    summary = {
        "dataset": args.dataset, "split": args.split, "model": args.model,
        "eval_type": "closed-book",
        "num_questions": n_q, "num_trajectories_per_q": k,
        "total_trajectories": total_trajs,
        "temperature": args.temperature,
        "max_gen_tokens": args.max_gen_tokens,
        "judge_model": args.judge_model,
        f"pass@{k}": pass_at_k, f"avg@{k}": avg_at_k,
        "trajectory_accuracy": correct_trajs / max(total_trajs, 1),
        "correct_trajectories": correct_trajs,
        "per_question": per_question,
    }
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  CLOSED-BOOK: {args.dataset} ({args.split})")
    print(f"  Model: {args.model}")
    print(f"  pass@{k}: {pass_at_k:.1%}  avg@{k}: {avg_at_k:.1%}")
    print(f"  traj acc: {correct_trajs}/{total_trajs} ({correct_trajs/max(total_trajs,1):.1%})")
    print(f"{'='*60}")
    await judge_http.aclose()


def main():
    p = argparse.ArgumentParser(description="Closed-book QA evaluation (no tools)")
    p.add_argument("--scheduler-url", default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--num-samples", type=int, default=-1, help="Limit to N samples (-1 = all)")
    p.add_argument("--num-trajectories", type=int, default=1, help="Trajectories per question")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-gen-tokens", type=int, default=MAX_GEN_TOKENS)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
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

