#!/usr/bin/env python3
"""
Load test for Elastic Serving.

Sends many concurrent requests to fully utilize multiple nodes/workers.
Uses the standard OpenAI client pointing at the scheduler proxy.

Usage:
    python scripts/test_load.py --scheduler-url http://localhost:8780 \
        --num-requests 100 --concurrency 32
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RequestResult:
    request_id: int
    success: bool
    latency: float
    tokens_generated: int = 0
    error: Optional[str] = None
    worker_hint: str = ""


async def send_request(
    client,
    model: str,
    request_id: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    """Send a single chat completion request."""
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.time() - t0
        content = resp.choices[0].message.content or ""
        tokens = resp.usage.completion_tokens if resp.usage else len(content.split())
        return RequestResult(
            request_id=request_id,
            success=True,
            latency=latency,
            tokens_generated=tokens,
        )
    except Exception as e:
        latency = time.time() - t0
        return RequestResult(
            request_id=request_id,
            success=False,
            latency=latency,
            error=str(e),
        )


async def run_load_test(
    scheduler_url: str,
    model: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    prompts: Optional[List[str]] = None,
):
    """Run the load test with controlled concurrency."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"{scheduler_url.rstrip('/')}/v1",
        api_key="EMPTY",
    )

    if prompts is None:
        prompts = [
            "What is the capital of France? Answer in one sentence.",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
            "What are the main differences between Python and Java?",
            "Describe the process of photosynthesis briefly.",
            "What is machine learning? Give a concise explanation.",
            "Name three famous scientists and their contributions.",
            "What causes earthquakes? Explain simply.",
            "How does the internet work at a high level?",
            "What is the meaning of life according to philosophy?",
            "Summarize the plot of Romeo and Juliet.",
            "What are the benefits of exercise?",
            "Explain how a computer processor works.",
            "What is the Big Bang theory?",
            "Describe the water cycle in nature.",
            "What makes a good leader?",
        ]

    # Check cluster status first
    import httpx
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{scheduler_url.rstrip('/')}/cluster_status")
        status = resp.json()

    print("=" * 70)
    print("LOAD TEST — Elastic Serving")
    print("=" * 70)
    print(f"Scheduler:     {scheduler_url}")
    print(f"Model:         {model}")
    print(f"Ready workers: {status['ready_workers']}")
    print(f"Total workers: {status['total_workers']}")
    print(f"Nodes active:  {status.get('total_nodes_active', '?')}")
    print(f"TP={status.get('tensor_parallel_size', '?')}, "
          f"DP/node={status.get('dp_per_node', '?')}")
    print(f"Requests:      {num_requests}")
    print(f"Concurrency:   {concurrency}")
    print(f"Max tokens:    {max_tokens}")
    print("=" * 70)

    if status['ready_workers'] == 0:
        print("\n⚠  No ready workers! Waiting for workers to come online...")
        for _ in range(120):  # wait up to 10 min
            await asyncio.sleep(5)
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{scheduler_url.rstrip('/')}/cluster_status")
                status = resp.json()
            ready = status['ready_workers']
            loading = status['loading_workers']
            pending = status['pending_slurm_jobs']
            print(f"  Ready: {ready}, Loading: {loading}, Pending SLURM: {pending}")
            if ready > 0:
                break
        else:
            print("Timed out waiting for workers. Aborting.")
            return

    print(f"\n🚀 Starting load test with {status['ready_workers']} ready workers...\n")

    sem = asyncio.Semaphore(concurrency)
    results: List[RequestResult] = []
    completed = 0

    async def bounded_request(req_id: int):
        nonlocal completed
        async with sem:
            prompt = prompts[req_id % len(prompts)]
            result = await send_request(client, model, req_id, prompt, max_tokens, temperature)
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == num_requests:
                success_count = sum(1 for r in results if r.success)
                avg_lat = sum(r.latency for r in results if r.success) / max(success_count, 1)
                print(f"  [{completed}/{num_requests}] "
                      f"success={success_count}, avg_latency={avg_lat:.2f}s")
            return result

    t_start = time.time()
    tasks = [asyncio.create_task(bounded_request(i)) for i in range(num_requests)]
    await asyncio.gather(*tasks)
    total_time = time.time() - t_start

    # Summarize results
    success_results = [r for r in results if r.success]
    fail_results = [r for r in results if not r.success]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total requests:    {num_requests}")
    print(f"Successful:        {len(success_results)}")
    print(f"Failed:            {len(fail_results)}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"Throughput:        {num_requests / total_time:.2f} req/s")

    if success_results:
        latencies = [r.latency for r in success_results]
        tokens = [r.tokens_generated for r in success_results]
        total_tokens = sum(tokens)
        latencies.sort()

        print(f"\nLatency (successful requests):")
        print(f"  Min:    {min(latencies):.2f}s")
        print(f"  Median: {latencies[len(latencies)//2]:.2f}s")
        print(f"  P90:    {latencies[int(len(latencies)*0.9)]:.2f}s")
        print(f"  P99:    {latencies[int(len(latencies)*0.99)]:.2f}s")
        print(f"  Max:    {max(latencies):.2f}s")
        print(f"  Avg:    {sum(latencies)/len(latencies):.2f}s")
        print(f"\nTokens generated:  {total_tokens}")
        print(f"Token throughput:   {total_tokens / total_time:.1f} tokens/s")

    if fail_results:
        print(f"\nErrors:")
        error_counts = {}
        for r in fail_results:
            err = r.error or "unknown"
            # Truncate long errors
            err_key = err[:100]
            error_counts[err_key] = error_counts.get(err_key, 0) + 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  [{count}x] {err}")

    # Get final cluster status
    async with httpx.AsyncClient() as http:
        resp = await http.get(f"{scheduler_url.rstrip('/')}/cluster_status")
        final_status = resp.json()

    print(f"\nCluster status after test:")
    print(f"  Ready workers: {final_status['ready_workers']}")
    for w in final_status.get('workers', []):
        print(f"    {w['worker_id']}: {w['status']} "
              f"(reqs={w['requests_served']}, {w['hostname']}:{w['port']})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Load test for Elastic Serving")
    parser.add_argument("--scheduler-url", type=str,
                        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-detected from scheduler if not set)")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    # Auto-detect model from scheduler
    if not args.model:
        import httpx
        try:
            resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "default")
        except Exception:
            args.model = "default"

    asyncio.run(run_load_test(
        scheduler_url=args.scheduler_url,
        model=args.model,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ))


if __name__ == "__main__":
    main()

