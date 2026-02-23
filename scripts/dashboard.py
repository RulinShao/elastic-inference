#!/usr/bin/env python3
"""
Live dashboard for monitoring cluster + eval job progress.

Usage:
    python scripts/dashboard.py
    python scripts/dashboard.py --output-dir results/webshaper_full
    watch -n 5 python scripts/dashboard.py   # auto-refresh
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-url", default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--output-dir", default="results/webshaper_full")
    args = parser.parse_args()

    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")

    print("=" * 62)
    print("  Elastic Inference — Dashboard")
    print("=" * 62)
    from datetime import datetime
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Cluster ----
    try:
        resp = httpx.get(f"{args.scheduler_url}/cluster_status", timeout=3)
        status = resp.json()
        model = status.get("model", "?").split("/")[-1]
        ready = status.get("ready_workers", 0)
        loading = status.get("loading_workers", 0)
        pending = status.get("pending_slurm_jobs", 0)
        print(f"  -- Cluster: {model} --")
        print(f"  Workers: {ready} ready, {loading} loading, {pending} pending SLURM")
        print()

        # vLLM metrics per worker
        workers = [w for w in status.get("workers", []) if w["status"] == "READY"]
        if workers:
            print(f"  {'Host':<22} {'Running':>8} {'Waiting':>8} {'KV Cache':>10}")
            print(f"  {'-'*52}")
            for w in workers:
                try:
                    m = httpx.get(f"http://{w['ip_address']}:{w['port']}/metrics", timeout=2)
                    text = m.text
                    run = wait = kv = "?"
                    for line in text.split("\n"):
                        if line.startswith("vllm:num_requests_running{"):
                            run = line.split()[-1]
                        elif line.startswith("vllm:num_requests_waiting{"):
                            wait = line.split()[-1]
                        elif line.startswith("vllm:kv_cache_usage_perc{"):
                            kv = f"{float(line.split()[-1])*100:.1f}%"
                    print(f"  {w['hostname']:<22} {run:>8} {wait:>8} {kv:>10}")
                except Exception:
                    print(f"  {w['hostname']:<22} {'?':>8} {'?':>8} {'?':>10}")
    except Exception:
        print(f"  Scheduler unreachable at {args.scheduler_url}")

    print()

    # ---- Eval Progress ----
    if not os.path.exists(traj_file):
        print(f"  No eval results at {traj_file}")
        return

    trajs = []
    with open(traj_file) as f:
        for line in f:
            try:
                trajs.append(json.loads(line))
            except Exception:
                pass

    n = len(trajs)
    qids = set(t["qid"] for t in trajs)
    n_q = len(qids)

    # Tool stats
    tool_counts = Counter()
    total_tools = 0
    for t in trajs:
        for tc in t.get("tool_calls", []):
            tool_counts[tc["tool"]] += 1
            total_tools += 1

    # Time stats
    times = [t.get("latency_s", 0) for t in trajs if t.get("latency_s", 0) > 0]
    avg_time = sum(times) / max(len(times), 1)
    total_time_h = sum(times) / 3600

    # Status
    statuses = Counter(t.get("status", "?") for t in trajs)
    n_boxed = sum(1 for t in trajs if t.get("boxed_answer"))

    # Quick accuracy (substring match)
    by_qid = {}
    for t in trajs:
        qid = t["qid"]
        if qid not in by_qid:
            by_qid[qid] = {"ref": t.get("reference_answer", ""), "trajs": []}
        by_qid[qid]["trajs"].append(t)

    correct = 0
    pass_count = 0
    for qid, data in by_qid.items():
        ref = data["ref"].strip().lower().replace(",", "").replace(" ", "")
        any_match = False
        for t in data["trajs"]:
            ans = (t.get("boxed_answer") or t.get("answer", ""))
            ans = ans.strip().lower().replace(",", "").replace(" ", "")
            if ref and ans and (ref in ans or ans in ref):
                correct += 1
                any_match = True
        if any_match:
            pass_count += 1

    print(f"  -- Eval Progress ({args.output_dir}) --")
    print(f"  Trajectories:  {n}/2000 ({n/20:.0f}%)")
    print(f"  Questions:     {n_q}/500")
    print(f"  Avg time/traj: {avg_time:.0f}s")
    print(f"  Total GPU time: {total_time_h:.1f}h")
    print(f"  Boxed answers: {n_boxed}/{n} ({n_boxed/max(n,1)*100:.0f}%)")
    print(f"  Status: {dict(statuses)}")
    print()

    # Tool breakdown
    print(f"  -- Tool Calls ({total_tools} total, {total_tools/max(n,1):.1f}/traj) --")
    for tool, cnt in tool_counts.most_common():
        bar_len = int(cnt / max(total_tools, 1) * 30)
        bar = "\u2588" * bar_len
        print(f"  {tool:30s} {cnt:5d} ({cnt/max(total_tools,1)*100:4.0f}%) {bar}")
    print()

    # Accuracy estimate
    print(f"  -- Quick Accuracy (substring match, not LLM judge) --")
    print(f"  Traj accuracy:  {correct}/{n} ({correct/max(n,1)*100:.1f}%)")
    print(f"  pass@k (est):   {pass_count}/{n_q} ({pass_count/max(n_q,1)*100:.1f}%)")
    print()

    # ETA
    if n > 0 and avg_time > 0:
        remaining = 2000 - n
        # Rough: with concurrency, actual wall time is much less
        eta_s = remaining * avg_time / 32  # assume ~32 effective concurrency
        eta_m = eta_s / 60
        print(f"  ETA (rough):    ~{eta_m:.0f} min ({remaining} trajectories left)")


if __name__ == "__main__":
    main()

