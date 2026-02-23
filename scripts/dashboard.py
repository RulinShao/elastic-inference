#!/usr/bin/env python3
"""
Live dashboard for monitoring cluster + eval job progress.

Usage:
    python scripts/dashboard.py                  # auto-refresh every 10s
    python scripts/dashboard.py --once           # print once and exit
    python scripts/dashboard.py --interval 5     # refresh every 5s
"""

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

# =============================================================================
# ANSI colors
# =============================================================================

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
WHITE = "\033[97m"
GRAY = "\033[90m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_RED = "\033[41m"


def bar(value, total, width=25, fill_color=GREEN, empty_color=GRAY):
    if total == 0:
        return f"{empty_color}{'░' * width}{RESET}"
    filled = int(value / total * width)
    return f"{fill_color}{'█' * filled}{empty_color}{'░' * (width - filled)}{RESET}"


def pct(value, total):
    if total == 0:
        return "  -"
    return f"{value / total * 100:5.1f}%"


def render(args):
    """Render one frame of the dashboard."""
    traj_file = os.path.join(args.output_dir, "trajectories.jsonl")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    print()
    print(f"  {BOLD}{CYAN}╔══════════════════════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}{CYAN}║         Elastic Inference — Live Dashboard          ║{RESET}")
    print(f"  {BOLD}{CYAN}╚══════════════════════════════════════════════════════╝{RESET}")
    print(f"  {DIM}{now}{RESET}")
    print()

    # ---- Cluster ----
    try:
        resp = httpx.get(f"{args.scheduler_url}/cluster_status", timeout=3)
        status = resp.json()
        model = status.get("model", "?").split("/")[-1]
        ready = status.get("ready_workers", 0)
        loading = status.get("loading_workers", 0)
        pending = status.get("pending_slurm_jobs", 0)

        print(f"  {BOLD}{WHITE}Cluster{RESET}  {DIM}{model}{RESET}")
        w_parts = []
        if ready:
            w_parts.append(f"{GREEN}{ready} ready{RESET}")
        if loading:
            w_parts.append(f"{YELLOW}{loading} loading{RESET}")
        if pending:
            w_parts.append(f"{DIM}{pending} pending{RESET}")
        print(f"  Workers: {', '.join(w_parts)}")
        print()

        # vLLM per-worker
        workers = [w for w in status.get("workers", []) if w["status"] == "READY"]
        if workers:
            print(f"  {DIM}{'Host':<22} {'Running':>8} {'Waiting':>8} {'KV Cache':>10}{RESET}")
            for w in workers:
                try:
                    m = httpx.get(f"http://{w['ip_address']}:{w['port']}/metrics", timeout=2)
                    run = wait = kv_val = "?"
                    for line in m.text.split("\n"):
                        if line.startswith("vllm:num_requests_running{"):
                            run = line.split()[-1]
                        elif line.startswith("vllm:num_requests_waiting{"):
                            wait = line.split()[-1]
                        elif line.startswith("vllm:kv_cache_usage_perc{"):
                            kv_val = float(line.split()[-1]) * 100

                    kv_str = f"{kv_val:.1f}%" if isinstance(kv_val, float) else "?"
                    run_color = GREEN if float(run) > 0 else DIM
                    wait_color = YELLOW if float(wait) > 0 else DIM
                    kv_color = RED if isinstance(kv_val, float) and kv_val > 80 else GREEN
                    print(f"  {BOLD}{w['hostname']:<22}{RESET} {run_color}{run:>8}{RESET} {wait_color}{wait:>8}{RESET} {kv_color}{kv_str:>10}{RESET}")
                except Exception:
                    print(f"  {w['hostname']:<22} {'?':>8} {'?':>8} {'?':>10}")
    except Exception:
        print(f"  {RED}Scheduler unreachable at {args.scheduler_url}{RESET}")

    print()

    # ---- Eval Progress ----
    if not os.path.exists(traj_file):
        print(f"  {DIM}No eval results at {traj_file}{RESET}")
        return

    trajs = []
    with open(traj_file) as f:
        for line in f:
            try:
                trajs.append(json.loads(line))
            except Exception:
                pass

    n = len(trajs)
    target = 2000
    qids = set(t["qid"] for t in trajs)
    n_q = len(qids)

    # Tool stats
    tool_counts = Counter()
    total_tools = 0
    for t in trajs:
        for tc in t.get("tool_calls", []):
            tool_counts[tc["tool"]] += 1
            total_tools += 1

    times = [t.get("latency_s", 0) for t in trajs if t.get("latency_s", 0) > 0]
    avg_time = sum(times) / max(len(times), 1)
    total_time_h = sum(times) / 3600
    statuses = Counter(t.get("status", "?") for t in trajs)
    n_boxed = sum(1 for t in trajs if t.get("boxed_answer"))
    n_err = statuses.get("error", 0)

    # Accuracy
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

    # Progress section
    print(f"  {BOLD}{WHITE}Eval Progress{RESET}")
    print(f"  {bar(n, target)} {BOLD}{n}{RESET}/{target} trajectories ({n/max(target,1)*100:.0f}%)")
    print(f"  {bar(n_q, 500, fill_color=BLUE)} {BOLD}{n_q}{RESET}/500 questions")
    print()
    print(f"  {DIM}Avg time/traj:{RESET}  {BOLD}{avg_time:.0f}s{RESET}    "
          f"{DIM}Total GPU:{RESET} {BOLD}{total_time_h:.1f}h{RESET}    "
          f"{DIM}Boxed:{RESET} {n_boxed}/{n} ({n_boxed/max(n,1)*100:.0f}%)"
          f"{'   ' + RED + str(n_err) + ' errors' + RESET if n_err else ''}")
    print()

    # Tool breakdown
    print(f"  {BOLD}{WHITE}Tool Calls{RESET}  {DIM}{total_tools} total, {total_tools/max(n,1):.1f}/traj{RESET}")
    tool_colors = {
        "browser.search": CYAN,
        "browser.open": BLUE,
        "browser.find": MAGENTA,
        "functions.paper_search": YELLOW,
        "functions.pubmed_search": GREEN,
    }
    for tool, cnt in tool_counts.most_common(6):
        tc = tool_colors.get(tool, DIM)
        pct_val = cnt / max(total_tools, 1) * 100
        bar_len = int(pct_val / 100 * 25)
        print(f"  {tc}{'█' * bar_len}{'░' * (25 - bar_len)}{RESET} "
              f"{tc}{tool:<30}{RESET} {BOLD}{cnt:>5}{RESET} {DIM}({pct_val:4.0f}%){RESET}")
    print()

    # Accuracy
    traj_acc = correct / max(n, 1) * 100
    pass_k = pass_count / max(n_q, 1) * 100
    acc_color = GREEN if traj_acc >= 50 else YELLOW if traj_acc >= 30 else RED
    pass_color = GREEN if pass_k >= 70 else YELLOW if pass_k >= 50 else RED

    print(f"  {BOLD}{WHITE}Accuracy{RESET}  {DIM}(substring match estimate){RESET}")
    print(f"  Traj accuracy:  {acc_color}{BOLD}{traj_acc:.1f}%{RESET}  ({correct}/{n})")
    print(f"  pass@k (est):   {pass_color}{BOLD}{pass_k:.1f}%{RESET}  ({pass_count}/{n_q})")
    print()

    # ETA
    if n > 0 and avg_time > 0:
        remaining = target - n
        eta_m = remaining * avg_time / 32 / 60
        print(f"  {DIM}ETA ~{eta_m:.0f} min  ({remaining} trajectories left){RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live dashboard for elastic inference")
    parser.add_argument("--scheduler-url", default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--output-dir", default="results/webshaper_full")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--_render", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._render or args.once:
        # Called by watch or --once: just render one frame
        render(args)
    else:
        # Auto-invoke watch --color with the same args
        cmd = [
            "watch", "-n", str(args.interval), "--color", "-t",
            sys.executable, __file__,
            "--_render",
            "--scheduler-url", args.scheduler_url,
            "--output-dir", args.output_dir,
        ]
        try:
            os.execvp("watch", cmd)
        except FileNotFoundError:
            # Fallback if watch not available: loop manually
            import time
            while True:
                os.system("clear")
                render(args)
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
