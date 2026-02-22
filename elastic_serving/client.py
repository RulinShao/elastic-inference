#!/usr/bin/env python3
"""
Elastic Serving Client
========================

A lightweight client that talks to the Elastic Serving Scheduler.

For inference, the scheduler exposes standard OpenAI-compatible endpoints at
/v1/chat/completions, /v1/completions, /v1/models, etc. You can use the
standard `openai` Python client pointing at the scheduler URL.

This module provides:
1. SchedulerClient — for cluster management (status, scaling, etc.)
2. Helper to create an OpenAI client pointed at the scheduler.

Usage:
    from elastic_serving.client import get_openai_client, SchedulerClient

    # For inference (OpenAI-compatible)
    client = get_openai_client("http://scheduler-host:8780")
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # For cluster management
    mgr = SchedulerClient("http://scheduler-host:8780")
    status = mgr.cluster_status()
    print(status)
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import requests


class SchedulerClient:
    """Client for managing the elastic serving cluster."""

    def __init__(self, scheduler_url: str = "http://localhost:8780"):
        self.url = scheduler_url.rstrip("/")

    def _call(self, method: str, endpoint: str, data=None, timeout: float = 30.0):
        url = f"{self.url}{endpoint}"
        try:
            if method == "GET":
                resp = requests.get(url, timeout=timeout)
            elif method == "POST":
                resp = requests.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unknown method: {method}")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            print(f"Cannot connect to scheduler at {self.url}")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            print(f"Error: {detail}")
            sys.exit(1)

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def cluster_status(self) -> Dict[str, Any]:
        return self._call("GET", "/cluster_status")

    def list_workers(self) -> Dict[str, Any]:
        return self._call("GET", "/workers")

    def list_models(self) -> Dict[str, Any]:
        return self._call("GET", "/v1/models")


def get_openai_client(scheduler_url: str = "http://localhost:8780", api_key: str = "EMPTY"):
    """
    Create an OpenAI client pointing at the elastic scheduler's proxy.

    Requires: pip install openai
    """
    from openai import OpenAI
    return OpenAI(
        base_url=f"{scheduler_url.rstrip('/')}/v1",
        api_key=api_key,
    )


def get_async_openai_client(scheduler_url: str = "http://localhost:8780", api_key: str = "EMPTY"):
    """
    Create an async OpenAI client pointing at the elastic scheduler's proxy.

    Requires: pip install openai
    """
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        base_url=f"{scheduler_url.rstrip('/')}/v1",
        api_key=api_key,
    )


# =============================================================================
# CLI
# =============================================================================


def cmd_status(args):
    client = SchedulerClient(args.scheduler_url)
    status = client.cluster_status()

    tp = status.get('tensor_parallel_size', 1)
    dp = status.get('dp_per_node', 1)

    print("=" * 60)
    print("ADAPTIVE SERVING CLUSTER STATUS")
    print("=" * 60)
    print(f"Model:      {status['model']}")
    print(f"Engine:     {status['engine']}")
    print(f"Parallelism: TP={tp}, DP={dp}/node")
    print()
    print(f"Nodes: {status.get('total_nodes_active', '?')}/{status['max_nodes']} active "
          f"({status.get('pending_slurm_jobs', 0)} pending SLURM)")
    print(f"Workers: {status['total_workers']} total "
          f"({status['ready_workers']} ready, "
          f"{status['loading_workers']} loading, "
          f"{status['offline_workers']} offline)")

    if status.get("workers"):
        print()
        print(f"{'WORKER ID':<40} {'HOST':<20} {'PORT':<6} {'STATUS':<10} {'TP':<4} {'REQS'}")
        print("-" * 90)
        for w in status["workers"]:
            print(
                f"{w['worker_id']:<40} "
                f"{w['hostname']:<20} "
                f"{w['port']:<6} "
                f"{w['status']:<10} "
                f"{w.get('tensor_parallel_size', '?'):<4} "
                f"{w['requests_served']}"
            )


def cmd_models(args):
    client = SchedulerClient(args.scheduler_url)
    resp = client.list_models()
    models = resp.get("data", [])
    if not models:
        print("No models available (no ready workers)")
        return
    for m in models:
        print(f"  {m.get('id', '?')}")


def cmd_health(args):
    client = SchedulerClient(args.scheduler_url)
    ok = client.health()
    if ok:
        print("Scheduler is healthy")
    else:
        print("Scheduler is unreachable")
        sys.exit(1)


def cmd_test(args):
    """Send a test chat completion request."""
    client = get_openai_client(args.scheduler_url)
    print("Sending test request...")
    try:
        resp = client.chat.completions.create(
            model=args.model or "default",
            messages=[{"role": "user", "content": args.prompt or "Say hello!"}],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"Response: {resp.choices[0].message.content}")
        print(f"Usage: {resp.usage}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Elastic Serving CLI Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scheduler-url",
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
        help="Scheduler URL (or set ELASTIC_SERVING_URL env var)",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Show cluster status")
    subparsers.add_parser("models", help="List available models")
    subparsers.add_parser("health", help="Check scheduler health")

    sub = subparsers.add_parser("test", help="Send a test request")
    sub.add_argument("--model", type=str, default=None)
    sub.add_argument("--prompt", type=str, default="Say hello!")
    sub.add_argument("--max-tokens", type=int, default=128)
    sub.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "status": cmd_status,
        "models": cmd_models,
        "health": cmd_health,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

