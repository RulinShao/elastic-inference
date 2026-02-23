#!/usr/bin/env python3
"""
Prefix Caching Benchmark
=========================

Profiles the acceleration from vLLM's --enable-prefix-caching for multi-round
agentic conversations (tool-call loops).

The test simulates a realistic agentic workflow:
  1. System prompt + tool definitions (fixed prefix)
  2. User question
  3. Round 1: model generates → tool call → tool result appended
  4. Round 2: same prefix + round 1 context → model generates → tool result
  5. ...up to N rounds

With prefix caching enabled, rounds 2–N should have significantly lower
time-to-first-token (TTFT) because the KV cache for the shared prefix is
reused instead of being re-computed.

This test measures:
  - TTFT (time to first token) per round
  - Total generation time per round
  - Prompt tokens per round
  - Tokens/s per round
  - Overall speedup from prefix reuse

Usage:
    # Against the elastic serving scheduler (prefix caching ON):
    python tests/test_prefix_caching.py --scheduler-url http://localhost:8780

    # Specify model explicitly:
    python tests/test_prefix_caching.py \\
        --scheduler-url http://localhost:8780 \\
        --model /checkpoint/maestro/models/gpt-oss-120b

    # Compare with prefix caching OFF (direct vLLM without the flag):
    python tests/test_prefix_caching.py \\
        --url-prefix-on http://localhost:8780 \\
        --url-prefix-off http://WORKER_IP:8001
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Simulated agentic conversation (realistic multi-round)
# =============================================================================

SYSTEM_PROMPT = """\
You are a deep research assistant with access to web search and URL reading tools. \
Your goal is to provide thorough, accurate, and well-sourced answers by actively \
researching the web.

You have two tools:
1. search(query) — Search the web for information.
2. open_url(url) — Open and read a web page.

Follow this approach: analyze the question, search strategically, read primary \
sources, cross-reference, iterate as needed. Cite your sources."""

TOOL_NAMESPACE = """\
# Tools

## functions

namespace functions {

// Search the web using a search engine. Returns the top results with titles, URLs, and short snippets.
type search = (_: {
// The search query.
query: string
}) => any;

// Open a URL and read its full text content.
type open_url = (_: {
// The full URL to open.
url: string
}) => any;

} // namespace functions"""

# Simulated multi-round conversation turns.
# Each turn adds more context (like a real tool-call loop would).
# We use fixed "fake" tool results to ensure reproducibility.

USER_QUESTION = (
    "What are the key architectural innovations in the Mamba state space model, "
    "and how does it compare to Transformers for long-sequence tasks?"
)

FAKE_SEARCH_RESULT = """\
[1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    URL: https://arxiv.org/abs/2312.00752
    We introduce Mamba, a new architecture based on structured state space models (SSMs) \
with a selection mechanism that allows the model to filter information based on input.

[2] Mamba-2: A New Architecture for Language Models
    URL: https://arxiv.org/abs/2405.21060
    Mamba-2 simplifies the Mamba architecture while improving performance. \
It establishes connections between SSMs and structured masked attention.

[3] State Space Models vs Transformers: A Comprehensive Comparison
    URL: https://blog.ml-research.org/ssm-vs-transformers
    State space models like Mamba offer O(n) complexity for sequence length n, \
compared to O(n^2) for standard attention. This makes them attractive for long sequences.

[4] Understanding Selective State Spaces in Mamba
    URL: https://mechanistic-interpretability.org/mamba-ssm
    The key innovation in Mamba is the selective scan mechanism that makes SSM \
parameters input-dependent, allowing content-based reasoning.

[5] Mamba in Practice: Real-World Benchmarks
    URL: https://huggingface.co/blog/mamba-benchmarks
    Mamba achieves comparable performance to Transformers on standard benchmarks \
while being 5x faster at inference for long sequences."""

FAKE_URL_CONTENT = """\
Mamba: Linear-Time Sequence Modeling with Selective State Spaces

Albert Gu, Tri Dao

Abstract: Foundation models, now powering most of the exciting applications in \
deep learning, are almost universally based on the Transformer architecture and \
its core attention module. Many subquadratic-time architectures such as linear \
attention, gated convolution and recurrent models, and structured state space \
models (SSMs) have been developed to address Transformers' computational \
inefficiency on long sequences, but they have not performed as well as attention \
on important modalities such as language.

We identify that a key weakness of such models is their inability to perform \
content-based reasoning, and make several improvements. First, we allow the SSM \
parameters to be functions of the input (selection mechanism). Second, we design \
a hardware-aware parallel algorithm for efficient computation. Third, we \
integrate the SSM into a simplified architecture (Mamba) without attention or MLP blocks.

Key Innovations:
1. Selective State Spaces: Unlike traditional SSMs with fixed parameters, Mamba \
makes the state space parameters (A, B, C, delta) input-dependent. This allows \
the model to selectively propagate or forget information along the sequence based \
on content, similar to how attention selects relevant tokens.

2. Hardware-Aware Algorithm: The selective scan is implemented with a parallel \
scan algorithm that avoids materializing the full state, using kernel fusion and \
recomputation to achieve GPU memory efficiency. This is up to 3x faster than \
the theoretical compute bound would suggest.

3. Simplified Architecture: Mamba removes the need for separate attention and MLP \
blocks found in Transformers. Each Mamba block combines selective SSM with gated \
linear projections, reducing the total parameter count and improving throughput.

Results:
- Language modeling: Mamba-3B matches Transformer-3B perplexity on The Pile \
while being 5x faster at generation.
- Long sequences: Mamba scales linearly O(n) with sequence length, vs O(n^2) \
for attention. On sequences of length 1M tokens, Mamba is 100x faster.
- DNA modeling: Mamba achieves state-of-the-art on GenomicsBenchmark.
- Audio: Mamba outperforms prior SSMs on speech generation tasks."""


def build_conversation_rounds() -> List[str]:
    """Build a series of incrementally growing prompts simulating a tool-call loop.

    Returns a list of prompts where each one extends the previous with more
    conversation context (mimicking what chat.py / generate_trajectories.py does).
    """
    # We build raw prompt strings to bypass needing the actual tokenizer for the test.
    # This matches the Harmony format that apply_chat_template produces.

    base = (
        f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
        f"Knowledge cutoff: 2024-06\n"
        f"Current date: 2026-02-22\n\n"
        f"Reasoning: medium\n\n"
        f"# Valid channels: analysis, commentary, final. "
        f"Channel must be included for every message.\n"
        f"Calls to these tools must go to the commentary channel: 'functions'.<|end|>"
        f"<|start|>developer<|message|># Instructions\n\n{SYSTEM_PROMPT}\n\n{TOOL_NAMESPACE}<|end|>"
    )

    # Round 1: system + tools + user question (just the base prompt)
    round1 = (
        f"{base}"
        f"<|start|>user<|message|>{USER_QUESTION}<|end|>"
        f"<|start|>assistant"
    )

    # Round 2: + assistant tool call + tool result (search)
    round2 = (
        f"{base}"
        f"<|start|>user<|message|>{USER_QUESTION}<|end|>"
        f'<|start|>assistant<|channel|>analysis<|message|>I need to search for information about the Mamba architecture.<|end|>'
        f'<|start|>assistant to=functions.search<|channel|>commentary json<|message|>'
        f'{{"query": "Mamba state space model architecture innovations"}}<|call|>'
        f'<|start|>functions.search to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(FAKE_SEARCH_RESULT)}<|end|>'
        f"<|start|>assistant"
    )

    # Round 3: + another tool call + tool result (open_url)
    round3 = (
        f"{base}"
        f"<|start|>user<|message|>{USER_QUESTION}<|end|>"
        f'<|start|>assistant<|channel|>analysis<|message|>I need to search for information about the Mamba architecture.<|end|>'
        f'<|start|>assistant to=functions.search<|channel|>commentary json<|message|>'
        f'{{"query": "Mamba state space model architecture innovations"}}<|call|>'
        f'<|start|>functions.search to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(FAKE_SEARCH_RESULT)}<|end|>'
        f'<|start|>assistant<|channel|>analysis<|message|>Let me read the original Mamba paper for details.<|end|>'
        f'<|start|>assistant to=functions.open_url<|channel|>commentary json<|message|>'
        f'{{"url": "https://arxiv.org/abs/2312.00752"}}<|call|>'
        f'<|start|>functions.open_url to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(FAKE_URL_CONTENT)}<|end|>'
        f"<|start|>assistant"
    )

    # Round 4: + yet another search (comparison query)
    comparison_search = """\
[1] Transformers vs SSMs: When to Use Which Architecture
    URL: https://ml-survey.org/transformers-vs-ssms
    Transformers excel at tasks requiring precise token-to-token attention, \
while SSMs like Mamba are superior for tasks requiring long-range dependencies.

[2] Scaling Laws for State Space Models
    URL: https://arxiv.org/abs/2402.12345
    SSMs follow similar scaling laws to Transformers but with better \
compute efficiency at longer context lengths."""

    round4 = (
        f"{base}"
        f"<|start|>user<|message|>{USER_QUESTION}<|end|>"
        f'<|start|>assistant<|channel|>analysis<|message|>I need to search for information about the Mamba architecture.<|end|>'
        f'<|start|>assistant to=functions.search<|channel|>commentary json<|message|>'
        f'{{"query": "Mamba state space model architecture innovations"}}<|call|>'
        f'<|start|>functions.search to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(FAKE_SEARCH_RESULT)}<|end|>'
        f'<|start|>assistant<|channel|>analysis<|message|>Let me read the original Mamba paper for details.<|end|>'
        f'<|start|>assistant to=functions.open_url<|channel|>commentary json<|message|>'
        f'{{"url": "https://arxiv.org/abs/2312.00752"}}<|call|>'
        f'<|start|>functions.open_url to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(FAKE_URL_CONTENT)}<|end|>'
        f'<|start|>assistant<|channel|>analysis<|message|>Now let me search for a direct comparison with Transformers.<|end|>'
        f'<|start|>assistant to=functions.search<|channel|>commentary json<|message|>'
        f'{{"query": "Mamba vs Transformer comparison long sequences benchmark"}}<|call|>'
        f'<|start|>functions.search to=assistant<|channel|>commentary<|message|>'
        f'{json.dumps(comparison_search)}<|end|>'
        f"<|start|>assistant"
    )

    return [round1, round2, round3, round4]


# =============================================================================
# Benchmark
# =============================================================================


@dataclass
class RoundResult:
    round_num: int
    prompt_tokens: int
    new_tokens: int  # tokens added since previous round
    completion_tokens: int
    ttft_ms: float  # time to first token (ms)
    total_time_s: float
    tokens_per_sec: float


@dataclass
class BenchmarkResult:
    label: str
    rounds: List[RoundResult] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return len(text) // 4


def run_round(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 128,
    prev_prompt_len: int = 0,
) -> RoundResult:
    """Send a completion request and measure timing."""
    prompt_tokens = estimate_tokens(prompt)
    new_tokens = prompt_tokens - prev_prompt_len

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,  # deterministic for fair comparison
        "stream": True,
    }

    # Stream to measure TTFT
    ttft = None
    chunks = []
    t_start = time.perf_counter()

    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST",
            f"{base_url}/v1/completions",
            json=payload,
            headers={"Authorization": "Bearer EMPTY"},
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        text = data["choices"][0].get("text", "")
                        if text and ttft is None:
                            ttft = (time.perf_counter() - t_start) * 1000  # ms
                        chunks.append(text)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

    t_end = time.perf_counter()
    total_time = t_end - t_start
    full_text = "".join(chunks)
    completion_tokens = estimate_tokens(full_text)

    if ttft is None:
        ttft = total_time * 1000

    tps = completion_tokens / max(total_time, 0.001)

    return RoundResult(
        round_num=0,  # filled by caller
        prompt_tokens=prompt_tokens,
        new_tokens=new_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft,
        total_time_s=total_time,
        tokens_per_sec=tps,
    )


def run_benchmark(
    base_url: str,
    model: str,
    prompts: List[str],
    max_tokens: int = 128,
    warmup: bool = True,
    label: str = "",
    num_repeats: int = 3,
) -> BenchmarkResult:
    """Run the multi-round benchmark, repeating for statistical reliability."""
    result = BenchmarkResult(label=label)

    # Warmup: send a short request to prime the engine
    if warmup:
        print(f"  Warming up ({label})...")
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": "Hello",
                        "max_tokens": 1,
                        "temperature": 0.0,
                    },
                    headers={"Authorization": "Bearer EMPTY"},
                )
                resp.raise_for_status()
        except Exception as e:
            print(f"  Warmup failed: {e}")

    # Aggregate results across repeats
    all_rounds = {i: [] for i in range(len(prompts))}

    for rep in range(num_repeats):
        if num_repeats > 1:
            print(f"  Run {rep + 1}/{num_repeats}...")
        prev_len = 0
        for i, prompt in enumerate(prompts):
            r = run_round(base_url, model, prompt, max_tokens, prev_len)
            r.round_num = i + 1
            all_rounds[i].append(r)
            prev_len = r.prompt_tokens

    # Average results
    for i in range(len(prompts)):
        rounds = all_rounds[i]
        avg = RoundResult(
            round_num=i + 1,
            prompt_tokens=rounds[0].prompt_tokens,
            new_tokens=rounds[0].new_tokens,
            completion_tokens=int(sum(r.completion_tokens for r in rounds) / len(rounds)),
            ttft_ms=sum(r.ttft_ms for r in rounds) / len(rounds),
            total_time_s=sum(r.total_time_s for r in rounds) / len(rounds),
            tokens_per_sec=sum(r.tokens_per_sec for r in rounds) / len(rounds),
        )
        result.rounds.append(avg)

    return result


# =============================================================================
# Reporting
# =============================================================================


def print_results(results: List[BenchmarkResult]):
    """Pretty-print benchmark results."""

    for res in results:
        print(f"\n{'=' * 80}")
        print(f"  {res.label}")
        print(f"{'=' * 80}")
        print(
            f"  {'Round':<8} {'Prompt':>8} {'New':>8} {'Gen':>6} "
            f"{'TTFT(ms)':>10} {'Total(s)':>10} {'Tok/s':>8}"
        )
        print(f"  {'-' * 68}")
        for r in res.rounds:
            print(
                f"  {r.round_num:<8} {r.prompt_tokens:>8} {r.new_tokens:>8} "
                f"{r.completion_tokens:>6} {r.ttft_ms:>10.1f} "
                f"{r.total_time_s:>10.2f} {r.tokens_per_sec:>8.1f}"
            )

    # TTFT comparison across rounds
    if len(results) >= 1:
        res = results[0]
        if len(res.rounds) >= 2:
            print(f"\n{'=' * 80}")
            print("  Prefix Caching Effect (TTFT across rounds)")
            print(f"{'=' * 80}")
            r1 = res.rounds[0]
            print(f"  Round 1 (cold):  TTFT = {r1.ttft_ms:>8.1f} ms  "
                  f"({r1.prompt_tokens} prompt tokens)")
            for r in res.rounds[1:]:
                speedup = r1.ttft_ms / max(r.ttft_ms, 0.001)
                # With prefix caching: TTFT should scale with new_tokens, not total
                # Without: TTFT scales with total prompt_tokens
                ttft_per_prompt_tok = r.ttft_ms / max(r.prompt_tokens, 1)
                ttft_per_new_tok = r.ttft_ms / max(r.new_tokens, 1) if r.new_tokens > 0 else 0
                print(
                    f"  Round {r.round_num} (warm):  TTFT = {r.ttft_ms:>8.1f} ms  "
                    f"({r.prompt_tokens} prompt, +{r.new_tokens} new)  "
                    f"TTFT/prompt_tok = {ttft_per_prompt_tok:.3f} ms  "
                    f"TTFT/new_tok = {ttft_per_new_tok:.3f} ms"
                )

            # Summary
            avg_warm_ttft = sum(r.ttft_ms for r in res.rounds[1:]) / len(res.rounds[1:])
            avg_warm_prompt = sum(r.prompt_tokens for r in res.rounds[1:]) / len(res.rounds[1:])
            avg_warm_new = sum(r.new_tokens for r in res.rounds[1:]) / len(res.rounds[1:])
            print()
            print(f"  Avg warm TTFT:    {avg_warm_ttft:.1f} ms")
            print(f"  Avg warm prompt:  {avg_warm_prompt:.0f} tokens")
            print(f"  Avg warm new:     {avg_warm_new:.0f} tokens")
            if avg_warm_prompt > 0:
                # If TTFT scales with new tokens (not total), prefix caching is working
                ratio = avg_warm_new / avg_warm_prompt
                print(f"  New/Total ratio:  {ratio:.2%}")
                print()
                if avg_warm_ttft < r1.ttft_ms * 0.8:
                    print(
                        f"  ✅ Prefix caching appears EFFECTIVE: warm TTFT "
                        f"({avg_warm_ttft:.0f}ms) << cold TTFT ({r1.ttft_ms:.0f}ms) "
                        f"despite larger prompts."
                    )
                else:
                    print(
                        f"  ⚠️  Prefix caching may NOT be active: warm TTFT "
                        f"({avg_warm_ttft:.0f}ms) ≈ cold TTFT ({r1.ttft_ms:.0f}ms). "
                        f"Check --enable-prefix-caching flag."
                    )

    # A/B comparison if two results
    if len(results) == 2:
        on, off = results[0], results[1]
        print(f"\n{'=' * 80}")
        print(f"  A/B Comparison: {on.label} vs {off.label}")
        print(f"{'=' * 80}")
        print(f"  {'Round':<8} {'TTFT ON(ms)':>12} {'TTFT OFF(ms)':>13} {'Speedup':>10}")
        print(f"  {'-' * 50}")
        for r_on, r_off in zip(on.rounds, off.rounds):
            speedup = r_off.ttft_ms / max(r_on.ttft_ms, 0.001)
            print(
                f"  {r_on.round_num:<8} {r_on.ttft_ms:>12.1f} "
                f"{r_off.ttft_ms:>13.1f} {speedup:>9.2f}x"
            )
        avg_on = sum(r.ttft_ms for r in on.rounds[1:]) / max(len(on.rounds[1:]), 1)
        avg_off = sum(r.ttft_ms for r in off.rounds[1:]) / max(len(off.rounds[1:]), 1)
        overall = avg_off / max(avg_on, 0.001)
        print(f"  {'':8} {'':>12} {'':>13} {'':>10}")
        print(f"  {'Avg warm':<8} {avg_on:>12.1f} {avg_off:>13.1f} {overall:>9.2f}x")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefix caching for multi-round agentic conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Profile current server (should have prefix caching ON):
  python tests/test_prefix_caching.py --scheduler-url http://localhost:8780

  # A/B comparison (requires two servers):
  python tests/test_prefix_caching.py \\
      --url-prefix-on http://localhost:8780 \\
      --url-prefix-off http://WORKER_IP:8001
""",
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
        help="Scheduler URL (with prefix caching enabled)",
    )
    parser.add_argument(
        "--url-prefix-on",
        type=str,
        default=None,
        help="URL for server WITH prefix caching (for A/B test)",
    )
    parser.add_argument(
        "--url-prefix-off",
        type=str,
        default=None,
        help="URL for server WITHOUT prefix caching (for A/B test)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name/path")
    parser.add_argument(
        "--max-tokens", type=int, default=128,
        help="Max tokens to generate per round (default: 128)",
    )
    parser.add_argument(
        "--num-repeats", type=int, default=3,
        help="Number of times to repeat each benchmark for averaging (default: 3)",
    )
    args = parser.parse_args()

    # Auto-detect model
    if not args.model:
        url = args.url_prefix_on or args.scheduler_url
        try:
            resp = httpx.get(f"{url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "")
        except Exception:
            pass
        if not args.model:
            try:
                resp = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=5)
                models = resp.json().get("data", [])
                if models:
                    args.model = models[0].get("id", "default")
            except Exception:
                args.model = "default"
        print(f"Model: {args.model}")

    # Build prompts
    prompts = build_conversation_rounds()
    print(f"Built {len(prompts)} conversation rounds")
    for i, p in enumerate(prompts):
        print(f"  Round {i + 1}: ~{estimate_tokens(p)} tokens")

    results = []

    if args.url_prefix_on and args.url_prefix_off:
        # A/B comparison
        print(f"\n--- Benchmarking WITH prefix caching: {args.url_prefix_on} ---")
        res_on = run_benchmark(
            args.url_prefix_on.rstrip("/"), args.model, prompts,
            args.max_tokens, label="Prefix Caching ON",
            num_repeats=args.num_repeats,
        )
        results.append(res_on)

        print(f"\n--- Benchmarking WITHOUT prefix caching: {args.url_prefix_off} ---")
        res_off = run_benchmark(
            args.url_prefix_off.rstrip("/"), args.model, prompts,
            args.max_tokens, label="Prefix Caching OFF",
            num_repeats=args.num_repeats,
        )
        results.append(res_off)
    else:
        # Single server profiling
        url = args.scheduler_url.rstrip("/")
        print(f"\n--- Benchmarking: {url} ---")
        res = run_benchmark(
            url, args.model, prompts, args.max_tokens,
            label="Prefix Caching (current server)",
            num_repeats=args.num_repeats,
        )
        results.append(res)

    print_results(results)


if __name__ == "__main__":
    main()


