#!/usr/bin/env python3
"""
SFT Trajectory Generation with Deep Research (gpt-oss-120b)
=============================================================

Uses the gpt-oss-120b model's native Harmony channel format for tool calling.
Generates via /v1/completions with tokenizer.apply_chat_template.

Tools: Serper (search) and Jina (web reader / URL fetcher).
Output: JSONL with full multi-turn trajectories.

Usage:
    python scripts/generate_trajectories.py \
        --scheduler-url http://localhost:8780 \
        --dataset sample --num-samples 3 \
        --output-dir results/trajectories
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import httpx

dotenv.load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")

# Tool definitions for tokenizer.apply_chat_template
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the web for information. Returns top results "
                "with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a URL and read its content. Returns the page text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to open",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a helpful research assistant. You can search the web and read web pages to find accurate, detailed answers to questions.

When answering a question:
1. Think step-by-step about what information you need.
2. Use the search tool to find relevant sources.
3. Use open_url to read promising results in detail.
4. Synthesize information from multiple sources.
5. Provide a clear, well-sourced answer.

Always verify claims across multiple sources when possible."""


# =============================================================================
# Tool Implementations
# =============================================================================

async def tool_search(query: str, http_client: httpx.AsyncClient) -> str:
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY not set"
    try:
        resp = await http_client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": 10},
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("organic", [])
        if not results:
            return f"No search results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title', '')}\n    URL: {r.get('link', '')}\n    {r.get('snippet', '')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


async def tool_open_url(url: str, http_client: httpx.AsyncClient) -> str:
    if not JINA_API_KEY:
        try:
            resp = await http_client.get(url, follow_redirects=True, timeout=30)
            return resp.text[:20000]
        except Exception as e:
            return f"Error fetching URL: {e}"
    try:
        resp = await http_client.get(
            f"https://r.jina.ai/{url}",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Accept": "text/plain",
                "X-Return-Format": "text",
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.text
        if len(content) > 30000:
            content = content[:30000] + "\n\n[... content truncated ...]"
        return content
    except Exception as e:
        return f"Error reading URL: {e}"


async def execute_tool(name: str, args: dict, http_client: httpx.AsyncClient) -> str:
    if name == "search":
        query = args.get("query", "")
        if not query:
            return "Error: search requires a 'query' parameter"
        return await tool_search(query, http_client)
    elif name == "open_url":
        url = args.get("url", "")
        # Model sometimes sends a query string instead of a URL — redirect to search
        if not url or not url.startswith(("http://", "https://")):
            query = url or args.get("query", "")
            if query:
                return f"Error: open_url requires a valid URL starting with http:// or https://. Got: '{query[:100]}'. Use the search tool instead."
            return "Error: open_url requires a 'url' parameter with a valid URL"
        return await tool_open_url(url, http_client)
    return f"Unknown tool: {name}"


# =============================================================================
# Harmony format parsing
# =============================================================================

def parse_tool_call_from_raw(text: str) -> Optional[Tuple[str, dict]]:
    """Parse a tool call from the raw generated text.

    The model generates:
      ... to=functions.TOOLNAME<|channel|>commentary json<|message|>"..."<|call|>
    or:
      ... to=functions.TOOLNAME<|channel|>commentary code<|message|>{...}<|call|>

    We also handle the case where the model generates inline tool calls
    in the format: `assistantcommentary to=functions.search json{...}`
    """
    # Pattern 1: Harmony special tokens
    m = re.search(r'to=functions\.(\w+)', text)
    if m:
        tool_name = m.group(1)
        # Try to find JSON args after <|message|> or after json/code marker
        msg_match = re.search(r'<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)', text, re.DOTALL)
        if msg_match:
            args_str = msg_match.group(1).strip()
        else:
            # Fallback: find JSON after the tool name
            after = text[m.end():]
            # Look for json{...} or code{...} pattern
            json_match = re.search(r'(?:json|code)\s*(\{.*?\})\s*$', after, re.DOTALL)
            if json_match:
                args_str = json_match.group(1)
            else:
                args_str = after.strip()

        # Clean up: remove quotes around JSON string
        if args_str.startswith('"') and args_str.endswith('"'):
            try:
                args_str = json.loads(args_str)  # unescape
            except Exception:
                args_str = args_str[1:-1]

        try:
            args = json.loads(args_str)
            return tool_name, args
        except json.JSONDecodeError:
            # Try to extract just the query
            query_match = re.search(r'"query"\s*:\s*"([^"]*)"', args_str)
            if query_match:
                return tool_name, {"query": query_match.group(1)}
            url_match = re.search(r'"url"\s*:\s*"([^"]*)"', args_str)
            if url_match:
                return tool_name, {"url": url_match.group(1)}

    return None


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
    max_rounds: int = 15,
    max_gen_tokens: int = 8192,
) -> Dict[str, Any]:
    """Generate a single research trajectory."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    round_num = 0
    total_tool_calls = 0

    # <|call|> marks end of tool call; <|end|> marks end of assistant turn
    stop_tokens = ["<|call|>", "<|end|>", "<|endoftext|>"]

    while round_num < max_rounds:
        round_num += 1

        # On the last round, don't stop at <|call|> — force a final answer
        is_last_round = (round_num == max_rounds)
        current_stops = ["<|end|>", "<|endoftext|>"] if is_last_round else stop_tokens

        # Build prompt
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(prompt))
        suffix = " (final)" if is_last_round else ""
        print(f"  [qid={qid}] Round {round_num}/{max_rounds} ({prompt_tokens} tokens){suffix}")

        # Generate via /v1/completions
        try:
            resp = await openai_http.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_gen_tokens,
                    "temperature": 0.7,
                    "stop": current_stops,
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["text"]
            finish_reason = data["choices"][0].get("finish_reason", "")
        except Exception as e:
            print(f"  [qid={qid}] Generation error: {e}")
            messages.append({"role": "assistant", "content": f"[Error: {e}]"})
            break

        # Check if this is a tool call (stopped at <|call|>) or final answer
        tool_call = parse_tool_call_from_raw(raw_text) if not is_last_round else None

        if tool_call:
            tool_name, tool_args = tool_call
            print(f"  [qid={qid}]   Tool: {tool_name}({json.dumps(tool_args)[:120]})")

            # Record the tool call in messages
            call_id = f"call_{round_num}"
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                }],
            })

            # Execute tool
            result = await execute_tool(tool_name, tool_args, http_client)
            total_tool_calls += 1

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
            })
            continue
        else:
            # Final answer — extract content from harmony channel format
            # Raw text looks like: "analysis...assistantfinal THE ANSWER"
            # or: "analysis...assistantcommentary...assistantfinal THE ANSWER"
            content = raw_text
            reasoning = ""

            # Extract the final channel content (everything after "assistantfinal" or just "final")
            final_match = re.search(r'(?:assistant)?final\s*(.*?)$', content, re.DOTALL)
            if final_match:
                final_content = final_match.group(1).strip()
                # Everything before "final" is reasoning/analysis
                reasoning = content[:final_match.start()].strip()
            else:
                # No "final" channel — treat entire text as content
                # Remove "analysis" prefix if present
                if content.startswith("analysis"):
                    reasoning_match = re.match(r'analysis(.*?)(?:assistant|$)', content, re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()
                        final_content = content[reasoning_match.end():].strip()
                    else:
                        final_content = content[len("analysis"):].strip()
                else:
                    final_content = content.strip()

            # Clean up reasoning: remove "assistant" prefixes
            reasoning = re.sub(r'\bassistant\b', '', reasoning).strip()
            # Remove repeated channel markers
            reasoning = re.sub(r'\b(analysis|commentary)\b', '', reasoning).strip()

            msg = {"role": "assistant", "content": final_content}
            if reasoning:
                msg["reasoning_content"] = reasoning

            messages.append(msg)
            print(f"  [qid={qid}] Final answer ({round_num} rounds, {total_tool_calls} tools, {len(final_content)} chars)")
            break

    return {
        "qid": qid,
        "question": question,
        "messages": messages,
        "num_rounds": round_num,
        "num_tool_calls": total_tool_calls,
    }


async def run_generation(
    scheduler_url: str,
    model: str,
    dataset_name: str,
    num_samples: int,
    concurrency: int,
    output_dir: str,
    max_rounds: int,
    max_gen_tokens: int,
):
    from transformers import AutoTokenizer

    print(f"Loading tokenizer for {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=300)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "sample":
        data = [
            {"qid": 1, "question": "What were the key findings of the most recent IPCC report on climate change, and how do they compare to the predictions made in the 2018 special report on 1.5°C warming?"},
            {"qid": 2, "question": "Who is the current CEO of Anthropic, when was the company founded, and what is their stated mission regarding AI safety?"},
            {"qid": 3, "question": "What is the latest breakthrough in room-temperature superconductivity research as of 2024, and what is the scientific consensus on the LK-99 claims?"},
            {"qid": 4, "question": "Describe the architecture and key innovations of the Mamba state space model. How does it compare to Transformers in terms of computational complexity for long sequences?"},
            {"qid": 5, "question": "What is the current state of nuclear fusion energy research? Describe the NIF's ignition achievement and the ITER project timeline."},
        ]
    else:
        try:
            sys.path.insert(0, "/tmp/OpenResearcher")
            from data_utils import load_dataset
            data = load_dataset(dataset_name)
        except Exception as e:
            print(f"Could not load dataset '{dataset_name}': {e}")
            return

    if num_samples > 0:
        data = data[:num_samples]

    # Wait for workers
    async with httpx.AsyncClient() as tmp:
        resp = await tmp.get(f"{scheduler_url.rstrip('/')}/cluster_status")
        status = resp.json()
    print(f"Cluster: {status['ready_workers']} ready workers")

    if status["ready_workers"] == 0:
        print("⚠  No ready workers! Waiting...")
        for _ in range(120):
            await asyncio.sleep(5)
            async with httpx.AsyncClient() as tmp:
                resp = await tmp.get(f"{scheduler_url.rstrip('/')}/cluster_status")
                status = resp.json()
            if status["ready_workers"] > 0:
                print(f"✅ {status['ready_workers']} workers ready")
                break
            print(f"  Waiting... (ready={status['ready_workers']}, loading={status['loading_workers']})")
        else:
            print("Timed out.")
            return

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"trajectories_{dataset_name}.jsonl")

    completed_qids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
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
            t0 = time.time()
            try:
                result = await generate_one_trajectory(
                    question=item["question"], qid=item["qid"],
                    base_url=scheduler_url.rstrip("/"), model=model,
                    tokenizer=tokenizer, http_client=http_client,
                    openai_http=openai_http, max_rounds=max_rounds,
                    max_gen_tokens=max_gen_tokens,
                )
                result["answer_ref"] = item.get("answer", "")
                result["latency_s"] = time.time() - t0
                result["status"] = "success"
            except Exception as e:
                result = {
                    "qid": item["qid"], "question": item["question"],
                    "messages": [], "error": traceback.format_exc(),
                    "latency_s": time.time() - t0, "status": "fail",
                }
            completed += 1
            print(f"[{completed}/{len(pending)}] qid={item['qid']} "
                  f"{result['status']} rounds={result.get('num_rounds',0)} "
                  f"tools={result.get('num_tool_calls',0)} "
                  f"time={result['latency_s']:.1f}s")
            return result

    tasks = [asyncio.create_task(process_one(item)) for item in pending]
    with open(output_file, "a") as writer:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.flush()

    await http_client.aclose()
    await openai_http.aclose()
    print(f"\n✅ Done! {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--scheduler-url", type=str,
                        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"))
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="sample")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=15)
    parser.add_argument("--max-gen-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=str, default="results/trajectories")
    args = parser.parse_args()

    if not args.model:
        try:
            resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5)
            args.model = resp.json().get("model", "default")
        except Exception:
            args.model = "default"

    asyncio.run(run_generation(
        scheduler_url=args.scheduler_url, model=args.model,
        dataset_name=args.dataset, num_samples=args.num_samples,
        concurrency=args.concurrency, output_dir=args.output_dir,
        max_rounds=args.max_rounds, max_gen_tokens=args.max_gen_tokens,
    ))


if __name__ == "__main__":
    main()
