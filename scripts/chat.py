#!/usr/bin/env python3
"""
Interactive CLI Chat with Browser Tools (Harmony Native Format)
================================================================

Chat with a model served by Elastic Serving.  The model uses the native
Harmony ``browser`` namespace (``browser.search``, ``browser.open``,
``browser.find``) backed by Serper (search) and Jina (URL reader).

Reasoning (analysis channel) is shown dimmed; the final answer is
highlighted.

Usage:
    python scripts/chat.py --scheduler-url http://localhost:8780
    python scripts/chat.py --scheduler-url http://localhost:8780 --verbose
    python scripts/chat.py --max-tool-calls 10 --temperature 0.9

Environment:
    SERPER_API_KEY       — for web search (Serper)
    JINA_API_KEY         — for URL reading (Jina Reader)
    ELASTIC_SERVING_URL  — default scheduler URL
"""

import argparse
import asyncio
import json
import os
import sys
import time

import dotenv
import httpx

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv()

from elastic_serving.tools import (
    DEFAULT_MAX_TOOL_CALLS,
    STOP_TOKENS,
    STOP_TOKENS_NO_CALL,
    SYSTEM_PROMPT,
    BrowserSession,
    append_tool_round,
    append_user_turn,
    build_initial_prompt,
    execute_custom_tool,
    extract_final_answer,
    parse_tool_call,
)

# =============================================================================
# ANSI colors
# =============================================================================


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    GRAY = "\033[90m"


def cprint(text: str, color: str = "", end: str = "\n"):
    print(f"{color}{text}{C.RESET}", end=end)


def print_tool_call(namespace: str, name: str, args: dict):
    short = json.dumps(args, ensure_ascii=False)
    if len(short) > 120:
        short = short[:117] + "..."
    cprint(f"  🔧 {namespace}.{name}({short})", C.YELLOW)


def print_tool_result(result: str, max_lines: int = 8):
    lines = result.split("\n")
    preview = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        preview += f"\n  ... ({len(lines) - max_lines} more lines)"
    for line in preview.split("\n"):
        cprint(f"  │ {line}", C.GRAY)


def print_reasoning(text: str):
    if not text.strip():
        return
    cprint("  💭 Reasoning:", C.DIM + C.ITALIC)
    for line in text.strip().split("\n"):
        cprint(f"  │ {line}", C.DIM)
    print()


def print_answer(text: str):
    cprint("  📝 Answer:", C.BOLD + C.GREEN)
    print()
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print()


# =============================================================================
# Chat engine
# =============================================================================


async def chat_turn(
    *,
    prompt: str,
    base_url: str,
    model: str,
    browser: BrowserSession,
    http_client: httpx.AsyncClient,
    openai_http: httpx.AsyncClient,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
    verbose: bool = False,
) -> tuple:
    """
    Run one user turn with the tool-call loop.

    Returns ``(final_raw_text, updated_prompt)`` where *updated_prompt*
    includes all tool rounds so far (for multi-turn reuse).
    """
    tool_call_count = 0

    while True:
        at_limit = tool_call_count >= max_tool_calls
        stops = STOP_TOKENS_NO_CALL if at_limit else STOP_TOKENS

        if verbose:
            prompt_tokens = len(prompt) // 4  # rough estimate
            suffix = " (final — tool limit)" if at_limit else ""
            cprint(
                f"  ⏳ Round {tool_call_count + 1} "
                f"(~{prompt_tokens} prompt tokens){suffix}",
                C.GRAY,
            )

        # Generate via /v1/completions (with retry on 503)
        t0 = time.time()
        max_retries = 60
        for attempt in range(max_retries):
            try:
                resp = await openai_http.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "max_tokens": max_gen_tokens,
                        "temperature": temperature,
                        "stop": stops,
                    },
                    headers={"Authorization": "Bearer EMPTY"},
                    timeout=300,
                )
                if resp.status_code == 503:
                    if attempt == 0:
                        cprint(
                            "  ⏳ No ready workers yet, waiting...",
                            C.YELLOW,
                        )
                    if attempt > 0 and attempt % 5 == 0:
                        cprint(f"  ⏳ Still waiting... ({attempt * 5}s)", C.GRAY)
                    await asyncio.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["text"]
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    if attempt == 0:
                        cprint("  ⏳ No ready workers, waiting...", C.YELLOW)
                    await asyncio.sleep(5)
                    continue
                cprint(f"  ❌ Generation error: {e}", C.RED)
                return f"[Error: {e}]", prompt
            except Exception as e:
                cprint(f"  ❌ Generation error: {e}", C.RED)
                return f"[Error: {e}]", prompt
        else:
            msg = "Timed out waiting for workers."
            cprint(f"  ❌ {msg}", C.RED)
            return f"[{msg}]", prompt

        elapsed = time.time() - t0
        if verbose:
            gen_tok = data.get("usage", {}).get(
                "completion_tokens", len(raw_text) // 4
            )
            cprint(
                f"  ⏱  {gen_tok} tokens in {elapsed:.1f}s "
                f"({gen_tok / max(elapsed, 0.01):.0f} tok/s)",
                C.GRAY,
            )

        # Try to parse a browser tool call
        tool_call = parse_tool_call(raw_text) if not at_limit else None

        if tool_call:
            ns, tool_name, tool_args = tool_call
            tool_call_count += 1
            print_tool_call(ns, tool_name, tool_args)

            # Execute — browser.* or functions.*
            if ns == "browser":
                result = await browser.execute(tool_name, tool_args)
            else:
                result = await execute_custom_tool(
                    tool_name, tool_args, http_client
                )
            print_tool_result(result)

            # Extend raw prompt
            prompt = append_tool_round(
                prompt, raw_text, tool_name, result, namespace=ns
            )
            continue
        else:
            # Final answer
            reasoning, answer = extract_final_answer(raw_text)
            if reasoning:
                print_reasoning(reasoning)
            print_answer(answer)

            # The raw_text includes the model's final output (stopped at <|end|>)
            return answer, prompt + raw_text


# =============================================================================
# Interactive REPL
# =============================================================================


async def interactive_chat(
    scheduler_url: str,
    model: str,
    tokenizer,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
    verbose: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
):
    http_client = httpx.AsyncClient(timeout=60)
    openai_http = httpx.AsyncClient(timeout=300)
    browser = BrowserSession(http_client)

    base_url = scheduler_url.rstrip("/")

    # State: raw_prompt carries the full conversation so prefix-caching works
    raw_prompt: str = ""
    turn_count = 0

    # Header
    print()
    cprint("╔══════════════════════════════════════════════════╗", C.CYAN)
    cprint("║     Elastic Serving — Interactive Research Chat  ║", C.CYAN)
    cprint("╚══════════════════════════════════════════════════╝", C.CYAN)
    cprint(f"  Model:      {model}", C.GRAY)
    cprint(f"  Server:     {base_url}", C.GRAY)
    cprint(f"  Tools:      browser.search, browser.open, browser.find, paper_search", C.GRAY)
    cprint(f"  Max calls:  {max_tool_calls} per turn", C.GRAY)
    print()
    cprint("  Commands: /clear  — reset conversation", C.GRAY)
    cprint("            /system — show system prompt", C.GRAY)
    cprint("            /verbose — toggle verbose mode", C.GRAY)
    cprint("            /quit or Ctrl+C — exit", C.GRAY)
    print()

    while True:
        try:
            user_input = input(f"{C.BOLD}{C.BLUE}You ❯ {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cprint("Goodbye!", C.CYAN)
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            cprint("Goodbye!", C.CYAN)
            break
        if user_input.lower() == "/clear":
            raw_prompt = ""
            turn_count = 0
            browser = BrowserSession(http_client)
            cprint("  ✅ Conversation cleared.", C.GREEN)
            continue
        if user_input.lower() == "/system":
            cprint("  Current system prompt:", C.GRAY)
            for line in system_prompt.split("\n"):
                cprint(f"  │ {line}", C.GRAY)
            continue
        if user_input.lower() == "/verbose":
            verbose = not verbose
            cprint(f"  Verbose mode: {'ON' if verbose else 'OFF'}", C.GREEN)
            continue

        # Build prompt
        if turn_count == 0:
            # First turn: use apply_chat_template for correct Harmony framing
            raw_prompt = build_initial_prompt(
                tokenizer,
                user_message=user_input,
                system_prompt=system_prompt,
            )
        else:
            # Subsequent turns: extend the raw prompt directly
            raw_prompt = append_user_turn(raw_prompt, "", user_input)

        turn_count += 1

        print()
        cprint(f"{'─' * 60}", C.GRAY)

        t0 = time.time()
        answer, raw_prompt = await chat_turn(
            prompt=raw_prompt,
            base_url=base_url,
            model=model,
            browser=browser,
            http_client=http_client,
            openai_http=openai_http,
            max_tool_calls=max_tool_calls,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            verbose=verbose,
        )
        elapsed = time.time() - t0

        cprint(f"{'─' * 60}", C.GRAY)
        cprint(f"  ⏱  Total turn: {elapsed:.1f}s", C.GRAY)
        print()

    await http_client.aclose()
    await openai_http.aclose()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with Harmony browser tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/chat.py --scheduler-url http://localhost:8780
  python scripts/chat.py --scheduler-url http://localhost:8780 --verbose
  python scripts/chat.py --max-tool-calls 10 --temperature 0.9

Environment Variables:
  ELASTIC_SERVING_URL  Default scheduler URL
  SERPER_API_KEY       Serper API key for web search
  JINA_API_KEY         Jina API key for URL reading
""",
    )
    parser.add_argument(
        "--scheduler-url",
        type=str,
        default=os.environ.get("ELASTIC_SERVING_URL", "http://localhost:8780"),
        help="Elastic Serving scheduler URL",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name/path")
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_MAX_TOOL_CALLS,
        help=f"Max browser tool calls per user turn (default: {DEFAULT_MAX_TOOL_CALLS})",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=8192,
        help="Max tokens per generation (default: 8192)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show token counts and timing per round",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system prompt (string or path to a .txt file)",
    )
    args = parser.parse_args()

    # Auto-detect model
    if not args.model:
        try:
            resp = httpx.get(
                f"{args.scheduler_url.rstrip('/')}/cluster_status", timeout=5
            )
            args.model = resp.json().get("model", "")
            if not args.model:
                print("Error: Could not detect model. Use --model.")
                sys.exit(1)
            print(f"Auto-detected model: {args.model}")
        except Exception as e:
            print(f"Error connecting to scheduler: {e}")
            sys.exit(1)

    # Health check
    try:
        resp = httpx.get(f"{args.scheduler_url.rstrip('/')}/health", timeout=5)
        ready = resp.json().get("ready_workers", 0)
        if ready == 0:
            print("Warning: No ready workers yet. Chat will wait for them.")
        else:
            print(f"Scheduler healthy: {ready} worker(s) ready.")
    except Exception:
        print(f"Warning: Cannot reach scheduler at {args.scheduler_url}")

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Tokenizer loaded.")

    # System prompt
    system_prompt = SYSTEM_PROMPT
    if args.system_prompt:
        if os.path.isfile(args.system_prompt):
            with open(args.system_prompt) as f:
                system_prompt = f.read()
        else:
            system_prompt = args.system_prompt

    asyncio.run(
        interactive_chat(
            scheduler_url=args.scheduler_url,
            model=args.model,
            tokenizer=tokenizer,
            max_tool_calls=args.max_tool_calls,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            verbose=args.verbose,
            system_prompt=system_prompt,
        )
    )


if __name__ == "__main__":
    main()
