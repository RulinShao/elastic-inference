#!/usr/bin/env python3
"""
Convert DR Tulu v1 SFT data to Qwen3.5 multi-turn tool-calling format.

Source: ``rl-rag/dr-tulu-sft-unified`` (has full tool calls + tool outputs)
Target: LLaMA-Factory sharegpt multi-turn format with Qwen3.5 tool protocols

Tool name mapping:
    serper_google_webpage_search   → web_search
    serper_fetch_webpage_content   → open_url
    semantic_scholar_snippet_search → paper_search
    search_papers_by_relevance     → paper_search

Roles:
    system, human, observation     → masked (not trained on)
    function_call, gpt             → trained on

Citations: ``<cite id="S_xxx">text</cite>`` tags are preserved as-is.
Snippet IDs from tool outputs are kept in the formatted responses so
cite references remain valid.

Usage::

    # Download + convert (default)
    python sft/convert_drtulu.py

    # From local file
    python sft/convert_drtulu.py --input sft/data/dr-tulu-sft-source.jsonl

    # Filter by type
    python sft/convert_drtulu.py --types long_form,short_form
"""

import argparse
import ast
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ── Qwen3.5 system prompt (matching elastic-serving) ─────────────────────────

QWEN35_SYSTEM_PROMPT = """\
You are a research assistant that answers questions by searching the web \
and reading sources. You have access to browser tools and an academic paper \
search tool (paper_search via Semantic Scholar) and a biomedical \
literature search tool (pubmed_search via PubMed/NCBI).

Support every non-trivial claim with evidence from your searches. Cite \
information by wrapping the exact claim span in <cite id="ID1,ID2">...</cite>, \
where id are snippet IDs from searched results (comma-separated if multiple \
sources support the same claim).

For short factual answers, also include the answer as \\boxed{answer}. \
Acknowledge uncertainty when evidence is thin or conflicting."""

# ── Tool definitions (matching elastic-serving Qwen3Adapter) ─────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. "
                "Returns titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "topn": {
                        "type": "integer",
                        "description": "Number of results (default: 10).",
                    },
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
                "Open and read a webpage by search result ID or full URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Search result number or full URL.",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paper_search",
            "description": (
                "Search Semantic Scholar for academic papers. "
                "Returns titles, authors, year, venue, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5).",
                    },
                    "year": {
                        "type": "string",
                        "description": "Year filter (e.g. '2024').",
                    },
                    "fields_of_study": {
                        "type": "string",
                        "description": "E.g. 'Computer Science'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_DEF_MAP = {t["function"]["name"]: t for t in TOOL_DEFINITIONS}

# ── Tool name + argument mapping ─────────────────────────────────────────────

TOOL_NAME_MAP = {
    "serper_google_webpage_search": "web_search",
    "serper_fetch_webpage_content": "open_url",
    "semantic_scholar_snippet_search": "paper_search",
    "search_papers_by_relevance": "paper_search",
}

# ── Parsing DR Tulu function_calls format ────────────────────────────────────


def parse_drtulu_function_call(
    fc_str: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse DR Tulu's ``tool_name(key=val, ...)`` format.

    Returns ``(tool_name, arguments_dict)`` or ``None``.
    """
    fc_str = fc_str.strip()
    m = re.match(r"(\w+)\((.*)\)$", fc_str, re.DOTALL)
    if not m:
        return None

    raw_name = m.group(1)
    args_str = m.group(2).strip()

    # Parse arguments via ast.literal_eval wrapped in a dict()
    args: Dict[str, Any] = {}
    if args_str:
        try:
            args = ast.literal_eval(f"dict({args_str})")
        except Exception:
            # Fallback: regex-based extraction of key='value' pairs
            for kv in re.finditer(
                r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\d+))", args_str
            ):
                key = kv.group(1)
                val = kv.group(2) or kv.group(3) or kv.group(4)
                if val and val.isdigit():
                    val = int(val)
                args[key] = val

    return raw_name, args


def map_tool_call(
    raw_name: str, raw_args: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Map DR Tulu tool name + args to Qwen3.5/elastic-serving format."""
    qwen_name = TOOL_NAME_MAP.get(raw_name, raw_name)

    if qwen_name == "web_search":
        mapped = {"query": raw_args.get("query", "")}
        topn = raw_args.get("num_results")
        if topn:
            mapped["topn"] = int(topn)
        return qwen_name, mapped

    elif qwen_name == "open_url":
        # serper_fetch_webpage_content uses 'query' for the URL
        url = raw_args.get("query", raw_args.get("url", ""))
        return qwen_name, {"id": url}

    elif qwen_name == "paper_search":
        mapped = {"query": raw_args.get("query", "")}
        if raw_args.get("limit"):
            mapped["limit"] = int(raw_args["limit"])
        if raw_args.get("year"):
            mapped["year"] = raw_args["year"]
        if raw_args.get("fieldsOfStudy"):
            mapped["fields_of_study"] = raw_args["fieldsOfStudy"]
        return qwen_name, mapped

    return qwen_name, raw_args


# ── Formatting tool responses ────────────────────────────────────────────────


def extract_env_text(env_content: str) -> str:
    """Extract the inner JSON/text from DR Tulu environment content.

    Env content is typically a Python dict repr:
      ``{'type': 'text', 'text': '{"searchParameters": ...}'}``
    or occasionally raw text / XML snippets.
    """
    env_content = env_content.strip()
    if not env_content:
        return "(no output)"

    # Try parsing as Python dict literal
    if env_content.startswith("{") or env_content.startswith("{'"):
        try:
            parsed = ast.literal_eval(env_content)
            if isinstance(parsed, dict) and "text" in parsed:
                return parsed["text"]
        except Exception:
            pass

    # Try JSON
    if env_content.startswith('{"'):
        try:
            parsed = json.loads(env_content)
            if isinstance(parsed, dict) and "text" in parsed:
                return parsed["text"]
        except Exception:
            pass

    # XML snippet format or plain text
    return env_content


def format_search_response(raw_json_text: str) -> str:
    """Format web search results (from Serper API) into readable text.

    Preserves snippet_ids for cite tag references.
    """
    try:
        data = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        return raw_json_text

    query = data.get("searchParameters", {}).get("q", "")
    results = data.get("organic", [])
    if not results:
        return raw_json_text

    lines = [f'Searched for "{query}"', ""]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        link = r.get("link", "")
        snippet = r.get("snippet", "")
        sid = r.get("snippet_id", "")

        lines.append(f"[{i}] {title}")
        lines.append(f"    URL: {link}")
        if snippet:
            if sid:
                lines.append(f"    Snippet [{sid}]: {snippet}")
            else:
                lines.append(f"    Snippet: {snippet}")
        lines.append("")

    return "\n".join(lines)


def format_browse_response(raw_json_text: str) -> str:
    """Format browse/fetch results into readable text."""
    try:
        data = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        return raw_json_text

    text = data.get("text", "")
    sid = data.get("snippet_id", "")
    url = data.get("metadata", {}).get("og:url", "")

    parts = []
    if url:
        parts.append(f"Page: {url}")
    if sid:
        parts.append(f"[{sid}]")
    if text:
        parts.append(text)

    return "\n".join(parts) if parts else raw_json_text


def format_paper_response(raw_json_text: str) -> str:
    """Format Semantic Scholar snippet results into readable text."""
    try:
        data = json.loads(raw_json_text)
    except (json.JSONDecodeError, TypeError):
        return raw_json_text

    snippets = data.get("data", [])
    if not snippets:
        return raw_json_text

    lines = ["Paper search results:", ""]
    for i, s in enumerate(snippets, 1):
        sid = s.get("snippet_id", "")
        paper = s.get("paper", {})
        title = paper.get("title", s.get("title", ""))
        snippet_obj = s.get("snippet", {})
        if isinstance(snippet_obj, dict):
            snippet_text = snippet_obj.get("text", "")
        else:
            snippet_text = str(snippet_obj)

        lines.append(f"[{i}] {title}")
        if sid:
            lines.append(f"    ID: {sid}")
        if snippet_text:
            if len(snippet_text) > 800:
                snippet_text = snippet_text[:800] + "..."
            lines.append(f"    Snippet: {snippet_text}")
        lines.append("")

    return "\n".join(lines)


def format_tool_response(tool_name: str, env_content: str) -> str:
    """Format environment content according to the tool that produced it."""
    raw_text = extract_env_text(env_content)

    if not raw_text or raw_text in ("(no output)", ""):
        return "(no output)"

    if "Search Failure" in raw_text:
        return raw_text

    if tool_name == "web_search":
        return format_search_response(raw_text)
    elif tool_name == "open_url":
        return format_browse_response(raw_text)
    elif tool_name == "paper_search":
        return format_paper_response(raw_text)

    return raw_text


# ── Main conversion ──────────────────────────────────────────────────────────


def make_function_call_content(
    tool_name: str, arguments: Dict[str, Any], thinking: str = ""
) -> str:
    """Build function_call turn content for LLaMA-Factory."""
    call_json = [{"name": tool_name, "arguments": arguments}]
    json_str = json.dumps(call_json, ensure_ascii=False)
    tool_call_block = f"<tool_call>\n{json_str}\n</tool_call>"

    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{tool_call_block}"
    return tool_call_block


def convert_example(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one DR Tulu example to Qwen3.5 multi-turn sharegpt format."""
    msgs = row["messages"]
    question = row.get("question", "")

    new_convs: List[Dict[str, str]] = []
    new_convs.append({"from": "system", "value": QWEN35_SYSTEM_PROMPT})

    # Extract user question (handle multimodal list format)
    user_content = ""
    for m in msgs:
        if m["role"] == "user":
            user_content = m.get("content", "")
            if isinstance(user_content, list):
                user_content = " ".join(
                    c.get("text", "")
                    for c in user_content
                    if isinstance(c, dict)
                )
            elif user_content.startswith("[{"):
                try:
                    parts = ast.literal_eval(user_content)
                    user_content = " ".join(
                        p.get("text", "")
                        for p in parts
                        if isinstance(p, dict)
                    )
                except Exception:
                    pass
            break

    if not user_content:
        user_content = question
    new_convs.append({"from": "human", "value": user_content})

    used_tools: set = set()
    last_tool_name = ""

    # Process assistant/environment turns
    i = 0
    while i < len(msgs):
        m = msgs[i]

        if m["role"] in ("system", "user"):
            i += 1
            continue

        if m["role"] == "assistant":
            content = str(m.get("content", ""))
            fc = m.get("function_calls", "")

            # Extract thinking from content
            think_match = re.search(
                r"<think>(.*?)</think>", content, re.DOTALL
            )
            thinking = think_match.group(1).strip() if think_match else ""

            if fc and isinstance(fc, str) and fc.strip():
                # Assistant turn with tool call
                parsed = parse_drtulu_function_call(fc)
                if parsed:
                    raw_name, raw_args = parsed
                    qwen_name, qwen_args = map_tool_call(raw_name, raw_args)
                    used_tools.add(qwen_name)
                    last_tool_name = qwen_name

                    fc_content = make_function_call_content(
                        qwen_name, qwen_args, thinking
                    )
                    new_convs.append(
                        {"from": "function_call", "value": fc_content}
                    )
                else:
                    # Couldn't parse tool call — treat as thinking-only turn
                    # Skip this as it would break alternation
                    i += 1
                    continue
            else:
                # Final answer turn (no tool call)
                # Keep full content (think + answer) — strip only closing tags
                answer_content = content.strip()
                # Replace <answer>\boxed{X}</answer> with just the content
                answer_content = re.sub(
                    r"<answer>(.*?)</answer>",
                    r"\1",
                    answer_content,
                    flags=re.DOTALL,
                )
                new_convs.append({"from": "gpt", "value": answer_content})

        elif m["role"] == "environment":
            env_content = str(m.get("content", ""))
            formatted = format_tool_response(last_tool_name, env_content)
            new_convs.append({"from": "observation", "value": formatted})

        i += 1

    # Validate: ensure it ends with a response role
    non_system = [c for c in new_convs if c["from"] != "system"]
    if not non_system or len(non_system) < 2:
        return None

    if non_system[-1]["from"] in ("observation", "human"):
        new_convs.append({"from": "gpt", "value": ""})
        non_system = [c for c in new_convs if c["from"] != "system"]

    # Validate strict alternation
    prompt_roles = {"human", "observation"}
    response_roles = {"gpt", "function_call"}
    valid = len(non_system) % 2 == 0
    for idx, msg in enumerate(non_system):
        expected = prompt_roles if idx % 2 == 0 else response_roles
        if msg["from"] not in expected:
            valid = False
            break

    if not valid:
        return None

    tools = [TOOL_DEF_MAP[t] for t in used_tools if t in TOOL_DEF_MAP]

    return {
        "id": row.get("id", row.get("instance_id", "")),
        "conversations": new_convs,
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert DR Tulu v1 SFT data to Qwen3.5 format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Local JSONL path (default: download from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated type filter (e.g. 'long_form,short_form')",
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Minimum tool calls to include",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=16384,
        help="Max tokenized length to include",
    )
    args = parser.parse_args()

    # Load data
    input_path = args.input
    if not input_path:
        local_cache = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            "dr-tulu-sft-source.jsonl",
        )
        if os.path.exists(local_cache):
            input_path = local_cache
            print(f"Using cached: {local_cache}")
        else:
            print("Downloading from HuggingFace...")
            import httpx

            url = (
                "https://huggingface.co/datasets/rl-rag/"
                "dr-tulu-sft-unified/resolve/main/converted_drtulu.jsonl"
            )
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(url)
            os.makedirs(os.path.dirname(local_cache), exist_ok=True)
            with open(local_cache, "w") as f:
                f.write(resp.text)
            input_path = local_cache
            print(f"Saved to {local_cache}")

    print(f"Loading: {input_path}")
    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} examples")

    # Filter
    type_filter = set(args.types.split(",")) if args.types else None
    filtered = []
    for row in rows:
        if type_filter and row.get("type") not in type_filter:
            continue
        if row.get("num_tool_calls", 0) < args.min_tool_calls:
            continue
        if row.get("tokenized_length", 0) > args.max_token_length:
            continue
        filtered.append(row)

    print(
        f"After filtering: {len(filtered)} examples"
        + (f" (types: {type_filter})" if type_filter else "")
    )

    # Convert
    converted = []
    role_counter: Counter = Counter()
    n_skipped = 0
    tool_counter: Counter = Counter()

    for row in filtered:
        result = convert_example(row)
        if result is None:
            n_skipped += 1
            continue

        for msg in result["conversations"]:
            role_counter[msg["from"]] += 1
        if result["tools"]:
            for t in json.loads(result["tools"]):
                tool_counter[t["function"]["name"]] += 1
        converted.append(result)

    print(
        f"\nConverted {len(converted)} examples "
        f"({n_skipped} skipped due to invalid structure)"
    )
    print("\nRole distribution:")
    for role, cnt in role_counter.most_common():
        print(f"  {role:20s}  {cnt}")
    print("\nTool usage:")
    for tool, cnt in tool_counter.most_common():
        print(f"  {tool:20s}  {cnt}")

    # Preview
    if converted:
        print("\n" + "=" * 80)
        print("Sample (first example):")
        print("=" * 80)
        ex = converted[0]
        for j, msg in enumerate(ex["conversations"][:12]):
            preview = msg["value"][:200]
            if len(msg["value"]) > 200:
                preview += f"... [{len(msg['value'])} chars]"
            print(f"  [{j}] from={msg['from']:20s} | {preview}")
        n_turns = len(ex["conversations"])
        if n_turns > 12:
            print(f"  ... ({n_turns} turns total)")

    # Save
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        output_dir, "drtulu-qwen35-multiturn.json"
    )

    with open(output_path, "w") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Total examples: {len(converted)}")


if __name__ == "__main__":
    main()
