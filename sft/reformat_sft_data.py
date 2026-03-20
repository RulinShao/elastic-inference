#!/usr/bin/env python3
"""
Reformat elastic-serving SFT trajectories to LLaMA-Factory multi-turn format.

Input:
  JSONL trajectories from ``generate_trajectories.py`` — each line contains
  ``turns`` with raw Qwen3-format assistant outputs (``<think>`` reasoning
  + ``<tool_call>`` XML blocks) and tool responses.

Output:
  JSON in LLaMA-Factory's sharegpt multi-turn format::

    system / human / function_call / observation / ... / gpt

  Roles and training masks (automatic via LLaMA-Factory multi-turn format):
    system, human, observation → masked (not trained on)
    function_call, gpt         → **trained on**

Usage::

    # From local trajectory JSONL
    python sft/reformat_sft_data.py \\
        --input sft/trajectories/trajectories.jsonl

    # From HuggingFace dataset
    python sft/reformat_sft_data.py \\
        --hf-dataset my-org/my-dr-trajectories --hf-split train
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# ── Tool definitions (matching elastic-serving/Qwen3Adapter tools) ───────────

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
                    "cursor": {
                        "type": "integer",
                        "description": "Search cursor to look up id from.",
                    },
                    "loc": {
                        "type": "integer",
                        "description": "Line number to scroll to.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_text",
            "description": (
                "Find exact matches of a pattern in the current page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for.",
                    },
                    "cursor": {
                        "type": "integer",
                        "description": "Page cursor.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute Python code in a stateful Jupyter environment. "
                "Variables persist across calls. Use print() to see output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    },
                },
                "required": ["code"],
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
                    "mode": {
                        "type": "string",
                        "description": "'snippets' (default) or 'papers'.",
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
                    "venue": {
                        "type": "string",
                        "description": "E.g. 'ACL', 'NeurIPS'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": (
                "Search PubMed for biomedical literature. "
                "Returns titles, authors, journal, abstract."
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
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_DEF_MAP = {t["function"]["name"]: t for t in TOOL_DEFINITIONS}

# ── Regex patterns for parsing Qwen3 XML tool calls ─────────────────────────

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_XML_PATTERN = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*(?:</tool_call>)?",
    re.DOTALL,
)
PARAM_PATTERN = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def parse_qwen3_tool_call(
    content: str,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Parse Qwen3 XML-style tool call from raw assistant content.

    Returns ``(think_text, tool_name, arguments)`` or ``None``.
    """
    tc_match = TOOL_CALL_XML_PATTERN.search(content)
    if not tc_match:
        return None

    func_name = tc_match.group(1)
    params_block = tc_match.group(2)

    args: Dict[str, Any] = {}
    for pm in PARAM_PATTERN.finditer(params_block):
        key = pm.group(1)
        value = pm.group(2).strip()
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        args[key] = value

    think_match = THINK_PATTERN.search(content[: tc_match.start()])
    think_text = think_match.group(1).strip() if think_match else ""

    return think_text, func_name, args


def make_function_call_content(
    tool_name: str, arguments: Dict[str, Any], thinking: str = ""
) -> str:
    """Build ``function_call`` turn content with JSON tool call.

    LLaMA-Factory's Qwen3 template expects JSON inside ``<tool_call>`` tags.
    """
    call_json = [{"name": tool_name, "arguments": arguments}]
    json_str = json.dumps(call_json, ensure_ascii=False)
    tool_call_block = f"<tool_call>\n{json_str}\n</tool_call>"

    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{tool_call_block}"
    return tool_call_block


def reformat_trajectory(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one trajectory from generate_trajectories.py output to
    LLaMA-Factory's multi-turn sharegpt format.
    """
    turns = example.get("turns", [])
    if not turns:
        return None

    system_prompt = example.get("system_prompt", "")
    question = example.get("question", "")

    new_convs: List[Dict[str, str]] = []
    if system_prompt:
        new_convs.append({"from": "system", "value": system_prompt})
    new_convs.append({"from": "human", "value": question})

    used_tools: set = set()

    for turn in turns:
        role = turn["role"]

        if role == "assistant":
            parsed = parse_qwen3_tool_call(turn["content"])
            if parsed:
                think_text, tool_name, tool_args = parsed
                used_tools.add(tool_name)
                fc = make_function_call_content(
                    tool_name, tool_args, think_text
                )
                new_convs.append({"from": "function_call", "value": fc})
            else:
                content = turn["content"]
                content = re.sub(
                    r"<\|im_end\|>.*$", "", content, flags=re.DOTALL
                ).strip()
                new_convs.append({"from": "gpt", "value": content})

        elif role == "tool":
            new_convs.append(
                {"from": "observation", "value": turn["content"]}
            )

    # Ensure the conversation ends with a response role
    non_system = [m for m in new_convs if m["from"] != "system"]
    if non_system and non_system[-1]["from"] in ("observation", "human"):
        new_convs.append({"from": "gpt", "value": ""})
        non_system = [m for m in new_convs if m["from"] != "system"]

    # Validate strict prompt/response alternation
    prompt_roles = {"human", "observation"}
    response_roles = {"gpt", "function_call"}
    valid = len(non_system) % 2 == 0
    for i, msg in enumerate(non_system):
        expected = prompt_roles if i % 2 == 0 else response_roles
        if msg["from"] not in expected:
            valid = False
            break

    if not valid:
        return None

    tools = [TOOL_DEF_MAP[t] for t in used_tools if t in TOOL_DEF_MAP]

    return {
        "id": example.get("id", ""),
        "conversations": new_convs,
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reformat elastic-serving trajectories to "
            "LLaMA-Factory multi-turn format"
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        default="sft/trajectories/trajectories.jsonl",
        help="Input JSONL from generate_trajectories.py",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HuggingFace dataset ID (alternative to --input)",
    )
    parser.add_argument(
        "--hf-split", type=str, default="train", help="HuggingFace split"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: sft/data/dr-sft-trajectories-multiturn.json)",
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Minimum tool calls to include a trajectory",
    )
    args = parser.parse_args()

    # Load data
    if args.hf_dataset:
        from datasets import load_dataset

        print(f"Loading HF dataset: {args.hf_dataset}")
        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        examples = list(ds)
    else:
        print(f"Loading: {args.input}")
        examples = []
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

    print(f"Loaded {len(examples)} trajectories")

    # Filter to successful trajectories with enough tool calls
    examples = [
        ex
        for ex in examples
        if ex.get("status") == "success"
        and ex.get("num_tool_calls", 0) >= args.min_tool_calls
    ]
    print(f"After filtering: {len(examples)} trajectories")

    # Reformat
    reformatted = []
    role_counter: Counter = Counter()
    n_skipped = 0

    for ex in examples:
        result = reformat_trajectory(ex)
        if result is None:
            n_skipped += 1
            continue

        for msg in result["conversations"]:
            role_counter[msg["from"]] += 1
        reformatted.append(result)

    print(
        f"\nReformatted {len(reformatted)} examples "
        f"({n_skipped} skipped due to invalid structure)"
    )
    print("\nRole distribution:")
    for role, cnt in role_counter.most_common():
        print(f"  {role:20s}  {cnt}")

    # Preview
    if reformatted:
        print("\n" + "=" * 80)
        print("Sample (first example):")
        print("=" * 80)
        ex = reformatted[0]
        for j, msg in enumerate(ex["conversations"][:12]):
            preview = msg["value"][:200]
            if len(msg["value"]) > 200:
                preview += f"... [{len(msg['value'])} chars]"
            print(f"  [{j}] from={msg['from']:20s} | {preview}")
        n_turns = len(ex["conversations"])
        if n_turns > 12:
            print(f"  ... ({n_turns} turns total)")
        if ex["tools"]:
            print(f"  tools: {ex['tools'][:200]}...")

    # Save
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        output_dir, "dr-sft-trajectories-multiturn.json"
    )

    with open(output_path, "w") as f:
        json.dump(reformatted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Total examples: {len(reformatted)}")


if __name__ == "__main__":
    main()
