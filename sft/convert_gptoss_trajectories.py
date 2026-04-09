#!/usr/bin/env python3
"""
Convert GPT-oss BrowseComp trajectories (Harmony format) to LLaMA-Factory
multi-turn SFT format for Qwen3.5 and Qwen3 models.

Source: elastic-serving eval results (results/browsecomp/trajectories.jsonl)
  - Harmony format assistant turns (<|channel|>...<|message|>...)
  - Tool outputs from BrowserSession (with cursors, line numbers, 【】 markers)

Target: LLaMA-Factory sharegpt multi-turn format
  - function_call turns with JSON tool calls in <tool_call> tags
  - observation turns with raw tool outputs (preserved as-is)
  - System prompt imported from elastic-serving

The tool outputs are kept exactly as elastic-serving produces them,
so training data perfectly matches inference.

Usage::

    python sft/convert_gptoss_trajectories.py \\
        --input results/browsecomp/trajectories.jsonl \\
        --results results/browsecomp/results.json \\
        --correct-only
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.dr_utils.prompts import SYSTEM_PROMPT
from elastic_serving.adapters import _QWEN_BROWSER_TOOLS, _QWEN_PAPER_TOOLS

TOOL_DEFINITIONS = list(_QWEN_BROWSER_TOOLS) + list(_QWEN_PAPER_TOOLS)
TOOL_DEF_MAP = {t["function"]["name"]: t for t in TOOL_DEFINITIONS}

# Harmony tool names -> Qwen tool names
HARMONY_TO_QWEN = {
    "browser.search": "web_search",
    "browser.open": "open_url",
    "browser.find": "find_text",
    "functions.paper_search": "paper_search",
    "functions.pubmed_search": "pubmed_search",
    "python.execute": "python",
}


def extract_thinking_from_harmony(content: str) -> str:
    """Extract the thinking/analysis text from Harmony format assistant content.

    Harmony format: <|channel|>analysis<|message|>thinking text...<|end|>...
    We convert to: <think>\nthinking text\n</think>
    """
    analysis_match = re.search(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)",
        content, re.DOTALL,
    )
    if analysis_match:
        return analysis_match.group(1).strip()

    content = re.sub(r"<\|[^|]+\|>", "", content).strip()
    content = re.sub(r"\b(analysis|commentary|assistant)\b", "", content).strip()
    return content


def extract_final_answer_from_harmony(content: str, final_answer: str = "") -> str:
    """Extract final answer from Harmony format, combining thinking + answer."""
    thinking = extract_thinking_from_harmony(content)

    if final_answer:
        answer = final_answer
    elif "<|channel|>final<|message|>" in content:
        parts = content.split("<|channel|>final<|message|>", 1)
        answer = re.sub(r"<\|[^|]+\|>", "", parts[1]).strip()
    else:
        answer = re.sub(r"<\|[^|]+\|>", "", content).strip()
        answer = re.sub(r"\b(analysis|commentary|assistant)\b", "", answer).strip()

    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{answer}"
    return answer


def make_function_call_json(tool_name: str, arguments: Dict[str, Any], thinking: str = "") -> str:
    """Build function_call content as JSON in <tool_call> tags.

    LLaMA-Factory's FunctionFormatter parses this and converts to the
    model-specific format (XML for Qwen3.5, JSON for Qwen3).
    """
    call_json = [{"name": tool_name, "arguments": arguments}]
    json_str = json.dumps(call_json, ensure_ascii=False)
    block = f"<tool_call>\n{json_str}\n</tool_call>"
    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{block}"
    return block


def convert_trajectory(traj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one GPT-oss Harmony trajectory to sharegpt multi-turn format."""
    conv = traj.get("conversation", [])
    if not conv:
        return None

    question = traj.get("question", "")
    new_convs: List[Dict[str, str]] = []
    new_convs.append({"from": "system", "value": SYSTEM_PROMPT})
    new_convs.append({"from": "human", "value": question})

    used_tools: set = set()

    for m in conv:
        role = m["role"]

        if role == "assistant":
            tc = m.get("tool_call", {})
            content = str(m.get("content", ""))
            final_answer = m.get("final_answer", "")
            thinking = extract_thinking_from_harmony(content)

            if tc and tc.get("tool"):
                harmony_tool = tc["tool"]
                qwen_name = HARMONY_TO_QWEN.get(harmony_tool, harmony_tool)
                args = tc.get("args", {})
                used_tools.add(qwen_name)
                fc = make_function_call_json(qwen_name, args, thinking)
                new_convs.append({"from": "function_call", "value": fc})
            else:
                answer_text = extract_final_answer_from_harmony(content, final_answer)
                new_convs.append({"from": "gpt", "value": answer_text})

        elif role == "tool":
            tool_output = str(m.get("content", ""))
            new_convs.append({"from": "observation", "value": tool_output})

    # Ensure ends with response role
    non_system = [c for c in new_convs if c["from"] != "system"]
    if not non_system or len(non_system) < 2:
        return None
    if non_system[-1]["from"] in ("observation", "human"):
        new_convs.append({"from": "gpt", "value": ""})
        non_system = [c for c in new_convs if c["from"] != "system"]

    # Validate alternation
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
        "id": f"{traj.get('qid', '')}_{traj.get('traj_idx', 0)}",
        "conversations": new_convs,
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-oss BrowseComp trajectories to LLaMA-Factory format"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Trajectory JSONL from eval results")
    parser.add_argument("--results", type=str, default=None,
                        help="results.json for filtering correct trajectories")
    parser.add_argument("--correct-only", action="store_true",
                        help="Only include trajectories judged correct")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=100,
                        help="Skip trajectories with more tool calls than this")
    args = parser.parse_args()

    # Load trajectories
    trajs = []
    with open(args.input) as f:
        for line in f:
            trajs.append(json.loads(line))
    print(f"Loaded {len(trajs)} trajectories")

    # Filter correct only
    if args.correct_only and args.results:
        with open(args.results) as f:
            res = json.load(f)
        correct_keys = set()
        for q in res.get("per_question", []):
            for a in q.get("answers", []):
                if a.get("correct"):
                    correct_keys.add((q["qid"], a.get("traj_idx", 0)))
        trajs = [t for t in trajs if (t["qid"], t.get("traj_idx", 0)) in correct_keys]
        print(f"After correct-only filter: {len(trajs)}")

    # Filter by conversation and tool calls
    trajs = [
        t for t in trajs
        if t.get("conversation")
        and t.get("num_tool_calls", 0) <= args.max_tool_calls
        and t.get("status") == "success"
    ]
    print(f"After filtering (has conv, ≤{args.max_tool_calls} tools, success): {len(trajs)}")

    # Convert
    converted = []
    role_counter: Counter = Counter()
    tool_counter: Counter = Counter()
    n_skipped = 0

    for t in trajs:
        result = convert_trajectory(t)
        if result is None:
            n_skipped += 1
            continue
        for msg in result["conversations"]:
            role_counter[msg["from"]] += 1
        if result["tools"]:
            for td in json.loads(result["tools"]):
                tool_counter[td["function"]["name"]] += 1
        converted.append(result)

    print(f"\nConverted {len(converted)} examples ({n_skipped} skipped)")
    print("\nRole distribution:")
    for role, cnt in role_counter.most_common():
        print(f"  {role:20s}  {cnt}")
    print("\nTool usage:")
    for tool, cnt in tool_counter.most_common():
        print(f"  {tool:20s}  {cnt}")

    # Preview
    if converted:
        ex = converted[0]
        print(f"\n{'='*60}")
        print(f"Sample (first example):")
        for j, msg in enumerate(ex["conversations"][:8]):
            v = msg["value"]
            preview = v[:200] + f"... [{len(v)} chars]" if len(v) > 200 else v
            print(f"  [{j}] {msg['from']:20s} | {preview}")

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, "browsecomp-gptoss-multiturn.json")

    with open(output_path, "w") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"Total: {len(converted)}")


if __name__ == "__main__":
    main()
