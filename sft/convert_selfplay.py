#!/usr/bin/env python3
"""
Convert the base model's own correct BrowseComp trajectories to LLaMA-Factory
SFT training format (self-play / on-policy SFT).

The base model's trajectories already use Qwen3.5 native format, so
conversion is straightforward — no tool name mapping or format conversion needed.

Usage:
    python sft/convert_selfplay.py \
        --input results/bc_qwen35_9b_base_v2/trajectories.jsonl \
        --results results/bc_qwen35_9b_base_v2/results.json \
        --output sft/data/browsecomp-selfplay.json
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elastic_serving.dr_utils.prompts import SYSTEM_PROMPT
from elastic_serving.adapters import _QWEN_BROWSER_TOOLS, _QWEN_PAPER_TOOLS

TOOL_DEFINITIONS = list(_QWEN_BROWSER_TOOLS) + list(_QWEN_PAPER_TOOLS)


def convert_trajectory(traj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert one base-model trajectory to sharegpt multi-turn format.

    The conversation is already in Qwen3.5 format:
      - assistant turns contain <think>...</think> and <tool_call><function=...> blocks
      - tool turns contain BrowserSession output
    """
    conv = traj.get("conversation", [])
    if not conv:
        return None

    question = traj.get("question", "")
    new_convs: List[Dict[str, str]] = []
    new_convs.append({"from": "system", "value": SYSTEM_PROMPT})
    new_convs.append({"from": "human", "value": question})

    for m in conv:
        role = m["role"]
        content = str(m.get("content", ""))

        if role == "assistant":
            if "<tool_call>" in content:
                # Tool call turn. The adapter prepends <think>\n before generation,
                # so the stored content may have </think> without the opening <think>.
                # The actual sequence is: <think>\n{reasoning}\n</think>\n\n<tool_call>...

                # Extract XML tool call
                tc_match = re.search(
                    r'<tool_call>\s*<function=(\w+)>(.*?)</function>',
                    content, re.DOTALL
                )
                if tc_match:
                    func_name = tc_match.group(1)
                    params_block = tc_match.group(2)
                    args = {}
                    for pm in re.finditer(
                        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
                        params_block, re.DOTALL
                    ):
                        key = pm.group(1)
                        value = pm.group(2).strip()
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            pass
                        args[key] = value

                    call_json = [{"name": func_name, "arguments": args}]
                    json_str = json.dumps(call_json, ensure_ascii=False)

                    # Extract thinking: handle both <think>...</think> and
                    # missing opening <think> (adapter prepends it at inference)
                    thinking = ""
                    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                    elif '</think>' in content:
                        thinking = content[:content.index('</think>')].strip()

                    block = f"<tool_call>\n{json_str}\n</tool_call>"
                    if thinking:
                        fc_value = f"<think>\n{thinking}\n</think>\n\n{block}"
                    else:
                        fc_value = block
                    new_convs.append({"from": "function_call", "value": fc_value})
                else:
                    continue
            else:
                # Final answer turn — same fix for missing <think> opening
                if '</think>' in content and '<think>' not in content:
                    content = '<think>\n' + content
                new_convs.append({"from": "gpt", "value": content})

        elif role == "tool":
            new_convs.append({"from": "observation", "value": content})

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

    return {
        "id": f"{traj.get('qid', '')}_{traj.get('traj_idx', 0)}",
        "conversations": new_convs,
        "tools": json.dumps(TOOL_DEFINITIONS, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    args = parser.parse_args()

    # Load results for correct filtering
    with open(args.results) as f:
        res = json.load(f)
    correct_qids = set()
    for q in res.get("per_question", []):
        if any(a.get("correct") for a in q.get("answers", [])):
            correct_qids.add(q["qid"])
    print(f"Correct questions: {len(correct_qids)}")

    # Load and filter trajectories
    trajs = []
    with open(args.input) as f:
        for line in f:
            t = json.loads(line)
            if t["qid"] in correct_qids:
                if t.get("conversation") and t.get("num_tool_calls", 0) <= args.max_tool_calls:
                    trajs.append(t)
    print(f"Correct trajectories with conv, ≤{args.max_tool_calls} TC: {len(trajs)}")

    # Convert
    converted = []
    skipped = 0
    for t in trajs:
        result = convert_trajectory(t)
        if result is None:
            skipped += 1
        else:
            converted.append(result)

    print(f"Converted: {len(converted)}, Skipped: {skipped}")

    # Stats
    tc_counts = [sum(1 for c in ex["conversations"] if c["from"] == "function_call") for ex in converted]
    if tc_counts:
        print(f"Tool calls: min={min(tc_counts)}, avg={sum(tc_counts)/len(tc_counts):.1f}, max={max(tc_counts)}")

    # Save
    output_path = args.output or os.path.join(os.path.dirname(__file__), "data", "browsecomp-selfplay.json")
    with open(output_path, "w") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

    # Show sample
    if converted:
        ex = converted[0]
        print(f"\nSample (first 4 turns):")
        for i, c in enumerate(ex["conversations"][:6]):
            v = c["value"][:150]
            print(f"  [{i}] {c['from']:15s} | {v}")


if __name__ == "__main__":
    main()
