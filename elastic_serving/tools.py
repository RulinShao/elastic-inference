"""
Orchestration layer for Elastic Serving tool-use agents (Harmony format).

This module provides:
  - Harmony format parsing (``parse_tool_call``, ``extract_final_answer``)
  - Prompt building (``build_initial_prompt``, ``append_tool_round``, ``append_user_turn``)
  - Configuration constants (``STOP_TOKENS``, ``BUILTIN_TOOLS``, ``DEFAULT_MAX_TOOL_CALLS``)

Tool implementations and prompts live in ``elastic_serving.deep_research_utils``:
  - ``deep_research_tools.py`` — BrowserSession, snippet_search, tool specs
  - ``prompts.py``             — SYSTEM_PROMPT, MODEL_IDENTITY

This module re-exports the most commonly used symbols so that callers can
do ``from elastic_serving.tools import ...`` without knowing the subpackage.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# Re-export tool implementations and prompts from deep_research_utils
from elastic_serving.deep_research_utils import (  # noqa: F401
    CUSTOM_TOOLS,
    LEGACY_SYSTEM_PROMPT,
    LEGACY_TOOLS,
    MODEL_IDENTITY,
    SNIPPET_SEARCH_TOOL,
    SYSTEM_PROMPT,
    BrowserSession,
    execute_custom_tool,
    execute_legacy_tool,
    snippet_search,
)

# Backward-compat alias
TOOLS = LEGACY_TOOLS
execute_tool = execute_legacy_tool

# =============================================================================
# Configuration
# =============================================================================

BUILTIN_TOOLS: List[str] = ["browser"]
"""Passed to ``apply_chat_template(builtin_tools=...)``."""

DEFAULT_MAX_TOOL_CALLS = 15
"""Default cap on tool calls per user turn."""

STOP_TOKENS = ["<|call|>", "<|end|>", "<|endoftext|>"]
"""vLLM stop strings for Harmony generation."""

STOP_TOKENS_NO_CALL = ["<|end|>", "<|endoftext|>"]
"""Stop strings that force a final answer (no more tool calls)."""


# =============================================================================
# Prompt helpers — build & extend raw Harmony prompts
# =============================================================================


def build_initial_prompt(
    tokenizer,
    user_message: str,
    system_prompt: str = SYSTEM_PROMPT,
    model_identity: str = MODEL_IDENTITY,
    custom_tools: Optional[List[dict]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build the initial prompt using ``apply_chat_template``.

    Uses ``builtin_tools=["browser"]`` so the model sees its native browser
    namespace.  ``system_prompt`` goes into the developer message.
    Custom tools (e.g. snippet_search) go into the ``functions`` namespace.
    """
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    kwargs = {
        "builtin_tools": BUILTIN_TOOLS,
        "model_identity": model_identity,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    tools = custom_tools if custom_tools is not None else CUSTOM_TOOLS
    if tools:
        kwargs["tools"] = tools

    return tokenizer.apply_chat_template(messages, **kwargs)


def append_tool_round(
    prompt: str,
    model_output: str,
    tool_name: str,
    tool_response: str,
    namespace: str = "browser",
) -> str:
    """Extend the raw prompt with a tool call + response.

    ``model_output`` is the raw text from vLLM (does NOT include the
    ``<|call|>`` stop token).  ``namespace`` is ``"browser"`` for built-in
    tools or ``"functions"`` for custom tools.
    """
    return (
        f"{prompt}{model_output}"
        f"<|call|>"
        f"<|start|>{namespace}.{tool_name} to=assistant"
        f"<|channel|>commentary<|message|>"
        f"{json.dumps(tool_response)}"
        f"<|end|>"
        f"<|start|>assistant"
    )


def append_user_turn(prompt: str, final_answer_text: str, user_message: str) -> str:
    """Extend the raw prompt with the model's final answer + a new user turn."""
    return (
        f"{prompt}{final_answer_text}"
        f"<|end|>"
        f"<|start|>user<|message|>{user_message}<|end|>"
        f"<|start|>assistant"
    )


# =============================================================================
# Harmony format parsing
# =============================================================================


def parse_tool_call(text: str) -> Optional[Tuple[str, str, dict]]:
    """Parse a tool call from raw model output.

    Returns ``(namespace, tool_name, args_dict)`` or ``None``.
    Handles both ``to=browser.search`` and ``to=functions.snippet_search``.
    """
    m = re.search(r"to=(browser|functions)\.(\w+)", text)
    if not m:
        return None

    namespace = m.group(1)
    tool_name = m.group(2)

    msg_match = re.search(
        r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", text, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        after = text[m.end():]
        json_match = re.search(r"(?:json|code)\s*(\{.*?\})\s*$", after, re.DOTALL)
        args_str = json_match.group(1) if json_match else after.strip()

    # Unescape if JSON-string-wrapped
    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            args_str = json.loads(args_str)
        except Exception:
            args_str = args_str[1:-1]

    try:
        args = json.loads(args_str)
        return namespace, tool_name, args
    except json.JSONDecodeError:
        for field in ("query", "pattern"):
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', args_str)
            if match:
                return namespace, tool_name, {field: match.group(1)}
        id_match = re.search(r'"id"\s*:\s*(\d+)', args_str)
        if id_match:
            return namespace, tool_name, {"id": int(id_match.group(1))}
        url_match = re.search(r'"id"\s*:\s*"(https?://[^"]*)"', args_str)
        if url_match:
            return namespace, tool_name, {"id": url_match.group(1)}

    return None


def parse_tool_call_from_raw(text: str) -> Optional[Tuple[str, dict]]:
    """Legacy parser for ``to=functions.*`` tool calls (2-tuple return)."""
    m = re.search(r"to=functions\.(\w+)", text)
    if not m:
        return None

    tool_name = m.group(1)
    msg_match = re.search(
        r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", text, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        after = text[m.end():]
        json_match = re.search(r"(?:json|code)\s*(\{.*?\})\s*$", after, re.DOTALL)
        args_str = json_match.group(1) if json_match else after.strip()

    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            args_str = json.loads(args_str)
        except Exception:
            args_str = args_str[1:-1]

    try:
        return tool_name, json.loads(args_str)
    except json.JSONDecodeError:
        query_match = re.search(r'"query"\s*:\s*"([^"]*)"', args_str)
        if query_match:
            return tool_name, {"query": query_match.group(1)}
        url_match = re.search(r'"url"\s*:\s*"([^"]*)"', args_str)
        if url_match:
            return tool_name, {"url": url_match.group(1)}

    return None


def extract_final_answer(raw_text: str) -> Tuple[str, str]:
    """Extract ``(reasoning, final_answer)`` from Harmony channel format."""
    reasoning = ""
    answer = raw_text

    if "<|channel|>final<|message|>" in raw_text:
        parts = raw_text.split("<|channel|>final<|message|>", 1)
        reasoning = parts[0].strip()
        answer = parts[1].strip()
        reasoning = re.sub(r"<\|[^|]+\|>", "", reasoning).strip()
        reasoning = re.sub(r"\bassistant\b", "", reasoning).strip()
        reasoning = re.sub(r"\b(analysis|commentary)\b", "", reasoning).strip()
    elif re.search(r"(?:assistant\s*)?final", raw_text):
        final_match = re.search(
            r"(?:assistant\s*)?final\s*(.*?)$", raw_text, re.DOTALL
        )
        if final_match:
            answer = final_match.group(1).strip()
            reasoning = raw_text[: final_match.start()].strip()
            reasoning = re.sub(r"\bassistant\b", "", reasoning).strip()
            reasoning = re.sub(r"\b(analysis|commentary)\b", "", reasoning).strip()
    elif raw_text.startswith("analysis"):
        answer = raw_text[len("analysis"):].strip()

    answer = re.sub(r"<\|[^|]+\|>", "", answer).strip()
    return reasoning, answer
