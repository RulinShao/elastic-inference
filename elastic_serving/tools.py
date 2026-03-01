"""
Orchestration layer for Elastic Serving tool-use agents (Harmony format).

This module provides:
  - Harmony format parsing (``parse_tool_call``, ``extract_final_answer``)
  - Prompt building (``build_initial_prompt``, ``append_tool_round``, ``append_user_turn``)
  - Configuration constants (``STOP_TOKENS``, ``BUILTIN_TOOLS``, ``DEFAULT_MAX_TOOL_CALLS``)

Tool implementations and prompts live in ``elastic_serving.dr_utils``:
  - ``tools.py``    — BrowserSession, paper_search, tool specs
  - ``prompts.py``  — SYSTEM_PROMPT, MODEL_IDENTITY

This module re-exports commonly used symbols so callers can do
``from elastic_serving.tools import ...`` without knowing the subpackage.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# Re-export from dr_utils
from elastic_serving.dr_utils import (  # noqa: F401
    CUSTOM_TOOLS,
    MODEL_IDENTITY,
    PAPER_SEARCH_TOOL,
    PUBMED_SEARCH_TOOL,
    PAPER_DETAILS_TOOL,
    PAPER_CITATIONS_TOOL,
    READ_PAPER_TOOL,
    SCHOLAR_SEARCH_TOOL,
    SYSTEM_PROMPT,
    BrowserSession,
    PythonSession,
    execute_custom_tool,
    paper_search,
    pubmed_search,
    paper_details,
    paper_citations,
    read_paper,
    scholar_search,
)

# =============================================================================
# Configuration
# =============================================================================

BUILTIN_TOOLS: List[str] = ["browser"]
"""Passed to ``apply_chat_template(builtin_tools=...)``.

Use ``BUILTIN_TOOLS_WITH_PYTHON`` to enable the python code tool.
"""

BUILTIN_TOOLS_WITH_PYTHON: List[str] = ["browser", "python"]
"""Builtin tools including the python code execution tool."""

DEFAULT_MAX_TOOL_CALLS = 15
"""Default cap on tool calls per user turn."""

# Harmony special tokens (<|call|>, <|end|>, etc.) are single tokens.
# To make vLLM's string-based `stop` matching work, we must pass
# `skip_special_tokens: false` in the request so the tokens appear
# in the decoded output text.
#
# <|end|> appears after EVERY channel (analysis, commentary, final),
# so it's NOT a valid stop token.  We stop at:
#   <|call|>   — tool call boundary
#   <|return|> — true EOS, generated after final answer
STOP_TOKENS = ["<|call|>", "<|return|>"]
"""vLLM stop strings for Harmony generation."""

STOP_TOKENS_NO_CALL = ["<|return|>"]
"""Stop strings that force a final answer (no more tool calls)."""

# Extra vLLM request params needed for Harmony special token handling
VLLM_EXTRA_BODY = {"skip_special_tokens": False}
"""Must be included in every /v1/completions request."""


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
    enable_python: bool = False,
) -> str:
    """Build the initial prompt using ``apply_chat_template``.

    Uses ``builtin_tools=["browser"]`` (or ``["browser", "python"]`` if
    *enable_python* is True) so the model sees its native tool namespaces.
    ``system_prompt`` goes into the developer message.
    Custom tools (e.g. paper_search) go into the ``functions`` namespace.
    """
    messages: List[Dict[str, str]] = []
    messages.append({"role": "system", "content": system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    builtin = BUILTIN_TOOLS_WITH_PYTHON if enable_python else BUILTIN_TOOLS
    kwargs = {
        "builtin_tools": builtin,
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
    tools, ``"python"`` for the python code tool, or ``"functions"`` for
    custom tools.
    """
    if namespace == "python":
        # Python builtin: bare "python to=assistant" (no dot/function)
        return (
            f"{prompt}{model_output}"
            f"<|call|>"
            f"<|start|>python to=assistant"
            f"<|channel|>commentary<|message|>"
            f"{tool_response}"
            f"<|end|>"
            f"<|start|>assistant"
        )
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
    Handles:
      - ``to=browser.search`` / ``to=browser.open`` / ``to=browser.find``
      - ``to=functions.paper_search`` / ``to=functions.pubmed_search``
      - ``to=python`` (bare name, raw code — not JSON)

    The model output may be in two forms:
    1. With special tokens (raw): ``to=browser.search ... <|message|>{...}``
    2. Decoded (skip_special_tokens=True): ``commentary to=browser.search code{...}``
    """
    # ---- Special case: python tool (bare name, raw code) ----
    py_match = re.search(r"to=python\b", text)
    if py_match:
        after_py = text[py_match.end():]
        # Extract code from <|message|>...<|call|> or <|message|>...$ 
        msg_match = re.search(
            r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", after_py, re.DOTALL
        )
        if msg_match:
            code = msg_match.group(1).strip()
        else:
            # Decoded text: skip "code" marker then take rest
            code_match = re.search(r"(?:code)\s*(.*)", after_py, re.DOTALL)
            code = code_match.group(1).strip() if code_match else after_py.strip()
        if code:
            return "python", "execute", {"code": code}

    # ---- browser.* and functions.* tools ----
    m = re.search(r"to=(browser|functions)\.(\w+)", text)
    if not m:
        return None

    namespace = m.group(1)
    tool_name = m.group(2)
    after = text[m.end():]

    # Strategy 1: <|message|> AFTER the tool name (not from start of text!)
    msg_match = re.search(
        r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", after, re.DOTALL
    )
    if msg_match:
        args_str = msg_match.group(1).strip()
    else:
        # Strategy 2: decoded text — find JSON after json/code marker
        # Match the LAST {...} block (handles nested braces)
        json_match = re.search(
            r"(?:json|code)\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", after, re.DOTALL
        )
        if json_match:
            args_str = json_match.group(1)
        else:
            # Strategy 3: any {...} after the tool name
            brace_match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", after)
            if brace_match:
                args_str = brace_match.group(1)
            else:
                args_str = after.strip()

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
