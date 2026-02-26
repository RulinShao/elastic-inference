"""
Model-specific tool format adapters.

Provides a ``ToolAdapter`` ABC with concrete implementations for:
  - ``HarmonyAdapter`` — gpt-oss Harmony format
  - ``Qwen3Adapter`` — Qwen3.5 native tool format

The adapter abstracts 5 model-specific operations:
  1. ``build_prompt`` — initial prompt construction
  2. ``parse_tool_call`` — extract tool calls from model output
  3. ``format_tool_response`` — append tool result to prompt
  4. ``extract_final_answer`` — extract reasoning + answer
  5. ``stop_tokens`` / ``extra_body`` — vLLM request config

Usage::

    adapter = detect_adapter(tokenizer)  # auto-detect from chat template
    prompt = adapter.build_prompt(tokenizer, question, ...)
    tc = adapter.parse_tool_call(raw_text)
    prompt = adapter.format_tool_response(prompt, raw_text, ...)
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from elastic_serving.dr_utils import CUSTOM_TOOLS, SYSTEM_PROMPT
from elastic_serving.dr_utils.prompts import MODEL_IDENTITY


# =============================================================================
# Abstract base class
# =============================================================================


class ToolAdapter(ABC):
    """Model-specific tool format adapter."""

    @property
    @abstractmethod
    def stop_tokens(self) -> List[str]:
        """vLLM stop strings for this model format."""
        ...

    @property
    @abstractmethod
    def stop_tokens_no_call(self) -> List[str]:
        """Stop strings that force a final answer (no more tool calls)."""
        ...

    @property
    def extra_body(self) -> Dict[str, Any]:
        """Extra vLLM request body params (e.g. skip_special_tokens)."""
        return {}

    @abstractmethod
    def build_prompt(
        self,
        tokenizer: Any,
        user_message: str,
        system_prompt: str = SYSTEM_PROMPT,
        custom_tools: Optional[List[dict]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        enable_python: bool = False,
    ) -> str:
        """Build the initial prompt using the model's chat template."""
        ...

    @abstractmethod
    def parse_tool_call(
        self, text: str
    ) -> Optional[Tuple[str, str, dict]]:
        """Parse a tool call from model output.

        Returns ``(namespace, tool_name, args_dict)`` or ``None``.
        """
        ...

    @abstractmethod
    def format_tool_response(
        self,
        prompt: str,
        model_output: str,
        tool_name: str,
        tool_response: str,
        namespace: str = "browser",
    ) -> str:
        """Append a tool call + response to the prompt."""
        ...

    @abstractmethod
    def extract_final_answer(self, raw_text: str) -> Tuple[str, str]:
        """Extract ``(reasoning, final_answer)`` from model output."""
        ...

    def append_user_turn(
        self, prompt: str, final_answer_text: str, user_message: str
    ) -> str:
        """Append the model's final answer + a new user turn (multi-turn)."""
        # Default: delegate to model-specific implementation
        raise NotImplementedError


# =============================================================================
# Harmony adapter (gpt-oss)
# =============================================================================


class HarmonyAdapter(ToolAdapter):
    """gpt-oss Harmony format adapter.

    Wraps the existing functions from ``elastic_serving.tools`` with zero
    behavior change.
    """

    @property
    def stop_tokens(self) -> List[str]:
        return ["<|call|>", "<|return|>"]

    @property
    def stop_tokens_no_call(self) -> List[str]:
        return ["<|return|>"]

    @property
    def extra_body(self) -> Dict[str, Any]:
        return {"skip_special_tokens": False}

    def build_prompt(
        self,
        tokenizer: Any,
        user_message: str,
        system_prompt: str = SYSTEM_PROMPT,
        custom_tools: Optional[List[dict]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        enable_python: bool = False,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        builtin = ["browser", "python"] if enable_python else ["browser"]
        kwargs: Dict[str, Any] = {
            "builtin_tools": builtin,
            "model_identity": MODEL_IDENTITY,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        tools = custom_tools if custom_tools is not None else CUSTOM_TOOLS
        if tools:
            kwargs["tools"] = tools

        return tokenizer.apply_chat_template(messages, **kwargs)

    def parse_tool_call(
        self, text: str
    ) -> Optional[Tuple[str, str, dict]]:
        # ---- Special case: python tool (bare name, raw code) ----
        py_match = re.search(r"to=python\b", text)
        if py_match:
            after_py = text[py_match.end():]
            msg_match = re.search(
                r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)",
                after_py,
                re.DOTALL,
            )
            if msg_match:
                code = msg_match.group(1).strip()
            else:
                code_match = re.search(r"(?:code)\s*(.*)", after_py, re.DOTALL)
                code = (
                    code_match.group(1).strip() if code_match else after_py.strip()
                )
            if code:
                return "python", "execute", {"code": code}

        # ---- browser.* and functions.* tools ----
        m = re.search(r"to=(browser|functions)\.(\w+)", text)
        if not m:
            return None

        namespace = m.group(1)
        tool_name = m.group(2)
        after = text[m.end():]

        # Strategy 1: <|message|> AFTER the tool name
        msg_match = re.search(
            r"<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)", after, re.DOTALL
        )
        if msg_match:
            args_str = msg_match.group(1).strip()
        else:
            # Strategy 2: decoded text — JSON after json/code marker
            json_match = re.search(
                r"(?:json|code)\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
                after,
                re.DOTALL,
            )
            if json_match:
                args_str = json_match.group(1)
            else:
                # Strategy 3: any {...} after the tool name
                brace_match = re.search(
                    r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", after
                )
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

    def format_tool_response(
        self,
        prompt: str,
        model_output: str,
        tool_name: str,
        tool_response: str,
        namespace: str = "browser",
    ) -> str:
        if namespace == "python":
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

    def extract_final_answer(self, raw_text: str) -> Tuple[str, str]:
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
                reasoning = re.sub(
                    r"\b(analysis|commentary)\b", "", reasoning
                ).strip()
        elif raw_text.startswith("analysis"):
            answer = raw_text[len("analysis"):].strip()

        answer = re.sub(r"<\|[^|]+\|>", "", answer).strip()
        return reasoning, answer

    def append_user_turn(
        self, prompt: str, final_answer_text: str, user_message: str
    ) -> str:
        return (
            f"{prompt}{final_answer_text}"
            f"<|end|>"
            f"<|start|>user<|message|>{user_message}<|end|>"
            f"<|start|>assistant"
        )


# =============================================================================
# Qwen3 adapter (Qwen3.5)
# =============================================================================

# Qwen3.5 tool specs — map our tool backends to Qwen3's function format
_QWEN_BROWSER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information. Returns titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
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
                "Open and read a webpage by search result ID or full URL. "
                "Use cursor to refer to a previous search."
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
            "description": "Find exact matches of a pattern in the current page.",
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
]

_QWEN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code. The code runs in a stateful Jupyter environment "
            "(variables persist). Use print() to see output."
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
}

_QWEN_PAPER_TOOLS = [
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
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5).",
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
                    "query": {"type": "string", "description": "Search query."},
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

# Map Qwen3 tool names back to our backend namespace/name
_QWEN_TOOL_MAP = {
    "web_search": ("browser", "search"),
    "open_url": ("browser", "open"),
    "find_text": ("browser", "find"),
    "python": ("python", "execute"),
    "paper_search": ("functions", "paper_search"),
    "pubmed_search": ("functions", "pubmed_search"),
}


class Qwen3Adapter(ToolAdapter):
    """Qwen3.5 native tool format adapter.

    Tool calls use XML-style ``<tool_call><function=name>`` format.
    Thinking uses ``<think>...</think>`` blocks.
    Tool responses are wrapped in ``<tool_response>`` inside user messages.
    """

    @property
    def stop_tokens(self) -> List[str]:
        return ["</tool_call>"]

    @property
    def stop_tokens_no_call(self) -> List[str]:
        return ["<|im_end|>"]

    @property
    def extra_body(self) -> Dict[str, Any]:
        return {}

    def _build_tools(self, enable_python: bool = False) -> List[dict]:
        """Build Qwen3-format tool definitions."""
        tools = list(_QWEN_BROWSER_TOOLS) + list(_QWEN_PAPER_TOOLS)
        if enable_python:
            tools.append(_QWEN_PYTHON_TOOL)
        return tools

    def build_prompt(
        self,
        tokenizer: Any,
        user_message: str,
        system_prompt: str = SYSTEM_PROMPT,
        custom_tools: Optional[List[dict]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        enable_python: bool = False,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tools = custom_tools if custom_tools is not None else self._build_tools(enable_python)

        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    def parse_tool_call(
        self, text: str
    ) -> Optional[Tuple[str, str, dict]]:
        """Parse Qwen3 tool call format.

        Format::

            <tool_call>
            <function=name>
            <parameter=key>
            value
            </parameter>
            </function>
            </tool_call>
        """
        # Find <tool_call>...<function=name> block
        tc_match = re.search(
            r"<tool_call>\s*<function=(\w+)>(.*?)</function>",
            text,
            re.DOTALL,
        )
        if not tc_match:
            return None

        func_name = tc_match.group(1)
        params_block = tc_match.group(2)

        # Extract parameters
        args = {}
        for param_match in re.finditer(
            r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
            params_block,
            re.DOTALL,
        ):
            key = param_match.group(1)
            value = param_match.group(2).strip()
            # Try to parse as JSON for structured values
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = value

        # Map Qwen tool name to our namespace/name
        if func_name in _QWEN_TOOL_MAP:
            namespace, tool_name = _QWEN_TOOL_MAP[func_name]
        else:
            namespace, tool_name = "functions", func_name

        return namespace, tool_name, args

    def format_tool_response(
        self,
        prompt: str,
        model_output: str,
        tool_name: str,
        tool_response: str,
        namespace: str = "browser",
    ) -> str:
        # Qwen3 format:
        #   {prompt}{model_output}</tool_call><|im_end|>
        #   <|im_start|>user
        #   <tool_response>
        #   {result}
        #   </tool_response><|im_end|>
        #   <|im_start|>assistant
        #   <think>
        return (
            f"{prompt}{model_output}"
            f"</tool_call><|im_end|>\n"
            f"<|im_start|>user\n"
            f"<tool_response>\n"
            f"{tool_response}\n"
            f"</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n"
        )

    def extract_final_answer(self, raw_text: str) -> Tuple[str, str]:
        """Extract reasoning and answer from Qwen3 output.

        Qwen3 uses ``<think>reasoning</think>answer`` format.
        """
        reasoning = ""
        answer = raw_text

        if "</think>" in raw_text:
            parts = raw_text.split("</think>", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
            # Remove <think> tag from reasoning
            reasoning = re.sub(r"<think>\s*", "", reasoning).strip()

        return reasoning, answer

    def append_user_turn(
        self, prompt: str, final_answer_text: str, user_message: str
    ) -> str:
        return (
            f"{prompt}{final_answer_text}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )


# =============================================================================
# Auto-detection
# =============================================================================


def detect_adapter(tokenizer: Any) -> ToolAdapter:
    """Auto-detect the appropriate adapter from a tokenizer's chat template.

    Checks the chat template string for format-specific markers:
      - Harmony: ``<|channel|>``, ``builtin_tools``
      - Qwen3: ``<tool_call>``, ``tool_response``
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    if isinstance(template, dict):
        # Some tokenizers have multiple templates
        template = str(template)

    if "<|channel|>" in template or "builtin_tools" in template:
        return HarmonyAdapter()
    elif "<tool_call>" in template or "tool_response" in template:
        return Qwen3Adapter()

    raise ValueError(
        "Cannot auto-detect model format from chat template. "
        "Use --model-format harmony or --model-format qwen3."
    )


def get_adapter(model_format: str, tokenizer: Any = None) -> ToolAdapter:
    """Get a ToolAdapter by name or auto-detect.

    Parameters
    ----------
    model_format : str
        ``"harmony"``, ``"qwen3"``, or ``"auto"``.
    tokenizer : Any, optional
        Required when ``model_format="auto"``.
    """
    if model_format == "harmony":
        return HarmonyAdapter()
    elif model_format == "qwen3":
        return Qwen3Adapter()
    elif model_format == "auto":
        if tokenizer is None:
            raise ValueError("tokenizer required for auto-detection")
        return detect_adapter(tokenizer)
    else:
        raise ValueError(f"Unknown model format: {model_format!r}")

