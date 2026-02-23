"""
Deep research utilities — tools, prompts, and helpers.

Submodules:
  prompts               — system prompts and model identity strings
  deep_research_tools   — tool specs, implementations, and dispatchers
"""

from elastic_serving.deep_research_utils.deep_research_tools import (
    CUSTOM_TOOLS,
    LEGACY_TOOLS,
    SNIPPET_SEARCH_TOOL,
    BrowserSession,
    execute_custom_tool,
    execute_legacy_tool,
    snippet_search,
)
from elastic_serving.deep_research_utils.prompts import (
    LEGACY_SYSTEM_PROMPT,
    MODEL_IDENTITY,
    SYSTEM_PROMPT,
)

__all__ = [
    # Prompts
    "SYSTEM_PROMPT",
    "MODEL_IDENTITY",
    "LEGACY_SYSTEM_PROMPT",
    # Tool specs
    "CUSTOM_TOOLS",
    "SNIPPET_SEARCH_TOOL",
    "LEGACY_TOOLS",
    # Tool implementations
    "BrowserSession",
    "snippet_search",
    "execute_custom_tool",
    "execute_legacy_tool",
]

