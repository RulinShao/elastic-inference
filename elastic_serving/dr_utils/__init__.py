"""
Deep research utilities — tools, prompts, and helpers.

Submodules:
  prompts   — system prompts and model identity strings
  tools     — tool specs, implementations, and dispatchers
"""

from elastic_serving.dr_utils.tools import (
    CUSTOM_TOOLS,
    PAPER_SEARCH_TOOL,
    PUBMED_SEARCH_TOOL,
    PAPER_DETAILS_TOOL,
    PAPER_CITATIONS_TOOL,
    READ_PAPER_TOOL,
    SCHOLAR_SEARCH_TOOL,
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
from elastic_serving.dr_utils.prompts import (
    MODEL_IDENTITY,
    SYSTEM_PROMPT,
)

__all__ = [
    # Prompts
    "SYSTEM_PROMPT",
    "MODEL_IDENTITY",
    # Tool specs
    "CUSTOM_TOOLS",
    "PAPER_SEARCH_TOOL",
    "PUBMED_SEARCH_TOOL",
    "PAPER_DETAILS_TOOL",
    "PAPER_CITATIONS_TOOL",
    "READ_PAPER_TOOL",
    "SCHOLAR_SEARCH_TOOL",
    # Tool implementations
    "BrowserSession",
    "PythonSession",
    "paper_search",
    "pubmed_search",
    "paper_details",
    "paper_citations",
    "read_paper",
    "scholar_search",
    "execute_custom_tool",
]
