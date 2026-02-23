"""
Tool implementations for deep research agents.

Each tool class/function:
  1. Defines an OpenAI-compatible tool spec (for ``apply_chat_template``).
  2. Implements the actual API call (Serper, Jina, Semantic Scholar, etc.).

Tools are split into two Harmony namespaces:
  ``browser.*``    — built-in (Serper search + Jina reader + local find)
  ``functions.*``  — custom (snippet_search via Semantic Scholar)

To add a new tool:
  1. Define its spec dict (same schema as OpenAI function calling).
  2. Add an ``async def`` implementation.
  3. Register it in ``CUSTOM_TOOLS`` and ``execute_custom_tool``.
"""

import os
from typing import Any, Dict, List, Optional

import httpx


# =============================================================================
# ---- Browser tools (browser.* namespace, built-in Harmony) ----
# =============================================================================


class BrowserSession:
    """
    Stateful browser session tracking search results and opened pages.

    Backed by Serper (search) and Jina Reader (open).  Output is formatted
    to match what the Harmony-trained model expects:
    - Search results use ``【{id}†{title}】`` link markers
    - Opened pages are prefixed with ``[{cursor}]`` and line-numbered
    - ``find`` returns matching lines with line numbers
    """

    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self._cursor_counter = 0
        self._pages: Dict[int, Dict[str, Any]] = {}
        self._search_results: Dict[int, List[Dict[str, Any]]] = {}
        self._current_cursor: Optional[int] = None

    def _next_cursor(self) -> int:
        self._cursor_counter += 1
        return self._cursor_counter

    # ---- browser.search ----

    async def search(self, query: str, topn: int = 10, **_kw) -> str:
        """Web search via Serper API."""
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return "Error: SERPER_API_KEY not set."
        try:
            resp = await self.http_client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": topn},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            return f"Search error: {e}"

        results = data.get("organic", [])
        if not results:
            return f"No results found for: {query}"

        cursor = self._next_cursor()
        self._current_cursor = cursor

        result_list: List[Dict[str, Any]] = []
        lines = [f'Searched for "{query}"', ""]
        for i, r in enumerate(results[:topn], 1):
            url = r.get("link", "")
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            result_list.append(
                {"id": i, "url": url, "title": title, "snippet": snippet}
            )
            lines.append(f"【{i}†{title}】")
            lines.append(f"URL: {url}")
            lines.append(snippet)
            lines.append("")

        self._search_results[cursor] = result_list
        self._pages[cursor] = {
            "url": None,
            "title": f"Search: {query}",
            "lines": lines,
        }

        numbered = "\n".join(
            f"L{i + 1}: {line}" for i, line in enumerate(lines)
        )
        return f"[{cursor}]\n{numbered}"

    # ---- browser.open ----

    async def open(
        self,
        id: Any = None,
        cursor: Any = None,
        loc: Any = None,
        num_lines: Any = None,
        view_source: bool = False,
        source: str = None,
        **_kw,
    ) -> str:
        """Open a URL by search-result id, direct URL, or scroll in a page."""
        target_url: Optional[str] = None

        # id as a full URL string
        if isinstance(id, str) and id.startswith("http"):
            target_url = id
        # id as a search-result number
        elif id is not None and id != -1:
            search_cursor = (
                cursor if cursor and cursor != -1 else self._current_cursor
            )
            if search_cursor and search_cursor in self._search_results:
                for r in self._search_results[search_cursor]:
                    if r["id"] == id:
                        target_url = r["url"]
                        break
            if not target_url:
                return (
                    f"Error: link id={id} not found. "
                    f"Use browser.search first, then browser.open with a "
                    f"valid id from the results."
                )
        # Scroll within an already-opened page
        elif cursor and cursor != -1 and cursor in self._pages:
            page = self._pages[cursor]
            start = (loc - 1) if loc and loc > 0 else 0
            n = num_lines if num_lines and num_lines > 0 else 50
            view = page["lines"][start : start + n]
            numbered = "\n".join(
                f"L{start + i + 1}: {line}" for i, line in enumerate(view)
            )
            return f"[{cursor}]\n{numbered}"
        else:
            return (
                "Error: provide an id from search results or a URL string. "
                'Example: browser.open({"id": 1}) or '
                'browser.open({"id": "https://example.com"})'
            )

        # Fetch via Jina Reader (or direct fallback)
        api_key = os.getenv("JINA_API_KEY", "")
        try:
            if api_key:
                resp = await self.http_client.get(
                    f"https://r.jina.ai/{target_url}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/plain",
                        "X-Return-Format": "text",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.text
            else:
                resp = await self.http_client.get(
                    target_url, follow_redirects=True, timeout=30
                )
                content = resp.text[:20000]
        except Exception as e:
            return f"Error opening URL: {e}"

        if len(content) > 30000:
            content = content[:30000]

        all_lines = content.split("\n")
        new_cursor = self._next_cursor()
        self._current_cursor = new_cursor
        self._pages[new_cursor] = {
            "url": target_url,
            "title": target_url,
            "lines": all_lines,
        }

        start = (loc - 1) if loc and loc > 0 else 0
        n = num_lines if num_lines and num_lines > 0 else len(all_lines)
        view = all_lines[start : start + n]
        numbered = "\n".join(
            f"L{start + i + 1}: {line}" for i, line in enumerate(view)
        )
        return f"[{new_cursor}]\n{numbered}"

    # ---- browser.find ----

    async def find(self, pattern: str, cursor: Any = None, **_kw) -> str:
        """Find exact text matches in the current or specified page."""
        target = cursor if cursor and cursor != -1 else self._current_cursor
        if not target or target not in self._pages:
            return "Error: no page open. Use browser.open first."

        page = self._pages[target]
        matches = []
        for i, line in enumerate(page["lines"]):
            if pattern.lower() in line.lower():
                matches.append((i + 1, line))

        if not matches:
            return f'No matches for "{pattern}" in [{target}].'

        out = [f'Found {len(matches)} match(es) for "{pattern}" in [{target}]:']
        for line_num, line in matches[:20]:
            out.append(f"L{line_num}: {line}")
        if len(matches) > 20:
            out.append(f"... and {len(matches) - 20} more matches")
        return "\n".join(out)

    # ---- dispatcher ----

    async def execute(self, tool_name: str, args: dict) -> str:
        """Dispatch a ``browser.*`` tool call."""
        if tool_name == "search":
            return await self.search(
                query=args.get("query", ""),
                topn=args.get("topn", 10),
            )
        elif tool_name == "open":
            return await self.open(
                id=args.get("id"),
                cursor=args.get("cursor"),
                loc=args.get("loc"),
                num_lines=args.get("num_lines"),
                view_source=args.get("view_source", False),
                source=args.get("source"),
            )
        elif tool_name == "find":
            return await self.find(
                pattern=args.get("pattern", ""),
                cursor=args.get("cursor"),
            )
        return f"Unknown browser tool: {tool_name}"


# =============================================================================
# ---- snippet_search (functions.* namespace, Semantic Scholar) ----
# =============================================================================

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,abstract,authors,year,url,citationCount,externalIds"
)

SNIPPET_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "snippet_search",
        "description": (
            "Search academic papers via Semantic Scholar. Returns paper "
            "titles, abstracts, authors, publication year, and URLs. "
            "Use this for scientific questions, research findings, "
            "benchmarks, and peer-reviewed evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for academic papers.",
                },
                "limit": {
                    "type": "number",
                    "description": (
                        "Maximum number of papers to return (default: 5)."
                    ),
                },
                "year": {
                    "type": "string",
                    "description": (
                        "Publication year filter. A single year (e.g. '2024') "
                        "or a range (e.g. '2022-2025')."
                    ),
                },
                "fields_of_study": {
                    "type": "string",
                    "description": (
                        "Comma-separated fields of study to filter by. "
                        "Examples: 'Computer Science', 'Medicine', "
                        "'Computer Science,Physics'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


async def snippet_search(
    query: str,
    http_client: httpx.AsyncClient,
    limit: int = 5,
    year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
) -> str:
    """Search academic papers via Semantic Scholar API."""
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, 20),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = await http_client.get(
            SEMANTIC_SCHOLAR_API,
            params=params,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Snippet search error: {e}"

    papers = data.get("data", [])
    if not papers:
        return f"No academic papers found for: {query}"

    lines = [f'Academic paper search: "{query}"', ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        year_val = p.get("year", "")
        citations = p.get("citationCount", 0)
        authors = p.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:4])
        if len(authors) > 4:
            author_str += " et al."
        abstract = p.get("abstract", "") or ""
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        url = p.get("url", "")
        ext = p.get("externalIds", {}) or {}
        doi = ext.get("DOI", "")

        lines.append(f"[{i}] {title}")
        if author_str:
            lines.append(f"    Authors: {author_str}")
        if year_val:
            lines.append(f"    Year: {year_val} | Citations: {citations}")
        if doi:
            lines.append(f"    DOI: {doi}")
        if url:
            lines.append(f"    URL: {url}")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# ---- Custom tool registry ----
# =============================================================================

CUSTOM_TOOLS = [SNIPPET_SEARCH_TOOL]
"""All custom tool specs — passed to ``apply_chat_template(tools=...)``."""


async def execute_custom_tool(
    name: str, args: dict, http_client: httpx.AsyncClient
) -> str:
    """Dispatch a ``functions.*`` tool call (custom tools)."""
    if name == "snippet_search":
        return await snippet_search(
            query=args.get("query", ""),
            http_client=http_client,
            limit=int(args.get("limit", 5)),
            year=args.get("year"),
            fields_of_study=args.get("fields_of_study"),
        )
    return f"Unknown custom tool: {name}"


# =============================================================================
# ---- Legacy tools (functions.search / functions.open_url) ----
# =============================================================================

LEGACY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web. Returns top results with titles, URLs, snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL and read its full text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to open."}
                },
                "required": ["url"],
            },
        },
    },
]


async def execute_legacy_tool(
    name: str, args: dict, http_client: httpx.AsyncClient
) -> str:
    """Legacy dispatcher for ``functions.search`` / ``functions.open_url``."""
    if name == "search":
        query = args.get("query", "")
        if not query:
            return "Error: search requires a 'query' parameter"
        api_key = os.getenv("SERPER_API_KEY", "")
        if not api_key:
            return "Error: SERPER_API_KEY not set."
        try:
            resp = await http_client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 10},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("organic", [])
            if not results:
                return f"No results found for: {query}"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"[{i}] {r.get('title', '')}\n"
                    f"    URL: {r.get('link', '')}\n"
                    f"    {r.get('snippet', '')}"
                )
            return "\n\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"
    elif name == "open_url":
        url = args.get("url", "")
        if not url or not url.startswith(("http://", "https://")):
            hint = url or args.get("query", "")
            if hint:
                return f"Error: open_url requires a valid URL. Got: '{hint[:100]}'."
            return "Error: open_url requires a 'url' parameter with a valid URL"
        api_key = os.getenv("JINA_API_KEY", "")
        try:
            if api_key:
                resp = await http_client.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "text/plain",
                        "X-Return-Format": "text",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                content = resp.text
                if len(content) > 30000:
                    content = content[:30000] + "\n\n[... truncated ...]"
                return content
            else:
                resp = await http_client.get(url, follow_redirects=True, timeout=30)
                return resp.text[:20000]
        except Exception as e:
            return f"Error reading URL: {e}"
    return f"Unknown tool: {name}"

