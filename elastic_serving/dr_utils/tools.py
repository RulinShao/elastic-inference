"""
Tool implementations for deep research agents.

Each tool class/function:
  1. Defines an OpenAI-compatible tool spec (for ``apply_chat_template``).
  2. Implements the actual API call (Serper, Jina, Semantic Scholar, etc.).

Tools are split into Harmony namespaces:
  ``browser.*``    — built-in (Serper search + Jina reader + local find)
  ``python``       — built-in (stateful Jupyter code execution)
  ``functions.*``  — custom (paper_search via Semantic Scholar)

To add a new tool:
  1. Define its spec dict (same schema as OpenAI function calling).
  2. Add an ``async def`` implementation.
  3. Register it in ``CUSTOM_TOOLS`` and ``execute_custom_tool``.

Several tools wrap functions from agent-papers-cli via ``asyncio.to_thread``:
  ``paper_details``    — single paper lookup (S2 ID / DOI / ArXiv)
  ``paper_citations``  — citation graph navigation
  ``read_paper``       — fetch & parse arxiv PDFs into sections
  ``scholar_search``   — Google Scholar via Serper
"""

import asyncio
import os
import queue
import threading
import time
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

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        blocked_domains: Optional[List[str]] = None,
    ):
        self.http_client = http_client
        self.blocked_domains = blocked_domains or []
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

        # Filter out blocked domains
        if self.blocked_domains:
            results = [
                r for r in results
                if not any(d in r.get("link", "") for d in self.blocked_domains)
            ]

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
            if self.blocked_domains and any(d in id for d in self.blocked_domains):
                return f"Error: domain blocked. URL: {id[:80]}"
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
# ---- paper_search (functions.* namespace, Semantic Scholar) ----
# =============================================================================

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_FIELDS = (
    "paperId,title,abstract,authors,authors.name,year,url,"
    "citationCount,externalIds,isOpenAccess,openAccessPdf,venue"
)

PAPER_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "paper_search",
        "description": (
            "Search academic papers via Semantic Scholar. Two modes: "
            "'papers' returns paper metadata (titles, abstracts, authors, "
            "PDF links); 'snippets' returns relevant text passages from "
            "within paper content (more useful for finding specific facts "
            "and claims). Use 'snippets' by default."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for academic papers.",
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "Search mode: 'snippets' (default) returns text "
                        "passages from paper content; 'papers' returns paper "
                        "metadata with abstracts and PDF links."
                    ),
                },
                "limit": {
                    "type": "number",
                    "description": "Max results to return (default: 5).",
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
                        "Comma-separated fields of study. "
                        "E.g. 'Computer Science', 'Medicine'."
                    ),
                },
                "venue": {
                    "type": "string",
                    "description": (
                        "Restrict to a venue (ISO4 abbreviation). "
                        "E.g. 'ACL', 'NeurIPS', 'Nature'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


def _s2_headers() -> Dict[str, str]:
    api_key = os.getenv("S2_API_KEY", "")
    return {"x-api-key": api_key} if api_key else {}


def _make_pdf_url(paper: dict) -> Optional[str]:
    """Construct open-access PDF URL from paper metadata."""
    oap = paper.get("openAccessPdf")
    if oap and oap.get("url"):
        return oap["url"]
    ext = paper.get("externalIds") or {}
    if ext.get("ArXiv"):
        return f"https://arxiv.org/pdf/{ext['ArXiv']}"
    if ext.get("ACL"):
        return f"https://www.aclweb.org/anthology/{ext['ACL']}.pdf"
    return None


async def _search_papers(
    query: str,
    http_client: httpx.AsyncClient,
    limit: int = 5,
    year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    venue: Optional[str] = None,
) -> str:
    """Paper metadata search via /paper/search."""
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, 20),
        "fields": S2_PAPER_SEARCH_FIELDS,
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    if venue:
        params["venue"] = venue

    try:
        resp = await http_client.get(
            f"{S2_API_BASE}/paper/search",
            params=params,
            headers=_s2_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Paper search error: {e}"

    papers = data.get("data", [])
    if not papers:
        return f"No papers found for: {query}"

    lines = [f'Paper search: "{query}"', ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        year_val = p.get("year", "")
        citations = p.get("citationCount", 0)
        venue_val = p.get("venue", "")
        authors = p.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:4])
        if len(authors) > 4:
            author_str += " et al."
        abstract = p.get("abstract", "") or ""
        if len(abstract) > 500:
            abstract = abstract[:500] + "..."
        s2_url = p.get("url", "")
        ext = p.get("externalIds", {}) or {}
        doi = ext.get("DOI", "")
        pdf_url = _make_pdf_url(p)

        lines.append(f"[{i}] {title}")
        if author_str:
            lines.append(f"    Authors: {author_str}")
        meta = []
        if year_val:
            meta.append(f"Year: {year_val}")
        if venue_val:
            meta.append(f"Venue: {venue_val}")
        meta.append(f"Citations: {citations}")
        lines.append(f"    {' | '.join(meta)}")
        if doi:
            lines.append(f"    DOI: {doi}")
        if pdf_url:
            lines.append(f"    PDF: {pdf_url}")
        if s2_url:
            lines.append(f"    URL: {s2_url}")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)


async def _search_snippets(
    query: str,
    http_client: httpx.AsyncClient,
    limit: int = 10,
    year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    venue: Optional[str] = None,
) -> str:
    """Snippet (passage) search via /snippet/search.

    Returns actual text passages from within paper content, not just
    abstracts.  Much more useful for finding specific facts and claims.
    """
    params: Dict[str, Any] = {
        "query": query,
        "limit": min(limit, 20),
    }
    if year:
        params["year"] = year
    if fields_of_study:
        params["fieldsOfStudy"] = fields_of_study
    if venue:
        params["venue"] = venue

    try:
        resp = await http_client.get(
            f"{S2_API_BASE}/snippet/search",
            params=params,
            headers=_s2_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"Snippet search error: {e}"

    snippets = data.get("data", [])
    if not snippets:
        return f"No paper snippets found for: {query}"

    lines = [f'Paper snippet search: "{query}"', ""]
    for i, s in enumerate(snippets, 1):
        # Extract snippet text — may be a string or a dict with "text" key
        raw_snippet = s.get("snippet", s.get("text", ""))
        if isinstance(raw_snippet, dict):
            snippet_text = raw_snippet.get("text", "")
            snippet_section = raw_snippet.get("section", "")
        else:
            snippet_text = raw_snippet
            snippet_section = s.get("section", "")

        paper = s.get("paper", {})
        title = paper.get("title", s.get("title", ""))
        authors = paper.get("authors", s.get("authors", []))
        year_val = paper.get("year", s.get("year", ""))
        paper_id = paper.get("paperId", s.get("paperId", ""))
        venue_val = paper.get("venue", s.get("venue", ""))

        author_str = ""
        if authors:
            author_str = ", ".join(
                a.get("name", a) if isinstance(a, dict) else str(a)
                for a in authors[:3]
            )
            if len(authors) > 3:
                author_str += " et al."

        lines.append(f"[{i}] {title}")
        meta = []
        if author_str:
            meta.append(author_str)
        if year_val:
            meta.append(str(year_val))
        if venue_val:
            meta.append(venue_val)
        if meta:
            lines.append(f"    {' | '.join(meta)}")
        if paper_id:
            lines.append(
                f"    URL: https://www.semanticscholar.org/paper/{paper_id}"
            )
        if snippet_section:
            lines.append(f"    Section: {snippet_section}")
        if snippet_text:
            # Truncate very long snippets
            if len(snippet_text) > 800:
                snippet_text = snippet_text[:800] + "..."
            lines.append(f"    Snippet: {snippet_text}")
        lines.append("")

    return "\n".join(lines)


async def paper_search(
    query: str,
    http_client: httpx.AsyncClient,
    mode: str = "snippets",
    limit: int = 5,
    year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    venue: Optional[str] = None,
) -> str:
    """Search academic papers via Semantic Scholar.

    mode='snippets' — returns text passages from paper content (default).
    mode='papers'   — returns paper metadata (titles, abstracts, PDF links).
    """
    if mode == "papers":
        return await _search_papers(
            query, http_client, limit=limit, year=year,
            fields_of_study=fields_of_study, venue=venue,
        )
    else:
        return await _search_snippets(
            query, http_client, limit=limit, year=year,
            fields_of_study=fields_of_study, venue=venue,
        )


# =============================================================================
# ---- pubmed_search (functions.* namespace, NCBI PubMed) ----
# =============================================================================

PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

PUBMED_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "pubmed_search",
        "description": (
            "Search biomedical and life science literature via PubMed "
            "(NCBI). Returns paper titles, abstracts, authors, journal, "
            "and PubMed URLs. Use this for medical, biological, clinical, "
            "and health science questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Supports PubMed syntax: "
                        "MeSH terms, boolean operators (AND/OR/NOT), "
                        "field tags like [Author], [Journal], [Title]."
                    ),
                },
                "limit": {
                    "type": "number",
                    "description": "Max results to return (default: 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def _extract_xml_text(element) -> str:
    """Extract all text from an XML element including nested tags."""
    if element is None:
        return ""
    return " ".join(t.strip() for t in element.itertext() if t.strip())


async def pubmed_search(
    query: str,
    http_client: httpx.AsyncClient,
    limit: int = 5,
) -> str:
    """Search PubMed via NCBI E-utilities API."""
    from xml.etree import ElementTree

    # Step 1: search for PubMed IDs
    try:
        resp = await http_client.get(
            f"{PUBMED_BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": min(limit, 20),
                "sort": "relevance",
                "retmode": "xml",
            },
            timeout=15,
        )
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        id_list = [el.text for el in root.findall("./IdList/Id") if el.text]
    except Exception as e:
        return f"PubMed search error: {e}"

    if not id_list:
        return f"No PubMed results for: {query}"

    # Step 2: fetch paper details
    try:
        resp = await http_client.get(
            f"{PUBMED_BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            },
            timeout=15,
        )
        resp.raise_for_status()
        papers_xml = ElementTree.fromstring(resp.content)
    except Exception as e:
        return f"PubMed fetch error: {e}"

    lines = [f'PubMed search: "{query}"', ""]

    for i, article_el in enumerate(papers_xml.findall("./PubmedArticle"), 1):
        article = article_el.find(".//Article")
        pmid_el = article_el.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title = _extract_xml_text(article.find(".//ArticleTitle")) if article is not None else ""

        # Abstract (may have labeled sections like BACKGROUND, METHODS, etc.)
        abstract_parts = []
        if article is not None and article.find(".//Abstract") is not None:
            for ab_text in article.findall(".//Abstract/AbstractText"):
                label = ab_text.attrib.get("Label", "")
                text = _extract_xml_text(ab_text)
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        if len(abstract) > 600:
            abstract = abstract[:600] + "..."

        # Authors
        authors = []
        if article is not None:
            for auth in article.findall(".//Author"):
                last = auth.find("./LastName")
                fore = auth.find("./ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{fore.text} {last.text}")
        author_str = ", ".join(authors[:4])
        if len(authors) > 4:
            author_str += " et al."

        # Journal and year
        journal = ""
        if article is not None:
            j_el = article.find(".//Journal/Title")
            if j_el is not None:
                journal = j_el.text or ""
        year = ""
        if article is not None:
            y_el = article.find(".//Journal/JournalIssue/PubDate/Year")
            if y_el is not None:
                year = y_el.text or ""

        lines.append(f"[{i}] {title}")
        if author_str:
            lines.append(f"    Authors: {author_str}")
        meta = []
        if year:
            meta.append(f"Year: {year}")
        if journal:
            meta.append(journal)
        if meta:
            lines.append(f"    {' | '.join(meta)}")
        if pmid:
            lines.append(f"    URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        if abstract:
            lines.append(f"    Abstract: {abstract}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# ---- Python code execution (Jupyter kernel) ----
# =============================================================================

# Pre-import setup run in every new kernel
_PYTHON_SANDBOX_SETUP = """\
import os, sys, math, json, re, collections, itertools, functools, statistics
try:
    import numpy as np
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
try:
    import sympy
except ImportError:
    pass
"""

MAX_PYTHON_OUTPUT = 30_000  # truncate output to this many chars


class PythonSession:
    """Stateful Jupyter kernel for code execution.

    Matches gpt-oss's native ``python`` builtin tool: a persistent Jupyter
    notebook environment with 120s timeout.

    Each call to :meth:`execute` runs code in the same kernel, so variables
    persist across calls (stateful, like ChatGPT's Code Interpreter).

    Parameters
    ----------
    timeout : int
        Max seconds per code execution (default 120, matching model prompt).
    allowed_dirs : list[str] | None
        If set, ``os.chdir`` is called to the first dir and a warning is
        printed for filesystem access outside these directories.
    """

    def __init__(
        self,
        timeout: int = 120,
        allowed_dirs: Optional[List[str]] = None,
    ):
        from jupyter_client import KernelManager

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=30)
        self.timeout = timeout

        # Execution stats
        self.stats = {"total": 0, "success": 0, "error": 0, "timeout": 0, "no_output": 0}

        # Sandbox init: pre-import packages + set working dir
        setup = _PYTHON_SANDBOX_SETUP
        if allowed_dirs:
            first_dir = allowed_dirs[0]
            setup += f"\nos.makedirs({first_dir!r}, exist_ok=True)\nos.chdir({first_dir!r})\n"
        self.execute(setup, _skip_stats=True)

    def execute(self, code: str, _skip_stats: bool = False) -> str:
        """Execute *code* in the kernel and return combined output.

        Returns stdout, display results, and error tracebacks concatenated.
        Output is truncated to ``MAX_PYTHON_OUTPUT`` characters.
        """
        if not _skip_stats:
            self.stats["total"] += 1
        msg_id = self.kc.execute(code)
        outputs: List[str] = []
        deadline = time.time() + self.timeout

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                outputs.append(f"\n[Execution timed out after {self.timeout}s]")
                break
            try:
                msg = self.kc.get_iopub_msg(timeout=remaining)
            except queue.Empty:
                outputs.append(f"\n[Execution timed out after {self.timeout}s]")
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append(content["text"])
            elif msg_type in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    outputs.append(text)
            elif msg_type == "error":
                # Strip ANSI escape codes from traceback
                import re as _re
                tb = "\n".join(content.get("traceback", []))
                tb = _re.sub(r"\x1b\[[0-9;]*m", "", tb)
                outputs.append(tb)
            elif (
                msg_type == "status"
                and content.get("execution_state") == "idle"
            ):
                break

        result = "\n".join(outputs).strip()
        if len(result) > MAX_PYTHON_OUTPUT:
            result = result[:MAX_PYTHON_OUTPUT] + f"\n... [output truncated at {MAX_PYTHON_OUTPUT} chars]"

        # Update stats
        if not _skip_stats:
            has_error = any("Error" in o or "Traceback" in o for o in outputs)
            timed_out = any("timed out" in o for o in outputs)
            if timed_out:
                self.stats["timeout"] += 1
            elif has_error:
                self.stats["error"] += 1
            elif not result or result == "(no output)":
                self.stats["no_output"] += 1
            else:
                self.stats["success"] += 1

        return result if result else "(no output)"

    def close(self):
        """Shut down the kernel and release resources."""
        try:
            self.kc.stop_channels()
        except Exception:
            pass
        try:
            self.km.shutdown_kernel(now=True)
        except Exception:
            pass


# =============================================================================
# ---- paper_details (functions.* namespace, Semantic Scholar) ----
# =============================================================================

PAPER_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "paper_details",
        "description": (
            "Get details for a specific paper by Semantic Scholar paper ID, "
            "DOI (prefix with 'DOI:'), or ArXiv ID (prefix with 'ArXiv:'). "
            "Returns title, authors, year, venue, citation count, abstract, "
            "and PDF link."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": (
                        "Paper identifier. Examples: a Semantic Scholar ID, "
                        "'DOI:10.18653/v1/2023.acl-long.1', "
                        "'ArXiv:2301.12345', or a Semantic Scholar URL."
                    ),
                },
            },
            "required": ["paper_id"],
        },
    },
}


async def paper_details(paper_id: str) -> str:
    """Look up a single paper by ID using agent-papers-cli."""
    try:
        from search.api import get_paper_details as _get_paper_details
    except ImportError:
        return (
            "Error: agent-papers-cli is not installed. "
            "Install it with: pip install -e ../agent-papers-cli"
        )

    try:
        result = await asyncio.to_thread(_get_paper_details, paper_id)
    except Exception as e:
        return f"Paper details error: {e}"

    lines = [f"Paper: {result.title}"]
    if result.authors:
        lines.append(f"Authors: {result.authors}")
    meta = []
    if result.year:
        meta.append(f"Year: {result.year}")
    if result.venue:
        meta.append(f"Venue: {result.venue}")
    if result.citation_count is not None:
        meta.append(f"Citations: {result.citation_count}")
    if meta:
        lines.append(" | ".join(meta))
    if result.arxiv_id:
        lines.append(f"ArXiv: {result.arxiv_id}")
    if result.url:
        lines.append(f"URL: {result.url}")
    if result.snippet:
        lines.append(f"Abstract: {result.snippet}")
    return "\n".join(lines)


# =============================================================================
# ---- paper_citations (functions.* namespace, Semantic Scholar) ----
# =============================================================================

PAPER_CITATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "paper_citations",
        "description": (
            "Get the citation graph for a paper. 'citations' returns papers "
            "that cite this paper; 'references' returns papers this paper "
            "cites. Useful for literature review and finding related work."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": (
                        "Semantic Scholar paper ID, DOI, or ArXiv ID."
                    ),
                },
                "direction": {
                    "type": "string",
                    "description": (
                        "'citations' (papers citing this one, default) "
                        "or 'references' (papers this one cites)."
                    ),
                },
                "limit": {
                    "type": "number",
                    "description": "Max results to return (default: 10).",
                },
            },
            "required": ["paper_id"],
        },
    },
}


async def paper_citations(
    paper_id: str,
    direction: str = "citations",
    limit: int = 10,
) -> str:
    """Get citations or references for a paper using agent-papers-cli."""
    try:
        from search.api import get_citations, get_references
    except ImportError:
        return (
            "Error: agent-papers-cli is not installed. "
            "Install it with: pip install -e ../agent-papers-cli"
        )

    try:
        if direction == "references":
            results = await asyncio.to_thread(
                get_references, paper_id, limit=limit
            )
            header = f"References (papers cited by {paper_id})"
        else:
            results = await asyncio.to_thread(
                get_citations, paper_id, limit=limit
            )
            header = f"Citations (papers citing {paper_id})"
    except Exception as e:
        return f"Citation lookup error: {e}"

    if not results:
        return f"No {direction} found for paper: {paper_id}"

    lines = [header, ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        meta = []
        if r.authors:
            meta.append(r.authors)
        if r.year:
            meta.append(str(r.year))
        if r.venue:
            meta.append(r.venue)
        if meta:
            lines.append(f"    {' | '.join(meta)}")
        if r.is_influential:
            lines.append("    ** Influential citation **")
        if r.paper_id:
            lines.append(
                f"    URL: https://www.semanticscholar.org/paper/{r.paper_id}"
            )
        if r.contexts:
            for ctx in r.contexts[:2]:
                lines.append(f"    Context: {ctx[:300]}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# ---- read_paper (functions.* namespace, arxiv PDF fetch + parse) ----
# =============================================================================

READ_PAPER_TOOL = {
    "type": "function",
    "function": {
        "name": "read_paper",
        "description": (
            "Fetch and read an arxiv paper. Downloads the PDF, parses it "
            "into structured sections, and returns the content. Supports "
            "multiple reading modes:\n"
            "  'outline' — table of contents with section headings (default)\n"
            "  'skim' — headings + first N sentences per section\n"
            "  'read' — full text of a specific section\n"
            "  'search' — search for keywords within the paper\n"
            "  'goto' — jump to a reference (e.g. s3 for section 3, "
            "e1 for external link 1, c5 for citation 5)\n"
            "Typical workflow: outline first, then skim or read sections "
            "of interest, use goto to follow references."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": (
                        "ArXiv ID (e.g. '2301.12345') or full URL "
                        "(e.g. 'https://arxiv.org/abs/2301.12345')."
                    ),
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "Reading mode: 'outline' (default), 'skim', "
                        "'read', 'search', or 'goto'."
                    ),
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Section name for 'read' mode (e.g. 'Introduction', "
                        "'Method'). Case-insensitive substring match."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for 'search' mode."
                    ),
                },
                "ref_id": {
                    "type": "string",
                    "description": (
                        "Reference ID for 'goto' mode (e.g. 's3' for "
                        "section 3, 'e1' for external link, 'c5' for "
                        "citation 5)."
                    ),
                },
                "num_lines": {
                    "type": "number",
                    "description": (
                        "Number of sentences per section for 'skim' mode "
                        "(default: 3)."
                    ),
                },
            },
            "required": ["arxiv_id"],
        },
    },
}


def _fetch_and_parse(arxiv_id: str):
    """Synchronous helper: fetch PDF + parse into Document."""
    from paper.api import fetch_paper, parse_paper

    paper_id, pdf_path = fetch_paper(arxiv_id)
    doc = parse_paper(paper_id, pdf_path)
    return doc


def _format_header(doc) -> list:
    """Format paper header lines."""
    lines = []
    if doc.metadata.title:
        lines.append(f"# {doc.metadata.title}")
    if doc.metadata.authors:
        lines.append(f"Authors: {', '.join(doc.metadata.authors)}")
    if doc.metadata.url:
        lines.append(f"URL: {doc.metadata.url}")
    lines.append("")
    return lines


def _do_outline(doc) -> str:
    """Return table of contents + abstract."""
    lines = _format_header(doc)
    lines.append("## Sections")
    for i, s in enumerate(doc.sections, 1):
        indent = "  " * (s.level - 1)
        lines.append(f"{indent}{i}. {s.heading} [ref=s{i}]")
    lines.append("")

    # Find abstract section
    abstract_sections = [
        s for s in doc.sections if "abstract" in s.heading.lower()
    ]
    if abstract_sections:
        lines.append("## Abstract")
        abstract_text = abstract_sections[0].content
        if len(abstract_text) > 2000:
            abstract_text = abstract_text[:2000] + "..."
        lines.append(abstract_text)
    elif doc.metadata.abstract:
        lines.append("## Abstract")
        lines.append(doc.metadata.abstract)

    lines.append("")
    lines.append(
        "Use read_paper with mode='read' and section=<name> to read a "
        "section, or mode='skim' for a quick overview."
    )
    return "\n".join(lines)


def _do_skim(doc, num_lines: int = 3) -> str:
    """Return headings + first N sentences per section."""
    lines = _format_header(doc)
    for i, section in enumerate(doc.sections, 1):
        indent = "  " * (section.level - 1)
        lines.append(f"{indent}## {section.heading} [ref=s{i}]")

        sentences = section.sentences[:num_lines]
        if sentences:
            for sent in sentences:
                lines.append(f"{indent}  {sent.text}")
        elif section.content:
            content_lines = [
                l.strip() for l in section.content.split("\n") if l.strip()
            ]
            for line in content_lines[:num_lines]:
                lines.append(f"{indent}  {line}")
        lines.append("")

    return "\n".join(lines)


def _do_read(doc, section_name: str) -> str:
    """Return full text of a specific section."""
    section_lower = section_name.lower()
    matched = [
        s for s in doc.sections if section_lower in s.heading.lower()
    ]
    if not matched:
        headings = [s.heading for s in doc.sections]
        return (
            f"Section '{section_name}' not found. "
            f"Available sections: {', '.join(headings)}"
        )
    lines = []
    for s in matched:
        lines.append(f"## {s.heading}")
        lines.append("")
        content = s.content
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def _do_search(doc, query: str) -> str:
    """Search for keywords within the paper."""
    query_lower = query.lower()
    matches = []
    for section in doc.sections:
        text = section.content
        text_lower = text.lower()
        pos = 0
        while True:
            idx = text_lower.find(query_lower, pos)
            if idx == -1:
                break
            # Get context around the match
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 200)
            context = text[start:end].strip()
            matches.append((section.heading, context))
            pos = idx + len(query)

    if not matches:
        return f'No matches found for "{query}" in this paper.'

    lines = [f'Search results for "{query}" ({len(matches)} match(es)):', ""]
    for i, (heading, context) in enumerate(matches[:20], 1):
        lines.append(f"[{i}] In section: {heading}")
        lines.append(f"    ...{context}...")
        lines.append("")
    if len(matches) > 20:
        lines.append(f"... and {len(matches) - 20} more matches")
    return "\n".join(lines)


def _do_goto(doc, ref_id: str) -> str:
    """Jump to a reference (section, external link, or citation)."""
    import re as _re

    # Parse ref_id: s3 -> section, e1 -> external, c5 -> citation
    m = _re.match(r"^(s|e|c|f|t|eq)(\d+)$", ref_id)
    if not m:
        return (
            f"Invalid ref_id: '{ref_id}'. "
            f"Use s<N> for sections, e<N> for external links, "
            f"c<N> for citations."
        )

    kind, num = m.group(1), int(m.group(2))

    if kind == "s":
        # Section reference
        if num < 1 or num > len(doc.sections):
            return (
                f"Section s{num} out of range. "
                f"Paper has {len(doc.sections)} sections."
            )
        section = doc.sections[num - 1]
        lines = [f"## {section.heading}", ""]
        # Show a preview (first 10 sentences or lines)
        sentences = section.sentences[:10]
        if sentences:
            for sent in sentences:
                lines.append(f"  {sent.text}")
        elif section.content:
            content_lines = [
                l.strip() for l in section.content.split("\n") if l.strip()
            ]
            for line in content_lines[:10]:
                lines.append(f"  {line}")
        total = len(section.sentences) or len(
            [l for l in section.content.split("\n") if l.strip()]
        )
        if total > 10:
            lines.append("")
            lines.append(
                f"Showing 10 of {total} sentences. "
                f"Use mode='read' section='{section.heading}' for full text."
            )
        return "\n".join(lines)

    elif kind == "e":
        # External link reference
        seen_urls = set()
        ext_idx = 0
        for link in doc.links:
            if link.kind == "external" and link.url not in seen_urls:
                seen_urls.add(link.url)
                ext_idx += 1
                if ext_idx == num:
                    lines = [f"External link e{num}:"]
                    lines.append(f"  Text: {link.text}")
                    lines.append(f"  URL: {link.url}")
                    lines.append(f"  Page: {link.page + 1}")
                    return "\n".join(lines)
        return f"External link e{num} not found."

    elif kind == "c":
        # Citation reference
        seen_cites = set()
        cite_idx = 0
        for link in doc.links:
            if link.kind == "citation" and link.text not in seen_cites:
                seen_cites.add(link.text)
                cite_idx += 1
                if cite_idx == num:
                    lines = [f"Citation c{num}: {link.text}"]
                    if link.target_page >= 0:
                        lines.append(f"  Target page: {link.target_page + 1}")
                    if link.dest_name:
                        lines.append(f"  Destination: {link.dest_name}")
                    return "\n".join(lines)
        return f"Citation c{num} not found."

    return f"Ref type '{kind}' not supported."


async def read_paper(
    arxiv_id: str,
    mode: str = "outline",
    section: Optional[str] = None,
    query: Optional[str] = None,
    ref_id: Optional[str] = None,
    num_lines: int = 3,
) -> str:
    """Fetch an arxiv paper and interact with its content.

    Supports modes: outline, skim, read, search, goto.
    """
    try:
        from paper.api import fetch_paper, parse_paper  # noqa: F401
    except ImportError:
        return (
            "Error: agent-papers-cli is not installed. "
            "Install with: pip install 'elastic-serving[papers]'"
        )

    try:
        doc = await asyncio.to_thread(_fetch_and_parse, arxiv_id)
    except Exception as e:
        return f"Error reading paper: {e}"

    if mode == "skim":
        return _do_skim(doc, num_lines=num_lines)
    elif mode == "read":
        if not section:
            return (
                "mode='read' requires a section name. "
                "Use mode='outline' first to see available sections."
            )
        return _do_read(doc, section)
    elif mode == "search":
        if not query:
            return "mode='search' requires a query argument."
        return _do_search(doc, query)
    elif mode == "goto":
        if not ref_id:
            return (
                "mode='goto' requires a ref_id argument (e.g. 's3', 'e1', 'c5'). "
                "Use mode='outline' to see available refs."
            )
        return _do_goto(doc, ref_id)
    else:
        # Default: outline
        return _do_outline(doc)


# =============================================================================
# ---- scholar_search (functions.* namespace, Google Scholar via Serper) ----
# =============================================================================

SCHOLAR_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "scholar_search",
        "description": (
            "Search Google Scholar for academic papers. Returns titles, "
            "authors, publication info, year, citation counts, and snippets. "
            "Useful as a complement to paper_search (Semantic Scholar) for "
            "broader coverage."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query for Google Scholar.",
                },
                "limit": {
                    "type": "number",
                    "description": "Max results to return (default: 10).",
                },
            },
            "required": ["query"],
        },
    },
}


async def scholar_search(query: str, limit: int = 10) -> str:
    """Search Google Scholar using agent-papers-cli."""
    try:
        from search.api import search_scholar as _search_scholar
    except ImportError:
        return (
            "Error: agent-papers-cli is not installed. "
            "Install it with: pip install -e ../agent-papers-cli"
        )

    try:
        results = await asyncio.to_thread(
            _search_scholar, query, num_results=limit
        )
    except Exception as e:
        return f"Scholar search error: {e}"

    if not results:
        return f"No Google Scholar results for: {query}"

    lines = [f'Google Scholar: "{query}"', ""]
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}")
        meta = []
        if r.authors:
            meta.append(r.authors)
        if r.year:
            meta.append(str(r.year))
        if r.citation_count is not None:
            meta.append(f"Cited by {r.citation_count}")
        if meta:
            lines.append(f"    {' | '.join(meta)}")
        if r.url:
            lines.append(f"    URL: {r.url}")
        if r.snippet:
            lines.append(f"    {r.snippet}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# ---- Custom tool registry ----
# =============================================================================

# Paper tools (paper_details, paper_citations, read_paper, scholar_search)
# require agent-papers-cli.  Set ENABLE_PAPER_TOOLS=1 to include them.
PAPER_TOOLS = [
    PAPER_DETAILS_TOOL,
    PAPER_CITATIONS_TOOL,
    READ_PAPER_TOOL,
    SCHOLAR_SEARCH_TOOL,
]
"""Paper tool specs — only included when ``ENABLE_PAPER_TOOLS`` is set."""

_BASE_TOOLS = [PAPER_SEARCH_TOOL, PUBMED_SEARCH_TOOL]


def _build_custom_tools() -> List[dict]:
    """Build the custom tools list, conditionally including paper tools."""
    tools = list(_BASE_TOOLS)
    if os.getenv("ENABLE_PAPER_TOOLS", "").strip() in ("1", "true", "yes"):
        tools.extend(PAPER_TOOLS)
    return tools


CUSTOM_TOOLS = _build_custom_tools()
"""All custom tool specs — passed to ``apply_chat_template(tools=...)``.

Set ``ENABLE_PAPER_TOOLS=1`` to include paper browsing tools
(requires ``agent-papers-cli``).
"""


async def execute_custom_tool(
    name: str, args: dict, http_client: httpx.AsyncClient
) -> str:
    """Dispatch a ``functions.*`` tool call (custom tools)."""
    if name == "paper_search":
        return await paper_search(
            query=args.get("query", ""),
            http_client=http_client,
            mode=args.get("mode", "snippets"),
            limit=int(args.get("limit", 5)),
            year=args.get("year"),
            fields_of_study=args.get("fields_of_study"),
            venue=args.get("venue"),
        )
    elif name == "pubmed_search":
        return await pubmed_search(
            query=args.get("query", ""),
            http_client=http_client,
            limit=int(args.get("limit", 5)),
        )
    elif name == "paper_details":
        return await paper_details(
            paper_id=args.get("paper_id", ""),
        )
    elif name == "paper_citations":
        return await paper_citations(
            paper_id=args.get("paper_id", ""),
            direction=args.get("direction", "citations"),
            limit=int(args.get("limit", 10)),
        )
    elif name == "read_paper":
        return await read_paper(
            arxiv_id=args.get("arxiv_id", ""),
            mode=args.get("mode", "outline"),
            section=args.get("section"),
            query=args.get("query"),
            ref_id=args.get("ref_id"),
            num_lines=int(args.get("num_lines", 3)),
        )
    elif name == "scholar_search":
        return await scholar_search(
            query=args.get("query", ""),
            limit=int(args.get("limit", 10)),
        )
    # Model sometimes confuses namespaces (e.g. functions.browser instead
    # of browser.search).  Return a helpful nudge rather than a hard error.
    return (
        f"Unknown tool: functions.{name}. "
        f"Available: paper_search, pubmed_search, paper_details, "
        f"paper_citations, read_paper, scholar_search. "
        f"For web search use browser.search; to open a page use browser.open."
    )

