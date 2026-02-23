"""
Tool implementations for deep research agents.

Each tool class/function:
  1. Defines an OpenAI-compatible tool spec (for ``apply_chat_template``).
  2. Implements the actual API call (Serper, Jina, Semantic Scholar, etc.).

Tools are split into two Harmony namespaces:
  ``browser.*``    — built-in (Serper search + Jina reader + local find)
  ``functions.*``  — custom (paper_search via Semantic Scholar)

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
# ---- Custom tool registry ----
# =============================================================================

CUSTOM_TOOLS = [PAPER_SEARCH_TOOL, PUBMED_SEARCH_TOOL]
"""All custom tool specs — passed to ``apply_chat_template(tools=...)``."""


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
    # Model sometimes confuses namespaces (e.g. functions.browser instead
    # of browser.search).  Return a helpful nudge rather than a hard error.
    return (
        f"Unknown tool: functions.{name}. "
        f"Available: paper_search, pubmed_search. "
        f"For web search use browser.search; to open a page use browser.open."
    )

