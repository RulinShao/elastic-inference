import logging
from aiohttp import ClientSession
from typing import AsyncIterator, Iterable, List, Sequence, Union
import time,json,html
import asyncio
from gpt_oss.tools.simple_browser.page_contents import (
    process_html,
)
from gpt_oss.tools.simple_browser.backend import (
    VIEW_SOURCE_PREFIX,
    BackendError,
    maybe_truncate,
)

from openai_harmony import (
    Author,
    Message,
    Role,
    TextContent,
)

from gpt_oss.tools.simple_browser.simple_browser_tool import SimpleBrowserTool,maybe_get_function_args
from gpt_oss.tools.simple_browser.backend import BackendError, maybe_truncate

import os


logger = logging.getLogger(__name__)

def filter_blocked_search_results(
    results: Iterable[tuple[str, str, str]],
    blocked_substrings: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Filter search results whose title/url/snippet match blocked substrings."""
    normalized_blocks = [s.strip().lower() for s in blocked_substrings if s and s.strip()]
    if not normalized_blocks:
        return list(results)

    filtered = []
    for title, url, summary in results:
        haystack = " ".join(v for v in (title, url, summary) if v).lower()
        if any(blocked in haystack for blocked in normalized_blocks):
            continue
        filtered.append((title, url, summary))
    return filtered


def sanitize_dict_keys(d):
    """Remove None keys from dictionary."""
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if k is not None}

class BrowserTool(SimpleBrowserTool):
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        def make_error_message(error: str) -> Message:
            return self.make_response(
                content=TextContent(text=json.dumps({"error": error})),
                author=Author(role=Role.TOOL, name=message.recipient),
            )

        function_args = maybe_get_function_args(message, tool_name=self.name)
        if function_args is None:
            yield make_error_message("Invalid function arguments")
            return

        _, function_name = message.recipient.split(".")
        if function_name not in ["search", "open", "find"]:
            yield make_error_message(f"Unknown function: {function_name}")
            return
        try:
            if function_name == "search":
                async for msg in self.search(**function_args):
                    yield msg
            elif function_name == "open":
                async for msg in self.open(**function_args):
                    yield msg
            elif function_name == "find":
                async for msg in self.find(**function_args):
                    yield msg
        except TypeError as e:
            error_text = f"Error: Invalid arguments for function '{function_name}'. Please check the function signature. Details: {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
        except Exception as e:
            error_text = f"An unexpected error occurred while executing function '{function_name}': {e}"
            error_content = TextContent(text=error_text)
            yield self.make_response(
                content=error_content,
                author=Author(role=Role.TOOL, name=message.recipient)
            )
    
    async def _open_url(self, url: str, direct_url_open: bool):
        """Use the cache, if available."""
        backend = self.backend
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            async with ClientSession() as session:
                page = await backend.fetch(url, session=session)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            raise BackendError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

class LocalServiceBrowserBackend:
    source = "web"
    
    def __init__(self, base_url, blocked_substrings: Sequence[str] | None = None):
        self.base_url = base_url
        self.blocked_substrings = list(blocked_substrings or [])
        
    async def _post(self, session: ClientSession, endpoint: str, payload: dict) -> dict:
        t0 = time.time()
        async with session.post(f"{self.base_url}{endpoint}", json=payload) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            return await resp.json()
                
    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        data = await self._post(
            session,
            "/search",
            {"query": query, "topn": topn},
        )
        results = data.get("results", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result['title'], quote=True),
                html.escape(result['url'], quote=True),
                html.escape(result['summary'], quote=True)
            ))
        title_url_summary = filter_blocked_search_results(
            title_url_summary,
            self.blocked_substrings,
        )
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 5,
        session = None,
    ):
        """Search for one or more queries. If query is a list, searches are executed in parallel."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary_all])}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        )
        
    async def fetch(self, url:str, session=None):
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX) :]
        
        data = await self._post(
            session,
            "/get_content",
            {"url": url},
        )
        
        if not data or not data.get("content"):
            raise BackendError(f"No content returned for {url}")
        return process_html(
            html=data.get("content", ""),
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=session,
        )

class SerperServiceBrowserBackend:
    """Browser backend using Serper API for search and scraping."""
    source = "web"

    def __init__(self, blocked_substrings: Sequence[str] | None = None):
        self.api_key = os.getenv("SERPER_API_KEY")
        self.search_url = "https://google.serper.dev/search"
        self.scrape_url = "https://scrape.serper.dev/"
        self.blocked_substrings = list(blocked_substrings or [])

    def _build_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise BackendError("SERPER_API_KEY not set.")
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    async def _search_single(
        self,
        query: str,
        topn: int,
        session: ClientSession,
    ) -> tuple[str, list]:
        """Execute a single search query and return query with results."""
        payload = {
            "q": query,
            "num": topn
        }
        headers = self._build_headers()

        async with session.post(self.search_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Search error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        results = data.get("organic", [])
        if not results:
            logger.warning(f"No results returned for query: '{query}'")
            return query, []

        title_url_summary = []
        for result in results:
            title_url_summary.append((
                html.escape(result.get('title', ''), quote=True),
                html.escape(result.get('link', ''), quote=True),
                html.escape(result.get('snippet', ''), quote=True)
            ))
        title_url_summary = filter_blocked_search_results(
            title_url_summary,
            self.blocked_substrings,
        )
        return query, title_url_summary

    async def search(
        self,
        query: Union[str, List[str]],
        topn: int = 5,
        session = None,
    ):
        """Search using Serper API. Supports single query or list of queries (parallel)."""
        # Handle single query
        if isinstance(query, str):
            query_list = [query]
            title_str = query
        else:
            query_list = query
            # Create a title with all query names
            title_str = " | ".join(query_list)

        # Execute searches in parallel
        tasks = [self._search_single(q, topn, session) for q in query_list]
        all_results = await asyncio.gather(*tasks)

        # Merge all results
        title_url_summary_all = []
        for query_str, title_url_summary in all_results:
            # Add results from each query
            if title_url_summary:
                title_url_summary_all.extend(title_url_summary)

        # If no results from any query, raise error
        if not title_url_summary_all:
            raise BackendError(f"No results returned for any query: {query_list}")

        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_url_summary_all])}
</ul>
</body></html>
"""

        pseudo_url = f"web-search://ts={int(time.time())}"
        return process_html(
            html=html_page,
            url=pseudo_url,
            title=title_str,
            display_urls=True,
            session=session,
        )

    async def fetch(self, url: str, session=None):
        """Fetch and scrape a URL using Serper API."""
        is_view_source = url.startswith(VIEW_SOURCE_PREFIX)
        if is_view_source:
            url = url[len(VIEW_SOURCE_PREFIX):]

        payload = {
            "url": url
        }
        headers = self._build_headers()

        async with session.post(self.scrape_url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                raise BackendError(
                    f"Fetch error {resp.status}: {await resp.text()}"
                )
            data = await resp.json()

        # Sanitize the response data to remove any None keys
        data = sanitize_dict_keys(data)

        if not data:
            raise BackendError(f"No content returned for {url}")

        # Serper scrape API returns 'text' (extracted content) and 'metadata' (with title)
        text_content = data.get("text", "")
        metadata = data.get("metadata", {})
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""

        if not text_content:
            raise BackendError(f"No content returned for {url}")

        return process_html(
            html=text_content,
            url=url,
            title=title,
            display_urls=True,
            session=session,
        )
