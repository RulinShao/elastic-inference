"""
Minimal Miro-style tool backends for Elastic Serving.

This module keeps the runtime self-contained and mirrors the tool names used by
MiroFlow's single-agent configs:

  - search_and_scrape_webpage.google_search
  - jina_scrape_llm_summary.scrape_and_extract_info
  - tool-python.create_sandbox
  - tool-python.run_python_code

The public API is intentionally small:

  - ``build_miro_tool_definitions(enable_python=...)``
  - ``execute_miro_tool(...)``
  - ``format_tool_result_for_llm(...)``
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import tempfile
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx


TOOL_RESULT_MAX_LENGTH = 100_000
DEFAULT_E2B_TEMPLATE_ID = "1av7fdjfvcparqo8efq6"
DEFAULT_E2B_TIMEOUT = 600
MAX_RESULT_LEN = 20_000
MAX_ERROR_LEN = 4_000
INVALID_SANDBOX_IDS = {
    "default",
    "sandbox1",
    "sandbox",
    "some_id",
    "new_sandbox",
    "python",
    "create_sandbox",
    "sandbox123",
    "temp",
    "sandbox-0",
    "sandbox-1",
    "sandbox_0",
    "sandbox_1",
    "new",
    "0",
    "auto",
    "default_sandbox",
    "none",
    "sandbox_12345",
    "dummy",
    "sandbox_01",
}

SERPER_BASE_URL = os.environ.get("SERPER_BASE_URL", "https://google.serper.dev")
JINA_BASE_URL = os.environ.get("JINA_BASE_URL", "https://r.jina.ai")
TMPFILES_DIR = os.path.join(
    os.environ.get("LOGS_DIR", os.path.join(tempfile.gettempdir(), "elastic_miro_logs")),
    "tmpfiles",
)


def looks_like_dir(path: str) -> bool:
    if os.path.isdir(path):
        return True
    if path.endswith(os.path.sep) or not os.path.splitext(path)[1]:
        return True
    return False


def truncate_result(result: str) -> str:
    if len(result) > MAX_RESULT_LEN:
        result = result[:MAX_RESULT_LEN] + " [Result truncated due to length limit]"
    return result


def build_miro_tool_definitions(
    enable_python: bool = False,
    enable_web_summary_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Return Miro-style MCP server definitions used in the system prompt."""
    servers: List[Dict[str, Any]] = [
        {
            "name": "search_and_scrape_webpage",
            "tools": [
                {
                    "name": "google_search",
                    "description": (
                        "Tool to perform web searches via Serper API and retrieve rich results.\n\n"
                        "It is able to retrieve organic search results, people also ask,\n"
                        "related searches, and knowledge graph."
                    ),
                    "schema": {
                        "type": "object",
                        "properties": {
                            "q": {
                                "type": "string",
                                "description": "Search query string",
                            },
                            "gl": {
                                "type": "string",
                                "description": (
                                    "Optional region code for search results in ISO "
                                    "3166-1 alpha-2 format (e.g., 'us')"
                                ),
                            },
                            "hl": {
                                "type": "string",
                                "description": (
                                    "Optional language code for search results in ISO "
                                    "639-1 format (e.g., 'en')"
                                ),
                            },
                            "location": {
                                "type": "string",
                                "description": (
                                    "Optional location for search results (e.g., "
                                    "'SoHo, New York, United States', 'California, "
                                    "United States')"
                                ),
                            },
                            "num": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                            },
                            "tbs": {
                                "type": "string",
                                "description": (
                                    "Time-based search filter ('qdr:h' for past hour, "
                                    "'qdr:d' for past day, 'qdr:w' for past week, "
                                    "'qdr:m' for past month, 'qdr:y' for past year)"
                                ),
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number of results to return (default: 1)",
                            },
                            "autocorrect": {
                                "type": "boolean",
                                "description": "Whether to autocorrect spelling in query",
                            },
                        },
                        "required": ["q"],
                    },
                }
            ],
        },
        {
            "name": "jina_scrape_llm_summary",
            "tools": [
                {
                    "name": "scrape_and_extract_info",
                    "description": (
                        "Scrape content from a URL, including web pages, PDFs, code files, "
                        "and other supported resources, and extract meaningful information "
                        "using an LLM.\nIf you need to extract information from a PDF, "
                        "please use this tool."
                    ),
                    "schema": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": (
                                    "The URL to scrape content from. Supports various "
                                    "types of URLs such as web pages, PDFs, raw text/code "
                                    "files (e.g., GitHub, Gist), and similar sources."
                                ),
                            },
                            "info_to_extract": {
                                "type": "string",
                                "description": (
                                    "The specific types of information to extract "
                                    "(usually a question)"
                                ),
                            },
                            "custom_headers": {
                                "type": "object",
                                "description": (
                                    "Additional headers to include in the scraping request"
                                ),
                            },
                        },
                        "required": ["url", "info_to_extract"],
                    },
                }
            ],
        },
    ]

    if enable_python:
        servers.append(
            {
                "name": "tool-python",
                "tools": [
                    {
                        "name": "create_sandbox",
                        "description": (
                            "Create a linux sandbox."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "timeout": {
                                    "type": "integer",
                                    "description": (
                                        "Time in seconds before the sandbox is "
                                        "automatically shutdown. The default is 600 seconds."
                                    ),
                                }
                            },
                        },
                    },
                    {
                        "name": "run_command",
                        "description": (
                            "Execute a lightweight shell command in the linux sandbox "
                            "(no long-running, blocking, or resource-heavy processes)."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "The command to execute.",
                                },
                                "sandbox_id": {
                                    "type": "string",
                                    "description": (
                                        "The id of the sandbox to execute the command in. "
                                        "To create a new sandbox, use tool "
                                        "`create_sandbox`."
                                    ),
                                },
                            },
                            "required": ["command", "sandbox_id"],
                        },
                    },
                    {
                        "name": "run_python_code",
                        "description": (
                            "Run short, safe python code in a sandbox and return the "
                            "execution result (avoid long loops or heavy tasks; must "
                            "finish quickly)."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "code_block": {
                                    "type": "string",
                                    "description": "The python code to run.",
                                },
                                "sandbox_id": {
                                    "type": "string",
                                    "description": (
                                        "The id of the sandbox to run the code in. Reuse "
                                        "existing sandboxes whenever possible. To create a "
                                        "new sandbox, use tool `create_sandbox`."
                                    ),
                                }
                            },
                            "required": ["code_block", "sandbox_id"],
                        },
                    },
                    {
                        "name": "upload_file_from_local_to_sandbox",
                        "description": (
                            "Upload a local file to the `/home/user` dir of the remote "
                            "python interpreter."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "sandbox_id": {
                                    "type": "string",
                                    "description": (
                                        "The id of the sandbox to run the code in. Reuse "
                                        "existing sandboxes whenever possible. To create a "
                                        "new sandbox, use tool `create_sandbox`."
                                    ),
                                },
                                "local_file_path": {
                                    "type": "string",
                                    "description": "The path of the file on local machine to upload.",
                                },
                                "sandbox_file_path": {
                                    "type": "string",
                                    "description": (
                                        "The path of directory to upload the file to in "
                                        "the sandbox. Default is `/home/user/`."
                                    ),
                                },
                            },
                            "required": ["sandbox_id", "local_file_path"],
                        },
                    },
                    {
                        "name": "download_file_from_internet_to_sandbox",
                        "description": (
                            "Download a file from the internet to the `/home/user` dir "
                            "of the sandbox (avoid large or slow URLs)."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "sandbox_id": {
                                    "type": "string",
                                    "description": (
                                        "The id of the sandbox to run the code in. Reuse "
                                        "existing sandboxes whenever possible. To create a "
                                        "new sandbox, use tool `create_sandbox`."
                                    ),
                                },
                                "url": {
                                    "type": "string",
                                    "description": "The URL of the file to download.",
                                },
                                "sandbox_file_path": {
                                    "type": "string",
                                    "description": (
                                        "The path of directory to download the file to in "
                                        "the sandbox. Default is `/home/user/`."
                                    ),
                                },
                            },
                            "required": ["sandbox_id", "url"],
                        },
                    },
                    {
                        "name": "download_file_from_sandbox_to_local",
                        "description": (
                            "Download a file from the sandbox to local system. Files in "
                            "sandbox cannot be processed by tools from other servers - "
                            "only local files and internet URLs can be processed by them."
                        ),
                        "schema": {
                            "type": "object",
                            "properties": {
                                "sandbox_id": {
                                    "type": "string",
                                    "description": (
                                        "The id of the sandbox to download the file from. "
                                        "To have a sandbox, use tool `create_sandbox`."
                                    ),
                                },
                                "sandbox_file_path": {
                                    "type": "string",
                                    "description": "The path of the file to download on the sandbox.",
                                },
                                "local_filename": {
                                    "type": "string",
                                    "description": (
                                        "Optional filename to save as. If not provided, "
                                        "uses the original filename from sandbox_file_path."
                                    ),
                                },
                            },
                            "required": ["sandbox_id", "sandbox_file_path"],
                        },
                    },
                ],
            }
        )

    return servers


def format_tool_result_for_llm(tool_call_execution_result: Dict[str, Any]) -> str:
    """Match MiroFlow's tool-result formatting for the next user message."""
    server_name = tool_call_execution_result["server_name"]
    tool_name = tool_call_execution_result["tool_name"]

    if "error" in tool_call_execution_result:
        return (
            f"Tool call to {tool_name} on {server_name} failed. "
            f"Error: {tool_call_execution_result['error']}"
        )

    content = tool_call_execution_result.get("result")
    if content is None:
        return (
            f"Tool call to {tool_name} on {server_name} completed, but produced "
            f"no specific output or result."
        )

    if len(content) > TOOL_RESULT_MAX_LENGTH:
        content = content[:TOOL_RESULT_MAX_LENGTH] + "\n... [Result truncated]"
    return content


def _is_banned_url(url: str) -> bool:
    if not url:
        return False
    banned_list = [
        "unifuncs",
        "huggingface.co/datasets",
        "huggingface.co/spaces",
    ]
    return any(banned in url for banned in banned_list)


async def _make_serper_request(
    http_client: httpx.AsyncClient,
    payload: Dict[str, Any],
    headers: Dict[str, str],
) -> httpx.Response:
    last_error: Optional[Exception] = None
    for _ in range(3):
        try:
            resp = await http_client.post(
                f"{SERPER_BASE_URL}/search",
                json=payload,
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_error = exc
            await asyncio.sleep(2)
    assert last_error is not None
    raise last_error


async def google_search(
    http_client: httpx.AsyncClient,
    *,
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: Optional[str] = None,
    num: Optional[int] = None,
    tbs: Optional[str] = None,
    page: Optional[int] = None,
    autocorrect: Optional[bool] = None,
    blocked_domains: Optional[List[str]] = None,
) -> str:
    """Miro-style Serper search returning a JSON string."""
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return json.dumps(
            {
                "success": False,
                "error": "SERPER_API_KEY environment variable not set",
                "results": [],
            },
            ensure_ascii=False,
        )

    if not q or not q.strip():
        return json.dumps(
            {
                "success": False,
                "error": "Search query 'q' is required and cannot be empty",
                "results": [],
            },
            ensure_ascii=False,
        )

    async def perform_search(search_query: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "q": search_query.strip(),
            "gl": gl,
            "hl": hl,
            "num": num if num is not None else 10,
        }
        if location:
            payload["location"] = location
        if tbs:
            payload["tbs"] = tbs
        if page is not None:
            payload["page"] = page
        if autocorrect is not None:
            payload["autocorrect"] = autocorrect

        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        response = await _make_serper_request(http_client, payload, headers)
        data = response.json()
        organic_results = []
        for item in data.get("organic", []):
            link = item.get("link", "")
            if _is_banned_url(link):
                continue
            if blocked_domains and any(d in link for d in blocked_domains):
                continue
            organic_results.append(item)
        return {
            "organic": organic_results,
            "searchParameters": data.get("searchParameters", {}),
        }

    try:
        result = await perform_search(q.strip())
        if not result["organic"] and '"' in q:
            query_without_quotes = q.replace('"', "").strip()
            if query_without_quotes:
                result = await perform_search(query_without_quotes)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {
                "success": False,
                "error": f"Unexpected error: {exc}",
                "results": [],
            },
            ensure_ascii=False,
        )


async def _scrape_url_with_jina(
    http_client: httpx.AsyncClient,
    url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    max_chars: int = 102400 * 4,
) -> Dict[str, Any]:
    if not url or not url.strip():
        return {
            "success": False,
            "content": "",
            "error": "URL cannot be empty",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        return {
            "success": False,
            "content": "",
            "error": "JINA_API_KEY environment variable is not set",
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }

    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    headers = {"Authorization": f"Bearer {api_key}"}
    if custom_headers:
        headers.update(custom_headers)

    last_error: Optional[Exception] = None
    for delay in (1, 2, 4, 8):
        try:
            response = await http_client.get(
                f"{JINA_BASE_URL}/{url}",
                headers=headers,
                timeout=httpx.Timeout(None, connect=20, read=60),
                follow_redirects=True,
            )
            response.raise_for_status()
            content = response.text[:max_chars]
            lines = content.splitlines()
            last_char_line = len(lines)
            return {
                "success": True,
                "content": content,
                "error": "",
                "line_count": len(lines),
                "char_count": len(content),
                "last_char_line": last_char_line,
                "all_content_displayed": len(content) < max_chars,
            }
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.HTTPStatusError,
        ) as exc:
            last_error = exc
            await asyncio.sleep(delay)

    return {
        "success": False,
        "content": "",
        "error": str(last_error) if last_error else "Unknown Jina error",
        "line_count": 0,
        "char_count": 0,
        "last_char_line": 0,
        "all_content_displayed": False,
    }


async def _scrape_url_direct(
    http_client: httpx.AsyncClient,
    url: str,
    custom_headers: Optional[Dict[str, str]] = None,
    max_chars: int = 102400 * 4,
) -> Dict[str, Any]:
    try:
        response = await http_client.get(
            url,
            headers=custom_headers,
            timeout=httpx.Timeout(None, connect=20, read=60),
            follow_redirects=True,
        )
        response.raise_for_status()
        content = response.text[:max_chars]
        # Strip the worst HTML noise without pulling in a parser dependency.
        content = re.sub(r"(?is)<script.*?>.*?</script>", "", content)
        content = re.sub(r"(?is)<style.*?>.*?</style>", "", content)
        content = re.sub(r"(?is)<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        lines = [content[i : i + 200] for i in range(0, len(content), 200)]
        return {
            "success": True,
            "content": content,
            "error": "",
            "line_count": len(lines),
            "char_count": len(content),
            "last_char_line": len(lines),
            "all_content_displayed": len(content) < max_chars,
        }
    except Exception as exc:
        return {
            "success": False,
            "content": "",
            "error": str(exc),
            "line_count": 0,
            "char_count": 0,
            "last_char_line": 0,
            "all_content_displayed": False,
        }


async def _extract_info_with_llm(
    http_client: httpx.AsyncClient,
    *,
    url: str,
    content: str,
    info_to_extract: str,
) -> Dict[str, Any]:
    base_url = os.environ.get("SUMMARY_LLM_BASE_URL")
    model_name = os.environ.get("SUMMARY_LLM_MODEL_NAME")
    api_key = os.environ.get("SUMMARY_LLM_API_KEY")

    if not base_url or not model_name or not api_key:
        return {
            "success": False,
            "extracted_info": "",
            "error": (
                "SUMMARY_LLM_BASE_URL, SUMMARY_LLM_MODEL_NAME, and "
                "SUMMARY_LLM_API_KEY must be set"
            ),
            "model_used": model_name or "",
            "tokens_used": 0,
        }

    prompt = (
        "You are extracting information from a scraped document.\n\n"
        f"Source URL: {url}\n"
        f"Question / information to extract: {info_to_extract}\n\n"
        "Return only the information relevant to the request. If the answer "
        "cannot be determined from the document, say so clearly.\n\n"
        "Document content:\n"
        f"{content[:120000]}"
    )

    try:
        response = await http_client.post(
            base_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 4096,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        extracted_info = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        usage = payload.get("usage", {}) or {}
        return {
            "success": True,
            "extracted_info": extracted_info,
            "error": "",
            "model_used": model_name,
            "tokens_used": usage.get("total_tokens", 0),
        }
    except Exception as exc:
        return {
            "success": False,
            "extracted_info": "",
            "error": str(exc),
            "model_used": model_name,
            "tokens_used": 0,
        }


async def scrape_and_extract_info(
    http_client: httpx.AsyncClient,
    *,
    url: str,
    info_to_extract: str,
    custom_headers: Optional[Dict[str, str]] = None,
    use_web_summary_llm: bool = True,
) -> str:
    if _is_banned_url(url):
        return json.dumps(
            {
                "success": False,
                "url": url,
                "extracted_info": "",
                "error": "Refusing to scrape a banned URL.",
                "scrape_stats": {},
                "tokens_used": 0,
            },
            ensure_ascii=False,
        )

    scrape_result = await _scrape_url_with_jina(http_client, url, custom_headers)
    if not scrape_result["success"]:
        scrape_result = await _scrape_url_direct(http_client, url, custom_headers)
        if not scrape_result["success"]:
            return json.dumps(
                {
                    "success": False,
                    "url": url,
                    "extracted_info": "",
                    "error": f"Scraping failed: {scrape_result['error']}",
                    "scrape_stats": {},
                    "tokens_used": 0,
                },
                ensure_ascii=False,
            )

    if not use_web_summary_llm:
        return json.dumps(
            {
                "success": True,
                "url": url,
                "extracted_info": scrape_result["content"],
                "error": "",
                "scrape_stats": {
                    "line_count": scrape_result["line_count"],
                    "char_count": scrape_result["char_count"],
                    "last_char_line": scrape_result["last_char_line"],
                    "all_content_displayed": scrape_result["all_content_displayed"],
                },
                "model_used": "",
                "tokens_used": 0,
                "summary_llm_used": False,
                "info_to_extract": info_to_extract,
            },
            ensure_ascii=False,
        )

    extracted_result = await _extract_info_with_llm(
        http_client,
        url=url,
        content=scrape_result["content"],
        info_to_extract=info_to_extract,
    )
    return json.dumps(
        {
            "success": extracted_result["success"],
            "url": url,
            "extracted_info": extracted_result["extracted_info"],
            "error": extracted_result["error"],
            "scrape_stats": {
                "line_count": scrape_result["line_count"],
                "char_count": scrape_result["char_count"],
                "last_char_line": scrape_result["last_char_line"],
                "all_content_displayed": scrape_result["all_content_displayed"],
            },
            "model_used": extracted_result["model_used"],
            "tokens_used": extracted_result["tokens_used"],
            "summary_llm_used": True,
            "info_to_extract": info_to_extract,
        },
        ensure_ascii=False,
    )


def _create_e2b_sandbox(timeout: int = DEFAULT_E2B_TIMEOUT) -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    timeout = min(int(timeout), DEFAULT_E2B_TIMEOUT)
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        sandbox = None
        try:
            sandbox = Sandbox.create(
                timeout=timeout,
                api_key=api_key,
                template=DEFAULT_E2B_TEMPLATE_ID,
            )
            sandbox_id = getattr(sandbox, "sandbox_id", None)
            if not sandbox_id:
                info = sandbox.get_info()
                sandbox_id = getattr(info, "sandbox_id", "")
            if not sandbox_id:
                raise RuntimeError("missing sandbox_id")
            return f"Sandbox created with sandbox_id: {sandbox_id}"
        except Exception as exc:
            if attempt == max_retries:
                error_details = str(exc)[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to create sandbox after {max_retries} attempts: "
                    f"{error_details}, please retry later."
                )
            time_to_sleep = attempt**2
            import time

            time.sleep(time_to_sleep)
        finally:
            try:
                if sandbox is not None:
                    sandbox.set_timeout(timeout)
            except Exception:
                pass


def _run_e2b_python(code_block: str, sandbox_id: str = "default") -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    if not sandbox_id or sandbox_id in INVALID_SANDBOX_IDS:
        try:
            sandbox = Sandbox.create(
                timeout=DEFAULT_E2B_TIMEOUT,
                api_key=api_key,
                template=DEFAULT_E2B_TEMPLATE_ID,
            )
            try:
                execution = sandbox.run_code(code_block)
                return truncate_result(str(execution))
            finally:
                sandbox.kill()
        except Exception as exc:
            error_details = str(exc)[:MAX_ERROR_LEN]
            return (
                "[ERROR]: Failed to run code in stateless mode. "
                f"Exception type: {type(exc).__name__}, Details: {error_details}"
            )

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
    except Exception:
        return (
            f"[ERROR]: Failed to connect to sandbox {sandbox_id}. "
            "Make sure the sandbox is created and the sandbox_id is correct."
        )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
            execution = sandbox.run_code(code_block)
            return truncate_result(str(execution))
        except Exception as exc:
            if attempt == max_retries:
                error_details = str(exc)[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to run code in sandbox {sandbox_id} after "
                    f"{max_retries} attempts. Exception type: {type(exc).__name__}, "
                    f"Details: {error_details}"
                )
            import time

            time.sleep(attempt**2)
        finally:
            try:
                sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
            except Exception:
                pass


def _run_e2b_command(command: str, sandbox_id: str) -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    if sandbox_id in INVALID_SANDBOX_IDS:
        return (
            f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a "
            "real sandbox first using the create_sandbox tool."
        )

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
    except Exception:
        return (
            f"[ERROR]: Failed to connect to sandbox {sandbox_id}. "
            "Make sure the sandbox is created and the sandbox_id is correct."
        )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
            result = sandbox.commands.run(command)
            return truncate_result(str(result))
        except Exception as exc:
            if attempt == max_retries:
                error_details = str(exc)[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to run command after {max_retries} attempts.\n\n"
                    f"Exception type: {type(exc).__name__}\nDetails: {error_details}"
                )
            import time

            time.sleep(attempt**2)
        finally:
            try:
                sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
            except Exception:
                pass


def _upload_file_from_local_to_sandbox(
    sandbox_id: str,
    local_file_path: str,
    sandbox_file_path: str = "/home/user",
) -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    if sandbox_id in INVALID_SANDBOX_IDS:
        return (
            f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a "
            "real sandbox first using the create_sandbox tool."
        )

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
    except Exception:
        return (
            f"[ERROR]: Failed to connect to sandbox {sandbox_id}. "
            "Make sure the sandbox is created and the sandbox_id is correct."
        )

    try:
        sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
        if not os.path.exists(local_file_path):
            return f"[ERROR]: Local file does not exist: {local_file_path}"
        if not os.path.isfile(local_file_path):
            return f"[ERROR]: Path is not a file: {local_file_path}"

        uploaded_file_path = os.path.normpath(
            os.path.join(sandbox_file_path, os.path.basename(local_file_path))
        )
        parent_dir = os.path.dirname(uploaded_file_path)
        if parent_dir and parent_dir != "/":
            mkdir_result = sandbox.commands.run(f"mkdir -p {shlex.quote(parent_dir)}")
            if getattr(mkdir_result, "exit_code", 0) != 0:
                mkdir_result_str = str(mkdir_result)[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to create directory {parent_dir} in sandbox "
                    f"{sandbox_id}: {mkdir_result_str}"
                )

        with open(local_file_path, "rb") as f:
            sandbox.files.write(uploaded_file_path, f)
        return f"File uploaded to {uploaded_file_path}"
    except Exception as exc:
        error_details = str(exc)[:MAX_ERROR_LEN]
        return (
            f"[ERROR]: Failed to upload file {local_file_path} to sandbox "
            f"{sandbox_id}: {error_details}"
        )
    finally:
        try:
            sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
        except Exception:
            pass


def _download_file_from_internet_to_sandbox(
    sandbox_id: str,
    url: str,
    sandbox_file_path: str = "/home/user",
) -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    if sandbox_id in INVALID_SANDBOX_IDS:
        return (
            f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a "
            "real sandbox first using the create_sandbox tool."
        )

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
    except Exception:
        return (
            f"[ERROR]: Failed to connect to sandbox {sandbox_id}. "
            "Make sure the sandbox is created and the sandbox_id is correct."
        )

    try:
        sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)

        parsed_url = urlparse(url)
        basename = os.path.basename(parsed_url.path) or "downloaded_file"
        if "?" in basename:
            basename = basename.split("?")[0]
        if "#" in basename:
            basename = basename.split("#")[0]

        if looks_like_dir(sandbox_file_path):
            downloaded_file_path = os.path.join(sandbox_file_path, basename)
        else:
            downloaded_file_path = sandbox_file_path
        downloaded_file_path = os.path.normpath(downloaded_file_path)

        parent_dir = os.path.dirname(downloaded_file_path)
        if parent_dir and parent_dir != "/":
            mkdir_result = sandbox.commands.run(f"mkdir -p {shlex.quote(parent_dir)}")
            if getattr(mkdir_result, "exit_code", 0) != 0:
                mkdir_result_str = str(mkdir_result)[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to create directory {parent_dir} in sandbox "
                    f"{sandbox_id}: {mkdir_result_str}"
                )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            safe_url = shlex.quote(url)
            safe_path = shlex.quote(downloaded_file_path)
            cmd = f"wget {safe_url} -O {safe_path}"
            try:
                result = sandbox.commands.run(cmd)
                if getattr(result, "exit_code", 0) == 0:
                    return f"File downloaded to {safe_path}"
                if attempt < max_retries:
                    import time

                    time.sleep(4**attempt)
                    continue
                error_details = ""
                if hasattr(result, "stderr") and result.stderr:
                    error_details = f"stderr: {result.stderr}"[:MAX_ERROR_LEN]
                return (
                    f"[ERROR]: Failed to download file from {url} to {downloaded_file_path} "
                    f"after {max_retries} attempts.\n\nexit_code: {result.exit_code}\n\n"
                    f"Details: {error_details}"
                )
            except Exception as exc:
                if attempt == max_retries:
                    error_details = str(exc)[:MAX_ERROR_LEN]
                    return (
                        f"[ERROR]: Failed to download file from {url} to "
                        f"{downloaded_file_path}. Exception: {error_details}"
                    )
                import time

                time.sleep(4**attempt)
    except Exception as exc:
        error_details = str(exc)[:MAX_ERROR_LEN]
        return f"[ERROR]: Failed to download file from {url}: {error_details}"
    finally:
        try:
            sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
        except Exception:
            pass


def _download_file_from_sandbox_to_local(
    sandbox_id: str,
    sandbox_file_path: str,
    local_filename: Optional[str] = None,
) -> str:
    from e2b_code_interpreter import Sandbox

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        raise RuntimeError("E2B_API_KEY is not set")

    if sandbox_id in INVALID_SANDBOX_IDS:
        return (
            f"[ERROR]: '{sandbox_id}' is not a valid sandbox_id. Please create a "
            "real sandbox first using the create_sandbox tool."
        )

    try:
        sandbox = Sandbox.connect(sandbox_id, api_key=api_key)
    except Exception:
        return (
            f"[ERROR]: Failed to connect to sandbox {sandbox_id}. "
            "Make sure the sandbox is created and the sandbox_id is correct."
        )

    try:
        sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
        os.makedirs(TMPFILES_DIR, exist_ok=True)

        check_result = sandbox.commands.run(
            f'test -d {shlex.quote(sandbox_file_path)} && echo "is_directory" || echo "not_directory"'
        )
        if getattr(check_result, "stdout", "") and "is_directory" in check_result.stdout:
            return (
                f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox {sandbox_id}: "
                "path is a directory, not a file."
            )

        check_file_result = sandbox.commands.run(
            f'test -f {shlex.quote(sandbox_file_path)} && echo "exists" || echo "not_exists"'
        )
        if getattr(check_file_result, "stdout", "") and "not_exists" in check_file_result.stdout:
            check_any_result = sandbox.commands.run(
                f'test -e {shlex.quote(sandbox_file_path)} && echo "exists" || echo "not_exists"'
            )
            if getattr(check_any_result, "stdout", "") and "not_exists" in check_any_result.stdout:
                return (
                    f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox "
                    f"{sandbox_id}: file does not exist."
                )

        if local_filename is None or not str(local_filename).strip():
            local_filename = os.path.basename(sandbox_file_path)
            if not local_filename or local_filename == "/":
                local_filename = "downloaded_file"

        local_file_path = os.path.join(
            TMPFILES_DIR, f"sandbox_{sandbox_id}_{local_filename}"
        )

        try:
            with open(local_file_path, "wb") as f:
                content = sandbox.files.read(sandbox_file_path, format="bytes")
                f.write(content)
        except Exception as read_error:
            error_msg = str(read_error).lower()
            if "directory" in error_msg or "is a directory" in error_msg:
                return (
                    f"[ERROR]: Cannot download '{sandbox_file_path}' from sandbox "
                    f"{sandbox_id}: path is a directory, not a file."
                )
            read_error_details = str(read_error)[:MAX_ERROR_LEN]
            return (
                f"[ERROR]: Failed to read file '{sandbox_file_path}' from sandbox "
                f"{sandbox_id}: {read_error_details}"
            )

        return f"File downloaded successfully to: {local_file_path}"
    except Exception as exc:
        error_details = str(exc)[:MAX_ERROR_LEN]
        return (
            f"[ERROR]: Failed to download file '{sandbox_file_path}' from sandbox "
            f"{sandbox_id}: {error_details}"
        )
    finally:
        try:
            sandbox.set_timeout(DEFAULT_E2B_TIMEOUT)
        except Exception:
            pass


async def create_sandbox(timeout: int = DEFAULT_E2B_TIMEOUT) -> str:
    return await asyncio.to_thread(_create_e2b_sandbox, timeout)


async def run_command(command: str, sandbox_id: str) -> str:
    return await asyncio.to_thread(_run_e2b_command, command, sandbox_id)


async def run_python_code(code_block: str, sandbox_id: str = "default") -> str:
    return await asyncio.to_thread(_run_e2b_python, code_block, sandbox_id)


async def upload_file_from_local_to_sandbox(
    sandbox_id: str,
    local_file_path: str,
    sandbox_file_path: str = "/home/user",
) -> str:
    return await asyncio.to_thread(
        _upload_file_from_local_to_sandbox,
        sandbox_id,
        local_file_path,
        sandbox_file_path,
    )


async def download_file_from_internet_to_sandbox(
    sandbox_id: str,
    url: str,
    sandbox_file_path: str = "/home/user",
) -> str:
    return await asyncio.to_thread(
        _download_file_from_internet_to_sandbox,
        sandbox_id,
        url,
        sandbox_file_path,
    )


async def download_file_from_sandbox_to_local(
    sandbox_id: str,
    sandbox_file_path: str,
    local_filename: Optional[str] = None,
) -> str:
    return await asyncio.to_thread(
        _download_file_from_sandbox_to_local,
        sandbox_id,
        sandbox_file_path,
        local_filename,
    )


async def execute_miro_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    http_client: httpx.AsyncClient,
    blocked_domains: Optional[List[str]] = None,
    enable_python: bool = False,
    use_web_summary_llm: bool = False,
) -> Dict[str, Any]:
    """Execute a single Miro-style tool call and return a result dict."""
    try:
        server_name = (server_name or "").strip()
        tool_name = (tool_name or "").strip()
        arguments = dict(arguments or {})

        if tool_name == "scrape_and_extract_info" and "info_to_extract" not in arguments:
            for wrong_name in ("description", "introduction"):
                if wrong_name in arguments:
                    arguments = dict(arguments)
                    arguments["info_to_extract"] = arguments.pop(wrong_name)
                    break

        if tool_name in ("python", "python_code", "run_python_code"):
            tool_name = "run_python_code"
            server_name = "tool-python"
            if "code_block" not in arguments and "code" in arguments:
                arguments["code_block"] = arguments.pop("code")
            arguments.setdefault("sandbox_id", "default")

        if tool_name in (
            "create_sandbox",
            "run_command",
            "upload_file_from_local_to_sandbox",
            "download_file_from_internet_to_sandbox",
            "download_file_from_sandbox_to_local",
        ):
            server_name = "tool-python"

        if server_name == "search_and_scrape_webpage" or tool_name == "google_search":
            server_name = "search_and_scrape_webpage"
            tool_name = "google_search"
            result = await google_search(
                http_client,
                blocked_domains=blocked_domains,
                **arguments,
            )
        elif server_name == "jina_scrape_llm_summary" or tool_name == "scrape_and_extract_info":
            server_name = "jina_scrape_llm_summary"
            tool_name = "scrape_and_extract_info"
            result = await scrape_and_extract_info(
                http_client,
                use_web_summary_llm=use_web_summary_llm,
                **arguments,
            )
        elif server_name == "tool-python" and tool_name == "create_sandbox":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            timeout = arguments.get("timeout", DEFAULT_E2B_TIMEOUT)
            result = await create_sandbox(timeout=timeout)
        elif server_name == "tool-python" and tool_name == "run_command":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            command = arguments.get("command")
            sandbox_id = str(arguments.get("sandbox_id", ""))
            if not command:
                raise RuntimeError("run_command requires a 'command' argument")
            result = await run_command(command, sandbox_id)
        elif server_name in ("tool-python", "stateless_python") and tool_name == "run_python_code":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            code = arguments.get("code_block")
            if not code:
                raise RuntimeError("Python tool requires a 'code_block' argument")
            sandbox_id = str(arguments.get("sandbox_id", "default"))
            result = await run_python_code(code, sandbox_id=sandbox_id)
        elif server_name == "tool-python" and tool_name == "upload_file_from_local_to_sandbox":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            sandbox_id = str(arguments.get("sandbox_id", ""))
            local_file_path = arguments.get("local_file_path")
            sandbox_file_path = arguments.get("sandbox_file_path", "/home/user")
            if not local_file_path:
                raise RuntimeError(
                    "upload_file_from_local_to_sandbox requires a 'local_file_path' argument"
                )
            result = await upload_file_from_local_to_sandbox(
                sandbox_id=sandbox_id,
                local_file_path=local_file_path,
                sandbox_file_path=sandbox_file_path,
            )
        elif server_name == "tool-python" and tool_name == "download_file_from_internet_to_sandbox":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            sandbox_id = str(arguments.get("sandbox_id", ""))
            url = arguments.get("url")
            sandbox_file_path = arguments.get("sandbox_file_path", "/home/user")
            if not url:
                raise RuntimeError(
                    "download_file_from_internet_to_sandbox requires a 'url' argument"
                )
            result = await download_file_from_internet_to_sandbox(
                sandbox_id=sandbox_id,
                url=url,
                sandbox_file_path=sandbox_file_path,
            )
        elif server_name == "tool-python" and tool_name == "download_file_from_sandbox_to_local":
            if not enable_python:
                raise RuntimeError("Python tool not enabled. Pass --enable-python.")
            sandbox_id = str(arguments.get("sandbox_id", ""))
            sandbox_file_path = arguments.get("sandbox_file_path")
            local_filename = arguments.get("local_filename")
            if not sandbox_file_path:
                raise RuntimeError(
                    "download_file_from_sandbox_to_local requires a 'sandbox_file_path' argument"
                )
            result = await download_file_from_sandbox_to_local(
                sandbox_id=sandbox_id,
                sandbox_file_path=sandbox_file_path,
                local_filename=local_filename,
            )
        else:
            raise RuntimeError(f"Unknown tool: {server_name}.{tool_name}")

        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        }
    except Exception as exc:
        return {
            "server_name": server_name,
            "tool_name": tool_name,
            "arguments": arguments,
            "error": str(exc),
        }
