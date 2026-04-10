"""
Miro-style message-based agent runtime for Elastic Serving.

This runtime mirrors the single-agent MiroFlow loop closely enough for first
pass evaluation work:

  - ``/v1/chat/completions`` instead of raw prompt completions
  - XML ``<use_mcp_tool>`` tool calls inside assistant text
  - message-based ``keep_tool_result`` trimming
  - Miro-style final-summary prompt that extracts a boxed answer
"""

from __future__ import annotations

import asyncio
import copy
import datetime as dt
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from elastic_serving.miro_tools import (
    build_miro_tool_definitions,
    execute_miro_tool,
    format_tool_result_for_llm,
)


FORMAT_ERROR_MESSAGE = "No \\boxed{} content found in the final answer."
MAX_CONSECUTIVE_ROLLBACKS = 5
DEFAULT_MAX_FINAL_ANSWER_RETRIES = 3

mcp_tags = [
    "<use_mcp_tool>",
    "</use_mcp_tool>",
    "<server_name>",
    "</server_name>",
    "<arguments>",
    "</arguments>",
]

refusal_keywords = [
    "time constraint",
    "I’m sorry, but I can’t",
    "I'm sorry, I cannot solve",
]

TOOL_SERVER_MAPPING = {
    "create_sandbox": "tool-python",
    "run_command": "tool-python",
    "run_python_code": "tool-python",
    "upload_file_from_local_to_sandbox": "tool-python",
    "download_file_from_internet_to_sandbox": "tool-python",
    "download_file_from_sandbox_to_local": "tool-python",
    "google_search": "search_and_scrape_webpage",
    "scrape_and_extract_info": "jina_scrape_llm_summary",
}


def generate_mcp_system_prompt(
    today: dt.date,
    mcp_servers: List[Dict[str, Any]],
) -> str:
    formatted_date = today.strftime("%Y-%m-%d")
    template = f"""In this environment you have access to a set of tools you can use to answer the user's question. 

You only have access to the tools provided below. You can only use one tool per message, and will receive the result of that tool in the user's next response. You use tools step-by-step to accomplish a given task, with each tool-use informed by the result of the previous tool-use. Today is: {formatted_date}

# Tool-Use Formatting Instructions 

Tool-use is formatted using XML-style tags. The tool-use is enclosed in <use_mcp_tool></use_mcp_tool> and each parameter is similarly enclosed within its own set of tags.

The Model Context Protocol (MCP) connects to servers that provide additional tools and resources to extend your capabilities. You can use the server's tools via the `use_mcp_tool`.

Description: 
Request to use a tool provided by a MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.

Parameters:
- server_name: (required) The name of the MCP server providing the tool
- tool_name: (required) The name of the tool to execute
- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema, quotes within string must be properly escaped, ensure it's valid JSON

Usage:
<use_mcp_tool>
<server_name>server name here</server_name>
<tool_name>tool name here</tool_name>
<arguments>
{{
"param1": "value1",
"param2": "value2 \\"escaped string\\""
}}
</arguments>
</use_mcp_tool>

Important Notes:
- Tool-use must be placed **at the end** of your response, **top-level**, and not nested within other tags.
- Always adhere to this format for the tool use to ensure proper parsing and execution.

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
"""
    for server in mcp_servers:
        template += f"\n## Server name: {server['name']}\n"
        for tool in server.get("tools", []):
            template += f"### Tool name: {tool['name']}\n"
            template += f"Description: {tool['description']}\n"
            template += f"Input JSON schema: {tool['schema']}\n"
    template += """
# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.
"""
    return template


def generate_agent_specific_system_prompt() -> str:
    return """
# Agent Specific Objective

You are a task-solving agent that uses tools step-by-step to answer the user's question. Your goal is to provide complete, accurate and well-reasoned answers using additional tools.
""".strip()


def generate_agent_summarize_prompt(task_description: str) -> str:
    return (
        "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
        "If a clear answer has already been provided earlier in the conversation, do not rethink or recalculate it — "
        "simply extract that answer and reformat it to match the required format below.\n"
        "If a definitive answer could not be determined, make a well-informed educated guess based on the conversation.\n\n"
        "The original question is repeated here for reference:\n\n"
        f'"{task_description}"\n\n'
        "Wrap your final answer in \\boxed{}.\n"
        "Your final answer should be:\n"
        "- a number, OR\n"
        "- as few words as possible, OR\n"
        "- a comma-separated list of numbers and/or strings.\n\n"
        "ADDITIONALLY, your final answer MUST strictly follow any formatting instructions in the original question — "
        "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n"
        "If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.\n"
        "If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.\n"
        "If you are asked for a comma-separated list, apply the above rules depending on whether the elements are numbers or strings.\n"
        "Do NOT include any punctuation such as '.', '!', or '?' at the end of the answer.\n"
        "Do NOT include any invisible or non-printable characters in the answer output.\n\n"
        "You must absolutely not perform any MCP tool call, tool invocation, search, scrape, code execution, or similar actions.\n"
        "You can only answer the original question based on the information already retrieved and your own internal knowledge.\n"
        "If you attempt to call any tool, it will be considered a mistake."
    )


def safe_json_loads(arguments_str: str) -> Dict[str, Any]:
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json

        repaired = repair_json(arguments_str, ensure_ascii=False)
        return json.loads(repaired)
    except Exception:
        pass

    try:
        fixed = (
            arguments_str.replace("'", '"')
            .replace("None", "null")
            .replace("True", "true")
            .replace("False", "false")
        )
        return json.loads(fixed)
    except Exception:
        return {"error": "Failed to parse arguments", "raw": arguments_str}


def extract_llm_response_text(text: str) -> str:
    match = re.search(r"<use_mcp_tool>", text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def parse_miro_tool_calls(text: str) -> List[Dict[str, Any]]:
    text = repair_mcp_tool_call_text(text)
    tool_calls = []
    tool_call_patterns = re.findall(
        r"<use_mcp_tool>\s*<server_name>(.*?)</server_name>\s*<tool_name>(.*?)</tool_name>\s*<arguments>\s*([\s\S]*?)\s*</arguments>\s*</use_mcp_tool>",
        text,
        re.DOTALL,
    )
    for server_name, tool_name, arguments_str in tool_call_patterns:
        arguments = safe_json_loads(arguments_str.strip())
        if isinstance(arguments, dict):
            arguments = {k: v for k, v in arguments.items() if v is not None}
        tool_calls.append(
            {
                "server_name": server_name.strip(),
                "tool_name": tool_name.strip(),
                "arguments": arguments,
                "id": None,
            }
        )
    return tool_calls


def repair_mcp_tool_call_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    for wrong_name in ("python", "python_code"):
        wrong_tag = f"<tool_name>{wrong_name}</tool_name>"
        if wrong_tag in text:
            text = text.replace(wrong_tag, "<tool_name>run_python_code</tool_name>")

    for tool_name, correct_server in TOOL_SERVER_MAPPING.items():
        tool_tag = f"<tool_name>{tool_name}</tool_name>"
        if tool_tag not in text:
            continue
        correct_server_tag = f"<server_name>{correct_server}</server_name>"
        if correct_server_tag in text:
            continue
        text = re.sub(
            r"<server_name>[^<]+</server_name>(\s*" + re.escape(tool_tag) + r")",
            correct_server_tag + r"\1",
            text,
        )
    return text


def _extract_boxed_content(text: str) -> str:
    boxed_re = re.compile(r"\\boxed\b", re.DOTALL)
    last_result = None
    i = 0
    n = len(text)

    while True:
        match = boxed_re.search(text, i)
        if not match:
            break
        j = match.end()
        while j < n and text[j].isspace():
            j += 1
        if j >= n or text[j] != "{":
            i = j
            continue

        depth = 0
        k = j
        escaped = False
        found_closing = False
        while k < n:
            ch = text[k]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_result = text[j + 1 : k]
                    i = k + 1
                    found_closing = True
                    break
            k += 1

        if not found_closing and depth > 0:
            last_result = text[j + 1 : n]
            i = k
        elif not found_closing:
            i = j + 1

    if not last_result:
        return ""
    if last_result in {"?", "??", "???", "？", "……", "…", "...", "unknown"}:
        return ""
    return last_result.strip()


def build_messages_for_llm(
    message_history: List[Dict[str, Any]],
    keep_tool_result: int,
) -> List[Dict[str, str]]:
    messages_copy = copy.deepcopy(message_history)
    if keep_tool_result == -1:
        return [{"role": m["role"], "content": m["content"]} for m in messages_copy]

    tool_indices = [
        idx for idx, msg in enumerate(messages_copy) if msg.get("kind") == "tool_result"
    ]
    keep = set(tool_indices[-keep_tool_result:]) if keep_tool_result > 0 else set()

    for idx, msg in enumerate(messages_copy):
        if msg.get("kind") == "tool_result" and idx not in keep:
            msg["content"] = "Tool result is omitted to save tokens."

    return [{"role": m["role"], "content": m["content"]} for m in messages_copy]


def _get_query_key(tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    if tool_name == "google_search":
        return f"{tool_name}:{arguments.get('q', '').strip()}"
    if tool_name == "scrape_and_extract_info":
        return (
            f"{tool_name}:{arguments.get('url', '').strip()}:"
            f"{arguments.get('info_to_extract', '').strip()}"
        )
    return None


def _extract_answer_text(text: str) -> str:
    return _extract_boxed_content(text)


def _rollback_last_assistant(
    history: List[Dict[str, Any]],
    conversation: Optional[List[Dict[str, Any]]],
) -> None:
    if history and history[-1].get("role") == "assistant":
        history.pop()
    if conversation and conversation[-1].get("role") == "assistant" and not conversation[-1].get(
        "final_answer"
    ):
        conversation.pop()


def _rollback_last_assistant_user_pair(
    history: List[Dict[str, Any]],
    conversation: Optional[List[Dict[str, Any]]],
) -> None:
    if history and history[-1].get("role") == "user":
        history.pop()
    if history and history[-1].get("role") == "assistant":
        history.pop()

    if conversation:
        while conversation and conversation[-1].get("role") == "tool":
            conversation.pop()
        if conversation and conversation[-1].get("role") == "assistant" and not conversation[-1].get(
            "final_answer"
        ):
            conversation.pop()


def _estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    return max(1, int(len(str(text)) / 4))


def _ensure_summary_context(
    history: List[Dict[str, Any]],
    summary_prompt: str,
    *,
    last_call_tokens: Dict[str, Any],
    max_context_length: int,
    max_tokens: int,
    conversation: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, List[Dict[str, Any]]]:
    last_prompt_tokens = int(last_call_tokens.get("prompt_tokens", 0) or 0)
    last_completion_tokens = int(last_call_tokens.get("completion_tokens", 0) or 0)
    buffer_factor = 1.5

    summary_tokens = int(_estimate_tokens(summary_prompt) * buffer_factor)
    last_user_tokens = 0
    if history and history[-1]["role"] == "user":
        last_user_tokens = int(_estimate_tokens(history[-1]["content"]) * buffer_factor)

    estimated_total = (
        last_prompt_tokens
        + last_completion_tokens
        + last_user_tokens
        + summary_tokens
        + max_tokens
        + 1000
    )

    if estimated_total >= max_context_length:
        _rollback_last_assistant_user_pair(history, conversation)
        return False, history

    return True, history


async def _chat_completion(
    openai_http: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    session_id: str,
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    headers = {
        "Authorization": "Bearer EMPTY",
        "x-upstream-session-id": session_id,
    }
    while True:
        response = await openai_http.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": False,
                "extra_body": (
                    {"repetition_penalty": repetition_penalty}
                    if repetition_penalty != 1.0
                    else {}
                ),
            },
            headers=headers,
            timeout=600,
        )
        if response.status_code == 503:
            await asyncio.sleep(10)
            continue
        response.raise_for_status()
        payload = response.json()
        choice = payload.get("choices", [{}])[0] or {}
        return (
            choice.get("message", {}).get("content", "") or "",
            payload.get("usage", {}) or {},
            choice.get("finish_reason"),
        )


async def generate_miro_trajectory(
    *,
    question: str,
    qid: str,
    base_urls: Any,
    model: str,
    traj_idx: int = 0,
    max_turns: int = 300,
    max_gen_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.05,
    save_conversation: bool = False,
    blocked_domains: Optional[List[str]] = None,
    enable_python: bool = False,
    keep_tool_result: int = -1,
    use_summary: bool = True,
    use_web_summary_llm: bool = True,
    max_context_length: int = 131072,
    context_compress_limit: int = 0,
) -> Dict[str, Any]:
    """Run one Miro-style message-based trajectory."""
    http_client = httpx.AsyncClient(timeout=120)
    openai_http = httpx.AsyncClient(timeout=600)
    tool_defs = build_miro_tool_definitions(
        enable_python=enable_python,
        enable_web_summary_llm=use_web_summary_llm,
    )
    system_prompt = (
        generate_mcp_system_prompt(dt.date.today(), tool_defs)
        + "\n\n"
        + generate_agent_specific_system_prompt()
    )

    history: List[Dict[str, Any]] = [
        {"role": "user", "content": question, "kind": "initial_user"}
    ]
    conversation = [] if save_conversation else None
    tool_calls_log: List[Dict[str, Any]] = []
    used_queries: Dict[str, int] = {}
    rollback_count = 0
    tool_call_count = 0
    turn_count = 0
    session_id = f"{qid}-{traj_idx}-{uuid.uuid4()}"
    final_answer = ""
    final_boxed_answer = ""
    intermediate_boxed_answers: List[str] = []
    last_call_tokens: Dict[str, Any] = {}

    if isinstance(base_urls, list):
        if not hasattr(generate_miro_trajectory, "_rr_counter"):
            generate_miro_trajectory._rr_counter = 0
        idx = generate_miro_trajectory._rr_counter % len(base_urls)
        generate_miro_trajectory._rr_counter += 1
        base_url = base_urls[idx]
    else:
        base_url = base_urls

    tag = f"qid={qid} t={traj_idx}"
    t0 = time.time()

    try:
        while turn_count < max_turns:
            turn_count += 1
            messages = [{"role": "system", "content": system_prompt}] + build_messages_for_llm(
                history, keep_tool_result
            )

            try:
                assistant_response_text, last_call_tokens, _ = await _chat_completion(
                    openai_http,
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    session_id=session_id,
                )
            except Exception as exc:
                print(f"  [{tag}] Error: {exc}")
                break

            assistant_response_text = repair_mcp_tool_call_text(assistant_response_text)
            boxed_content = _extract_boxed_content(assistant_response_text)
            if boxed_content:
                intermediate_boxed_answers.append(boxed_content)
            history.append(
                {
                    "role": "assistant",
                    "content": assistant_response_text,
                    "kind": "assistant",
                }
            )

            visible_text = extract_llm_response_text(assistant_response_text)
            if save_conversation:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": assistant_response_text,
                        "visible_text": visible_text,
                    }
                )

            tool_calls = parse_miro_tool_calls(assistant_response_text)
            if not tool_calls:
                has_mcp_tags = any(tag_text in assistant_response_text for tag_text in mcp_tags)
                matched_refusals = [
                    keyword for keyword in refusal_keywords if keyword in assistant_response_text
                ]
                if has_mcp_tags:
                    if rollback_count < MAX_CONSECUTIVE_ROLLBACKS - 1:
                        _rollback_last_assistant(history, conversation)
                        turn_count -= 1
                        rollback_count += 1
                        print(
                            f"  [{tag}] Rollback malformed MCP response "
                            f"({rollback_count}/{MAX_CONSECUTIVE_ROLLBACKS})"
                        )
                        continue
                    print(
                        f"  [{tag}] End after max malformed MCP rollbacks "
                        f"({rollback_count}/{MAX_CONSECUTIVE_ROLLBACKS})"
                    )
                    break
                if matched_refusals:
                    if rollback_count < MAX_CONSECUTIVE_ROLLBACKS - 1:
                        _rollback_last_assistant(history, conversation)
                        turn_count -= 1
                        rollback_count += 1
                        print(
                            f"  [{tag}] Rollback refusal response "
                            f"({rollback_count}/{MAX_CONSECUTIVE_ROLLBACKS}): "
                            f"{matched_refusals}"
                        )
                        continue
                    print(
                        f"  [{tag}] End after max refusal rollbacks "
                        f"({rollback_count}/{MAX_CONSECUTIVE_ROLLBACKS})"
                    )
                    break
                final_answer = assistant_response_text
                break

            merged_results: List[str] = []
            should_rollback = False

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}

                query_key = _get_query_key(tool_name, arguments)
                if query_key and used_queries.get(query_key, 0) > 0:
                    # Mirror Miro's anti-duplicate bias by rolling back the step.
                    if rollback_count < MAX_CONSECUTIVE_ROLLBACKS - 1:
                        _rollback_last_assistant(history, conversation)
                        turn_count -= 1
                        rollback_count += 1
                        should_rollback = True
                        break
                    merged_results.append(
                        (
                            f"Tool call to {tool_name} on {server_name} failed. "
                            f"Error: duplicate query detected for {query_key}"
                        )
                    )
                    rollback_count = 0
                    continue

                tool_call_count += 1
                short = json.dumps(arguments, ensure_ascii=False)[:80]
                print(
                    f"  [{tag}] Tool {tool_call_count} "
                    f"(turn {turn_count}/{max_turns}): "
                    f"{server_name}.{tool_name}({short})"
                )

                tool_result = await execute_miro_tool(
                    server_name,
                    tool_name,
                    arguments,
                    http_client=http_client,
                    blocked_domains=blocked_domains,
                    enable_python=enable_python,
                    use_web_summary_llm=use_web_summary_llm,
                )
                if query_key and "error" not in tool_result:
                    used_queries[query_key] = used_queries.get(query_key, 0) + 1

                result_text = format_tool_result_for_llm(tool_result)
                merged_results.append(result_text)
                tool_calls_log.append(
                    {
                        "round": tool_call_count,
                        "tool": f"{server_name}.{tool_name}",
                        "args": arguments,
                        "result_len": len(result_text),
                    }
                )

                if save_conversation:
                    conversation.append(
                        {
                            "role": "tool",
                            "tool": f"{server_name}.{tool_name}",
                            "content": result_text[:30000],
                        }
                    )

            if should_rollback:
                continue

            rollback_count = 0
            if merged_results:
                history.append(
                    {
                        "role": "user",
                        "content": "\n".join(merged_results),
                        "kind": "tool_result",
                    }
                )
                if use_summary:
                    summary_prompt = generate_agent_summarize_prompt(question)
                    pass_length_check, history = _ensure_summary_context(
                        history,
                        summary_prompt,
                        last_call_tokens=last_call_tokens,
                        max_context_length=max_context_length,
                        max_tokens=max_gen_tokens,
                        conversation=conversation,
                    )
                    if not pass_length_check:
                        turn_count = max_turns
                        print(f"  [{tag}] Context limit reached, triggering summary")
                        break

        if use_summary:
            summary_prompt = generate_agent_summarize_prompt(question)
            summary_history = history + [
                {"role": "user", "content": summary_prompt, "kind": "summary_prompt"}
            ]
            max_final_answer_retries = (
                DEFAULT_MAX_FINAL_ANSWER_RETRIES if keep_tool_result == -1 else 1
            )
            for retry_idx in range(max_final_answer_retries):
                summary_messages = [{"role": "system", "content": system_prompt}] + build_messages_for_llm(
                    summary_history, keep_tool_result
                )
                final_answer, _, _ = await _chat_completion(
                    openai_http,
                    base_url=base_url,
                    model=model,
                    messages=summary_messages,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    session_id=session_id,
                )
                final_boxed_answer = _extract_answer_text(final_answer)
                if final_boxed_answer:
                    break
                print(
                    f"  [{tag}] Final summary missing boxed answer "
                    f"(attempt {retry_idx + 1}/{max_final_answer_retries})"
                )

            if save_conversation:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                        "final_answer": True,
                    }
                )

        boxed_answer = final_boxed_answer or _extract_answer_text(final_answer)
        if (context_compress_limit <= 0) and not boxed_answer and intermediate_boxed_answers:
            boxed_answer = intermediate_boxed_answers[-1]
            if not final_answer:
                final_answer = f"\\boxed{{{boxed_answer}}}"
        elapsed = time.time() - t0
        print(
            f"  [{tag}] Done: {tool_call_count} tools, {elapsed:.1f}s, "
            f"answer={boxed_answer or final_answer[:80]}"
        )
        return {
            "qid": qid,
            "traj_idx": traj_idx,
            "question": question,
            "answer": final_answer,
            "boxed_answer": boxed_answer,
            "num_tool_calls": tool_call_count,
            "tool_calls": tool_calls_log,
            "conversation": conversation,
            "latency_s": elapsed,
            "status": "success" if (final_answer or boxed_answer) else "no_final_answer",
        }
    finally:
        await http_client.aclose()
        await openai_http.aclose()
