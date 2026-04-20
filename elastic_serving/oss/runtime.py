import datetime
from typing import Any, Dict, List, Optional, Sequence

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)

from elastic_serving.oss.browser import BrowserTool, SerperServiceBrowserBackend
from elastic_serving.oss.vllm_generator import vLLMAsyncGenerator

DEVELOPER_CONTENT = """
You are a helpful assistant and harmless assistant.

You will be able to use a set of browsering tools to answer user queries.

Tool for browsing.
The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
Cite information from the tool using the following format:
`\u3010{cursor}\u2020L{line_start}(-L{line_end})?\u3011`, for example: `\u30106\u2020L9-L11\u3011` or `\u30108\u2020L3\u3011`.
Do not quote more than 10 words directly from the tool output.
sources=web
""".strip()


class BrowserPool:
    def __init__(self, blocked_substrings: Sequence[str] | None = None):
        self.sessions: Dict[Any, BrowserTool] = {}
        self.blocked_substrings = list(blocked_substrings or [])

    def init_session(self, qid: Any) -> dict:
        tool = BrowserTool(
            backend=SerperServiceBrowserBackend(
                blocked_substrings=self.blocked_substrings,
            )
        )
        self.sessions[qid] = tool
        return tool.tool_config

    async def call(self, qid: Any, tool_call_msg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool = self.sessions[qid]
        tool_call_msg = Message.from_dict(tool_call_msg_dict)
        results = []
        async for msg in tool.process(tool_call_msg):
            results.append(msg)
        return [m.to_dict() for m in results]

    def search_results(self, qid: Any) -> List[Dict[str, Any]]:
        tool = self.sessions.get(qid)
        if not tool:
            return []
        return list(getattr(tool.backend, "search_results", []))

    def searched_urls(self, qid: Any) -> List[str]:
        return list(
            {
                result["url"]
                for result in self.search_results(qid)
                if result.get("url")
            }
        )

    def cleanup(self, qid: Any):
        if qid in self.sessions:
            del self.sessions[qid]


async def _generate_with_retry(
    generator,
    tokens: List[int],
    stop_tokens: List[int],
    encoding,
    max_retries: int = 20,
) -> List[Message]:
    assert max_retries > 0
    last_exception = None

    for attempt in range(1, max_retries + 1):
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        parse_error = None
        draining = False

        stream = generator.generate(tokens, stop_tokens)
        try:
            async for token_id in stream:
                if not draining:
                    try:
                        parser.process(token_id)
                    except Exception as pe:
                        parse_error = pe
                        draining = True

            if parse_error is not None:
                last_exception = parse_error
                print(f"\n--- Generation failed on attempt {attempt}/{max_retries} (parse error) ---")
                continue

            return parser.messages

        except Exception as e:
            last_exception = e
            print(f"\n--- Generation failed on attempt {attempt}/{max_retries} ---")

        finally:
            try:
                await stream.aclose()
            except Exception:
                pass

    if last_exception:
        raise last_exception
    raise RuntimeError("Generation failed after retries without a captured exception.")


class OSSEngineRuntime:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        default_reasoning_effort: str = "high",
        blocked_substrings: Sequence[str] | None = None,
    ):
        self.generator = vLLMAsyncGenerator(
            model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.browser_pool = BrowserPool(blocked_substrings=blocked_substrings)
        self.default_reasoning_effort = default_reasoning_effort

    async def run_one(
        self,
        *,
        question: str,
        qid: Any,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        tool_config = self.browser_pool.init_session(qid)

        effort = reasoning_effort or self.default_reasoning_effort
        system_message_content = (
            SystemContent.new()
            .with_reasoning_effort(
                {
                    "high": ReasoningEffort.HIGH,
                    "medium": ReasoningEffort.MEDIUM,
                    "low": ReasoningEffort.LOW,
                }[effort]
            )
            .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
            .with_tools(tool_config)
        )
        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_message_content),
            Message.from_role_and_content(Role.DEVELOPER, DEVELOPER_CONTENT),
            Message.from_role_and_content(Role.USER, "Question: " + question),
        ]

        try:
            while True:
                last_message = messages[-1]

                if getattr(last_message, "recipient", None) and str(last_message.recipient).startswith("browser."):
                    tool_call_msg_dict = last_message.to_dict()
                    result_msgs_dict = await self.browser_pool.call(qid, tool_call_msg_dict)
                    result_msgs = [Message.from_dict(m) for m in result_msgs_dict]
                    messages += result_msgs
                    continue

                if last_message.author.role == Role.ASSISTANT and getattr(last_message, "channel", None) == "final":
                    break

                conversation = Conversation.from_messages(messages)
                tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

                new_messages = await _generate_with_retry(
                    self.generator,
                    tokens,
                    self.encoding.stop_tokens_for_assistant_actions(),
                    self.encoding,
                )

                messages += new_messages
                if (
                    messages
                    and messages[-1].author.role == Role.ASSISTANT
                    and getattr(messages[-1], "channel", None) == "final"
                ):
                    break

            conv = Conversation.from_messages(messages)
            return {
                "messages": [m.to_dict() for m in conv.messages],
                "searched_urls": self.browser_pool.searched_urls(qid),
                "search_results": self.browser_pool.search_results(qid),
            }

        finally:
            self.browser_pool.cleanup(qid)

    def shutdown(self) -> None:
        self.generator.shutdown()
