"""``Lfm2ToolCallParser`` — the pythonic LFM2 tool-call span as the toolkit's parse result.

An open span (vLLM strips the stop token) is reported as unterminated rather
than silently dropped: on a negative case it would otherwise read as a
correct abstention.
"""

from __future__ import annotations

from avet.text.parsed_call import ParsedCall
from avet.text.tool_call_parse import ToolCallParse
from avet.text.tool_call_parser import TOOL_CALL_PARSERS

from lfm2_audio.core.chat_format import TOOL_CALL_SPAN
from lfm2_audio.orchestrator.tool_parser import StreamingToolCallParser


@TOOL_CALL_PARSERS.register("lfm2")
class Lfm2ToolCallParser:
    """``<|tool_call_start|>[fn(arg="v")]<|tool_call_end|>`` → calls, errors, open span."""

    def parse(self, text: str) -> ToolCallParse:
        parser = StreamingToolCallParser()
        calls = tuple(ParsedCall(name=call.name, arguments=dict(call.arguments)) for call in parser.feed(text))
        span = TOOL_CALL_SPAN.search(text)
        return ToolCallParse(
            calls=calls,
            errors=tuple(parser.errors),
            unterminated=parser.in_tool_call,
            raw_span=span.group(0) if span else "",
        )
