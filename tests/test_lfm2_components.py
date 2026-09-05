"""The LFM2 cleaner and parser the toolkit picks up through the ``avet.components`` entry point."""

from __future__ import annotations

from avet.bootstrap import ComponentLoader
from avet.text.text_cleaner import TEXT_CLEANERS
from avet.text.tool_call_parser import TOOL_CALL_PARSERS

from lfm2_audio.avet_components.text_cleaner import Lfm2TextCleaner
from lfm2_audio.avet_components.tool_call_parser import Lfm2ToolCallParser
from lfm2_audio.core.chat_format import TOOL_CALL_END, TOOL_CALL_START, render_tool_calls


def test_cleaner_should_drop_the_call_span_and_the_markers() -> None:
    text = render_tool_calls([("web_search", {"query": "x"})]) + " Voici la réponse.<|text_end|>"

    assert Lfm2TextCleaner().clean(text) == "Voici la réponse."


def test_parser_should_read_calls_from_the_span() -> None:
    parse = Lfm2ToolCallParser().parse(render_tool_calls([("web_search", {"query": "météo à Paris"})]))

    assert [call.as_dict() for call in parse.calls] == [{"name": "web_search", "arguments": {"query": "météo à Paris"}}]
    assert parse.parsed
    assert parse.raw_span.startswith(TOOL_CALL_START)


def test_parser_should_flag_positional_arguments_errors_and_open_spans() -> None:
    parser = Lfm2ToolCallParser()

    positional = parser.parse(f'{TOOL_CALL_START}[web_search("x")]{TOOL_CALL_END}')
    broken = parser.parse(f"{TOOL_CALL_START}[not a call!!{TOOL_CALL_END}")
    unterminated = parser.parse(f'{TOOL_CALL_START}[web_search(query="x")]')

    assert positional.calls[0].arguments == {"_positional_0": "x"}
    assert broken.errors and not broken.calls and broken.attempted
    assert unterminated.unterminated and unterminated.attempted


def test_components_should_be_registered_through_the_entry_point() -> None:
    ComponentLoader.reset()
    ComponentLoader.load()

    assert TEXT_CLEANERS.resolve("lfm2") is Lfm2TextCleaner
    assert TOOL_CALL_PARSERS.resolve("lfm2") is Lfm2ToolCallParser
