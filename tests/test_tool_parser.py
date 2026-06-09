import pytest

from s2s_toolcalling.orchestrator.tool_parser import (
    StreamingToolCallParser,
    ToolCallParseError,
    parse_tool_call_block,
)

CALL = '<|tool_call_start|>[check_appointment(visitor_name="Marie Dupont", host_name="Claire Martin")]<|tool_call_end|>'


def test_parse_block_single():
    calls = parse_tool_call_block('[get_guest_wifi()]')
    assert calls[0].name == "get_guest_wifi"
    assert calls[0].arguments == {}


def test_parse_block_without_list():
    calls = parse_tool_call_block('guide_visitor(destination="cafétéria")')
    assert calls[0].arguments == {"destination": "cafétéria"}


def test_parse_block_multiple_calls():
    calls = parse_tool_call_block('[f(a=1), g(b="x")]')
    assert [c.name for c in calls] == ["f", "g"]


def test_parse_block_invalid_syntax():
    with pytest.raises(ToolCallParseError):
        parse_tool_call_block("not a call!!")


def test_parse_block_rejects_non_literal():
    with pytest.raises(ToolCallParseError):
        parse_tool_call_block("f(a=os.system('rm -rf /'))")


def test_streaming_whole():
    parser = StreamingToolCallParser()
    calls = parser.feed(CALL)
    assert len(calls) == 1
    assert calls[0].arguments["visitor_name"] == "Marie Dupont"
    assert parser.visible_text == ""


def test_streaming_token_by_token():
    parser = StreamingToolCallParser()
    calls = []
    # le marqueur arrive entier (token spécial), le reste en petits morceaux
    pieces = ["Je vérifie. ", "<|tool_call_start|>", "[check_appointment(", 'visitor_name="Ma', 'rie")', "]", "<|tool_call_end|>"]
    for p in pieces:
        calls.extend(parser.feed(p))
    assert len(calls) == 1
    assert calls[0].arguments == {"visitor_name": "Marie"}
    assert parser.visible_text == "Je vérifie. "


def test_streaming_text_before_and_after():
    parser = StreamingToolCallParser()
    parser.feed("Bonjour ! " + CALL)
    parser.feed(" Et voilà.")
    assert parser.visible_text == "Bonjour !  Et voilà."


def test_streaming_open_span_masks_text():
    parser = StreamingToolCallParser()
    parser.feed("Ok. <|tool_call_start|>[secret_so_far(")
    assert parser.in_tool_call
    assert parser.visible_text == "Ok. "


def test_streaming_partial_marker_held_back():
    parser = StreamingToolCallParser()
    parser.feed("Bonjour <|tool_ca")
    assert parser.visible_text == "Bonjour "


def test_streaming_malformed_call_recorded_as_error():
    parser = StreamingToolCallParser()
    calls = parser.feed("<|tool_call_start|>[broken(]<|tool_call_end|>")
    assert calls == []
    assert parser.errors
