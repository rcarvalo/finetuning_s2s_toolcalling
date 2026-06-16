from s2s_toolcalling.data import synth_dialogues as sd
from s2s_toolcalling.data.dialogue_schema import parse_dialogue
from s2s_toolcalling.tools.toolcalling_en import build_toolcalling_en_registry

REGISTRY = build_toolcalling_en_registry()


def _case(target, **kw):
    return sd.SynthCase(target=target, **kw)


def test_verify_valid_web_search():
    case = _case("web_search", utterance="search the news about Mars", arguments={"query": "news about Mars"})
    assert sd.verify_case(case, REGISTRY) is None


def test_verify_valid_db_query():
    case = _case("db_query", utterance="how many open orders", arguments={"question": "how many open orders"})
    assert sd.verify_case(case, REGISTRY) is None


def test_verify_valid_negative():
    case = _case("none", utterance="hey how are you", answer="I'm doing great, thanks!")
    assert sd.verify_case(case, REGISTRY) is None


def test_negative_with_arguments_rejected():
    case = _case("none", utterance="hi", answer="hello", arguments={"query": "x"})
    assert sd.verify_case(case, REGISTRY) is not None


def test_negative_without_answer_rejected():
    case = _case("none", utterance="hi", answer="")
    assert "answer" in sd.verify_case(case, REGISTRY)


def test_negative_with_placeholder_answer_rejected():
    case = _case("none", utterance="what time is it", answer="It's [current time].")
    assert "placeholder" in sd.verify_case(case, REGISTRY)
    ok = _case("none", utterance="thanks", answer="You're welcome!")
    assert sd.verify_case(ok, REGISTRY) is None


def test_missing_required_argument_rejected():
    case = _case("web_search", utterance="search something", arguments={})
    assert "missing required" in sd.verify_case(case, REGISTRY)


def test_unknown_tool_rejected():
    case = _case("translate", utterance="translate this", arguments={"text": "x"})
    assert "unknown tool" in sd.verify_case(case, REGISTRY)


def test_empty_utterance_rejected():
    case = _case("web_search", utterance="   ", arguments={"query": "x"})
    assert sd.verify_case(case, REGISTRY) == "empty utterance"


def test_args_with_special_chars_round_trip():
    # guillemets / accents / virgules ne doivent pas casser le round-trip pythonic.
    q = 'orders for "Acmé, Inc." since 2024'
    case = _case("db_query", utterance="...", arguments={"question": q})
    assert sd.verify_case(case, REGISTRY) is None


def test_parse_generation_response_plain_json():
    text = '[{"utterance": "find flights to Tokyo", "tool": "web_search", "arguments": {"query": "flights to Tokyo"}}]'
    cases = sd.parse_generation_response(text, target="web_search", style="direct command", depth="explicit arguments")
    assert len(cases) == 1 and cases[0].arguments == {"query": "flights to Tokyo"}


def test_parse_generation_response_fenced_and_negative():
    text = '```json\n[{"utterance": "good morning", "tool": "none", "answer": "Good morning!"}]\n```'
    cases = sd.parse_generation_response(text, target="none", style="polite question", depth="explicit arguments")
    assert len(cases) == 1 and cases[0].target == "none" and cases[0].answer == "Good morning!"


def test_parse_generation_response_garbage_returns_empty():
    assert sd.parse_generation_response("not json at all", target="none", style="", depth="") == []


def test_contamination_filter_flags_near_duplicate():
    f = sd.ContaminationFilter(held_out=["what is the weather in Paris today"], threshold=0.5)
    assert f.is_contaminated("what is the weather in Paris today?")
    assert not f.is_contaminated("how many employees are in the sales team")


def test_jaccard_trigram_bounds():
    assert sd.jaccard_trigram("hello world", "hello world") == 1.0
    assert sd.jaccard_trigram("abc", "xyz") == 0.0


def test_case_to_dialogue_is_valid_schema_positive():
    case = _case("web_search", utterance="search dogs", arguments={"query": "dogs"}, style="direct command", depth="explicit arguments")
    dlg = sd.case_to_dialogue(case, 0, tools=["web_search", "db_query"])
    parsed = parse_dialogue(dlg)  # ne doit pas lever
    assert parsed.turns[0].role == "user" and parsed.turns[0].text == "search dogs"
    assert parsed.turns[1].tool_calls[0].name == "web_search"


def test_case_to_dialogue_is_valid_schema_negative():
    case = _case("none", utterance="thanks", answer="You're welcome!")
    dlg = sd.case_to_dialogue(case, 1, tools=["web_search", "db_query"])
    parsed = parse_dialogue(dlg)
    assert parsed.turns[1].role == "assistant" and parsed.turns[1].text == "You're welcome!"
    assert not parsed.turns[1].tool_calls


def test_loop_verify_valid_positive():
    case = _case("db_query", utterance="how many open orders", arguments={"question": "how many open orders"},
                 answer="There are 4 open orders right now.")
    case.tool_result = {"count": 4}
    assert sd.verify_case(case, REGISTRY, mode="loop") is None


def test_loop_positive_needs_tool_result():
    case = _case("web_search", utterance="latest news", arguments={"query": "latest news"}, answer="Here's the latest.")
    assert "tool_result" in sd.verify_case(case, REGISTRY, mode="loop")


def test_loop_positive_needs_spoken_answer():
    case = _case("web_search", utterance="latest news", arguments={"query": "latest news"})
    case.tool_result = {"results": []}
    assert "spoken answer" in sd.verify_case(case, REGISTRY, mode="loop")


def test_loop_positive_rejects_placeholder_answer():
    case = _case("web_search", utterance="time in Tokyo", arguments={"query": "time in Tokyo"},
                 answer="It is [current time] in Tokyo.")
    case.tool_result = {"time": "10:30"}
    assert "placeholder" in sd.verify_case(case, REGISTRY, mode="loop")


def test_loop_case_to_dialogue_positive_is_four_turns():
    from s2s_toolcalling.data.dialogue_schema import parse_dialogue

    case = _case("db_query", utterance="how many orders", arguments={"question": "how many orders"},
                 answer="You have 4 open orders.")
    case.tool_result = {"count": 4}
    dlg = sd.case_to_dialogue(case, 0, tools=["web_search", "db_query"], mode="loop")
    roles = [t["role"] for t in dlg["turns"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert dlg["turns"][2]["content"] == {"count": 4}
    assert dlg["turns"][3]["text"] == "You have 4 open orders."
    parse_dialogue(dlg)  # ne doit pas lever


def test_loop_case_to_dialogue_negative_is_two_turns():
    case = _case("none", utterance="thanks", answer="You're welcome!")
    dlg = sd.case_to_dialogue(case, 1, tools=["web_search", "db_query"], mode="loop")
    assert [t["role"] for t in dlg["turns"]] == ["user", "assistant"]


def test_loop_parse_generation_response():
    text = ('[{"utterance": "how many open orders", "tool": "db_query", '
            '"arguments": {"question": "how many open orders"}, '
            '"tool_result": {"count": 4}, "answer": "There are 4 open orders."}]')
    cases = sd.parse_generation_response(text, target="db_query", style="direct command",
                                         depth="explicit arguments", mode="loop")
    assert len(cases) == 1
    assert cases[0].tool_result == {"count": 4}
    assert cases[0].answer == "There are 4 open orders."


def test_dedup_cases():
    cases = [
        _case("web_search", utterance="Search for cats", arguments={"query": "cats"}),
        _case("web_search", utterance="search for cats", arguments={"query": "cats"}),
        _case("web_search", utterance="search for dogs", arguments={"query": "dogs"}),
    ]
    assert len(sd.dedup_cases(cases)) == 2
