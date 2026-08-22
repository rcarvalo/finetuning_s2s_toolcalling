from lfm2_audio.core import chat_format as cf


def test_render_tool_call_kwargs():
    assert cf.render_tool_call("check_appointment", {"visitor_name": "Marie Dupont"}) == (
        'check_appointment(visitor_name="Marie Dupont")'
    )


def test_render_tool_call_types():
    rendered = cf.render_tool_call("f", {"s": "été", "i": 3, "x": 1.5, "b": True, "n": None, "l": [1, "a"]})
    assert rendered == 'f(s="été", i=3, x=1.5, b=True, n=None, l=[1, "a"])'


def test_render_tool_calls_block():
    block = cf.render_tool_calls([("get_guest_wifi", {})])
    assert block == "<|tool_call_start|>[get_guest_wifi()]<|tool_call_end|>"


def test_render_tool_response_json():
    out = cf.render_tool_response({"found": True, "salle": "B2"})
    assert out.startswith("<|tool_response_start|>") and out.endswith("<|tool_response_end|>")
    assert '"found":true' in out


def test_system_prompt_contains_tool_list():
    tools = [{"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}}}]
    prompt = cf.build_system_prompt("instructions", tools)
    assert prompt.startswith("instructions")
    assert "<|tool_list_start|>" in prompt and "<|tool_list_end|>" in prompt
    assert '"name":"t"' in prompt


def test_roundtrip_with_parser():
    from lfm2_audio.orchestrator.tool_parser import StreamingToolCallParser

    block = cf.render_tool_calls([("notify_employee", {"employee_name": "Karim Benali", "message": "Visiteur là"})])
    calls = StreamingToolCallParser().feed(block)
    assert len(calls) == 1
    assert calls[0].name == "notify_employee"
    assert calls[0].arguments == {"employee_name": "Karim Benali", "message": "Visiteur là"}
