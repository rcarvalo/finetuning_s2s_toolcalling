"""Shared tool-prompt resolver (`lfm2_audio.evaluation.tool_prompt`).

The eval CLI and the in-training scoring callback must render the exact prompt
training embedded — one resolver, asserted against the packing renderer.
"""

from __future__ import annotations

import json

from lfm2_audio.core.chat_format import TOOL_LIST_START, build_system_prompt
from lfm2_audio.core.prompt import DEFAULT_SYSTEM
from lfm2_audio.evaluation.tool_prompt import resolve_system, resolve_tool_definitions


def test_should_fall_back_to_the_plain_prompt_without_tools() -> None:
    assert resolve_system(None) == DEFAULT_SYSTEM


def test_should_render_the_en_shortcut_like_training_does() -> None:
    from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS

    assert resolve_system("en") == build_system_prompt(tool_definitions=TOOLCALLING_EN_TOOL_DEFINITIONS)


def test_should_load_definitions_from_a_json_file(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([{"name": "lookup", "parameters": {}}]), encoding="utf-8")

    prompt = resolve_system(str(path))

    assert TOOL_LIST_START in prompt
    assert "lookup" in prompt


def test_should_return_none_definitions_for_none() -> None:
    assert resolve_tool_definitions(None) is None
