"""The evaluation must declare the tools, exactly as training did.

Regression: the campaign built the model with DEFAULT_SYSTEM ("Respond with
interleaved text and audio"), which names no tool. Training embeds the
definitions in the system prompt, so evaluating without them asks the model to
call tools it was never told about — every tool-calling metric then reads zero
for the base model and a fine-tune alike, which is exactly what we measured.
"""

from __future__ import annotations

import json

from lfm2_audio.cli.eval.suite import build_parser, build_system
from lfm2_audio.core.chat_format import TOOL_LIST_END, TOOL_LIST_START
from lfm2_audio.core.prompt import DEFAULT_SYSTEM


def _args(*argv: str):
    return build_parser().parse_args(list(argv))


def test_should_keep_the_plain_system_prompt_when_no_tool_is_requested() -> None:
    assert build_system(_args()) == DEFAULT_SYSTEM


def test_should_embed_the_english_tool_set_on_the_shortcut() -> None:
    prompt = build_system(_args("--tool-definitions", "en"))

    assert TOOL_LIST_START in prompt and TOOL_LIST_END in prompt
    assert "web_search" in prompt
    assert "db_query" in prompt


def test_should_read_tool_definitions_from_a_file(tmp_path) -> None:
    path = tmp_path / "tools.json"
    path.write_text(json.dumps([{"name": "lookup", "description": "d", "parameters": {}}]), encoding="utf-8")

    prompt = build_system(_args("--tool-definitions", str(path)))

    assert "lookup" in prompt
    assert TOOL_LIST_START in prompt


def test_should_match_what_training_renders() -> None:
    """The eval prompt must be byte-identical to the packing one, or the
    comparison measures prompt drift rather than the model."""
    from lfm2_audio.core.chat_format import build_system_prompt
    from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS

    assert build_system(_args("--tool-definitions", "en")) == build_system_prompt(
        tool_definitions=TOOLCALLING_EN_TOOL_DEFINITIONS
    )
