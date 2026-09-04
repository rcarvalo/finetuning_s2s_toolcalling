"""System prompt of a tool-calling evaluation — one resolver for every context.

Training embeds the tool definitions in the system prompt
(`preprocess_sft --tool-definitions`). Any evaluation of tool calling must
render the *same* prompt, or it measures prompt drift instead of the model:
the eval CLI and the in-training scoring callback both resolve through here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lfm2_audio.core.chat_format import build_system_prompt
from lfm2_audio.core.prompt import DEFAULT_SYSTEM

EN_SHORTCUT = "en"
"""Named shortcut for the web_search + db_query set used by the EN corpus."""


def resolve_tool_definitions(spec: str | None) -> list[dict[str, Any]] | None:
    """``None``, the ``"en"`` shortcut, or a path to a JSON list of tools."""
    if spec is None:
        return None
    if spec == EN_SHORTCUT:
        from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS

        return TOOLCALLING_EN_TOOL_DEFINITIONS
    definitions: list[dict[str, Any]] = json.loads(Path(spec).read_text(encoding="utf-8"))
    return definitions


def resolve_system(tool_definitions: str | None, instructions: str | None = None) -> str:
    """The campaign's system prompt: tool-aware when tools are declared.

    ``instructions`` is a path to a text file replacing the default instruction
    sentence, the tool list staying exactly as at training. It exists for one
    measurement: the trained prompt names its two tools and orders "call at
    most one", and the 28/08 probe showed that rule alone keeps the model from
    ever picking a third tool. Whether a wording change frees it is a
    prompt-only question, cheaper than any retrain — so it must be askable.
    """
    definitions = resolve_tool_definitions(tool_definitions)
    if definitions is None:
        return DEFAULT_SYSTEM
    if instructions is None:
        return build_system_prompt(tool_definitions=definitions)
    text = Path(instructions).read_text(encoding="utf-8").strip()
    return build_system_prompt(instructions=text, tool_definitions=definitions)
