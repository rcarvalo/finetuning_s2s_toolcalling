"""The dialogue JSONL schema now lives in the evaluation toolkit; historical import path kept.

One file serves TTS, training and evaluation, so the schema belongs to the
package all three depend on. ``ToolCall`` keeps its historical name here.
"""

from __future__ import annotations

from avet.datasets.dialogue import Dialogue, load_dialogues, parse_dialogue
from avet.datasets.dialogue_meta import DialogueMeta
from avet.datasets.dialogue_tool_call import DialogueToolCall as ToolCall
from avet.datasets.dialogue_turn import Role, Turn
from avet.errors import DialogueValidationError

VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")

__all__ = [
    "VALID_ROLES",
    "Dialogue",
    "DialogueMeta",
    "DialogueValidationError",
    "Role",
    "ToolCall",
    "Turn",
    "load_dialogues",
    "parse_dialogue",
]
