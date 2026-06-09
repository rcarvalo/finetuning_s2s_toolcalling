"""Parsing streaming des tool calls dans le flux texte de la génération interleaved.

Le modèle émet, dans le flux texte (« inner monologue »), des appels pythonic
hérités du backbone LFM2.5 :

    <|tool_call_start|>[check_appointment(visitor_name="Marie Dupont")]<|tool_call_end|>

Ce module accumule les morceaux de texte décodés token par token, détecte les
spans complets ``start...end``, et parse les appels via ``ast`` (aucun eval).
Python pur — testable sans GPU ni torch.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from s2s_toolcalling.data.chat_format import TOOL_CALL_END, TOOL_CALL_START


class ToolCallParseError(ValueError):
    pass


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


def _literal(node: ast.expr, raw: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as e:
        raise ToolCallParseError(f"non-literal argument in tool call: {raw!r}") from e


def parse_tool_call_block(block: str) -> list[ParsedToolCall]:
    """Parse le contenu entre les marqueurs : ``[fn(a=1), fn2(b="x")]`` ou ``fn(a=1)``."""
    block = block.strip()
    if not block:
        raise ToolCallParseError("empty tool call block")

    try:
        tree = ast.parse(block, mode="eval")
    except SyntaxError as e:
        raise ToolCallParseError(f"invalid tool call syntax: {block!r}") from e

    body = tree.body
    call_nodes: list[ast.expr] = list(body.elts) if isinstance(body, ast.List) else [body]

    calls: list[ParsedToolCall] = []
    for node in call_nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ToolCallParseError(f"expected function call(s), got: {block!r}")
        arguments: dict[str, Any] = {}
        # Les positionnels sont acceptés mais signalés : le format LFM2 est kwargs-only.
        for i, arg in enumerate(node.args):
            arguments[f"_positional_{i}"] = _literal(arg, block)
        for kw in node.keywords:
            if kw.arg is None:
                raise ToolCallParseError(f"**kwargs not supported in tool call: {block!r}")
            arguments[kw.arg] = _literal(kw.value, block)
        calls.append(ParsedToolCall(name=node.func.id, arguments=arguments, raw=block))
    return calls


@dataclass
class StreamingToolCallParser:
    """Accumule le texte décodé et émet les tool calls complets.

    Usage :
        parser = StreamingToolCallParser()
        events = parser.feed(piece)   # liste de ParsedToolCall complétés par ce morceau
        parser.visible_text           # texte « propre » (spans de tool call retirés)
        parser.in_tool_call           # True si un span est ouvert et pas encore fermé
    """

    buffer: str = ""
    _consumed: int = 0  # offset jusqu'auquel les spans complets ont été extraits
    errors: list[str] = field(default_factory=list)

    def feed(self, piece: str) -> list[ParsedToolCall]:
        self.buffer += piece
        completed: list[ParsedToolCall] = []

        while True:
            start = self.buffer.find(TOOL_CALL_START, self._consumed)
            if start == -1:
                break
            end = self.buffer.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
            if end == -1:
                break  # span incomplet, on attend la suite
            inner = self.buffer[start + len(TOOL_CALL_START) : end]
            self._consumed = end + len(TOOL_CALL_END)
            try:
                completed.extend(parse_tool_call_block(inner))
            except ToolCallParseError as e:
                self.errors.append(str(e))
        return completed

    @property
    def in_tool_call(self) -> bool:
        return self.buffer.find(TOOL_CALL_START, self._consumed) != -1

    @property
    def visible_text(self) -> str:
        """Texte sans les spans de tool call (complets ou ouverts) ni marqueur partiel final."""
        out: list[str] = []
        pos = 0
        while True:
            start = self.buffer.find(TOOL_CALL_START, pos)
            if start == -1:
                tail = self.buffer[pos:]
                out.append(_strip_partial_marker(tail))
                break
            out.append(self.buffer[pos:start])
            end = self.buffer.find(TOOL_CALL_END, start + len(TOOL_CALL_START))
            if end == -1:
                break  # span ouvert : tout ce qui suit est masqué
            pos = end + len(TOOL_CALL_END)
        return "".join(out)


def _strip_partial_marker(tail: str) -> str:
    """Retire un éventuel préfixe de marqueur en fin de buffer (ex. ``<|tool_ca``)."""
    for n in range(min(len(TOOL_CALL_START) - 1, len(tail)), 0, -1):
        if TOOL_CALL_START.startswith(tail[-n:]):
            return tail[:-n]
    return tail
