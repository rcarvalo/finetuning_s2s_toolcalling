"""``ToolCallScorer`` with this model family's parser wired in by default.

The toolkit's scorer works on any parser; here the LFM2 pythonic span is the
default, so ``ToolCallScorer()`` keeps meaning what it always meant.
"""

from __future__ import annotations

from avet.scorers.text.tool_call_scorer import ToolCallScorer as _ToolCallScorer
from avet.scorers.toolcall.argument_match import ArgMatch
from avet.text.tool_call_parser import ToolCallParser

from lfm2_audio.avet_components.tool_call_parser import Lfm2ToolCallParser


class ToolCallScorer(_ToolCallScorer):
    """BFCL-style tool-call accuracy on LFM2 replies."""

    def __init__(
        self,
        parser: ToolCallParser | None = None,
        *,
        arg_match: ArgMatch = "token_f1",
        threshold: float = 0.7,
    ) -> None:
        super().__init__(parser or Lfm2ToolCallParser(), arg_match=arg_match, threshold=threshold)


__all__ = ["ToolCallScorer"]
