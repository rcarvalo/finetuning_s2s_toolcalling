"""``ToolCallDiagnosis`` now lives in the evaluation toolkit; ``diagnose`` reads LFM2 text into it.

The toolkit's diagnosis works on a parse result, never on raw text: how a
model spells a call belongs to its parser. This module binds the LFM2 parser
so callers keep passing the generated text.
"""

from __future__ import annotations

from typing import Any

from avet.scorers.toolcall.argument_match import ArgMatch
from avet.scorers.toolcall.tool_call_diagnosis import OUTCOMES, ToolCallDiagnosis

from lfm2_audio.avet_components.tool_call_parser import Lfm2ToolCallParser


def diagnose(
    case_id: str,
    predicted_text: str,
    expected_calls: list[dict[str, Any]],
    *,
    arg_match: ArgMatch = "exact",
    threshold: float = 0.7,
) -> ToolCallDiagnosis:
    """Diagnose one case from the model's final text and the expected calls."""
    return ToolCallDiagnosis.of(
        case_id,
        Lfm2ToolCallParser().parse(predicted_text),
        expected_calls,
        text=predicted_text,
        arg_match=arg_match,
        threshold=threshold,
    )


__all__ = ["OUTCOMES", "ToolCallDiagnosis", "diagnose"]
