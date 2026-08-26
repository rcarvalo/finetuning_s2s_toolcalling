"""``TurnTrajectoryBuilder`` — the steps of a turn, recorded as it is produced.

Kept apart from the generators so both of them — a local model and a serverless
endpoint — record the same shape. A trajectory assembled differently on each
path would be worse than none: the viewer would show two runs that cannot be
read side by side.

The steps are derived from what a reply already carries (its raw text, its tool
calls, its timings); nothing extra is generated, and nothing is inferred that
the run did not actually do.
"""

from __future__ import annotations

from typing import Any

from lfm2_audio.core.prompt import spoken_part
from lfm2_audio.ds.reply import Reply
from lfm2_audio.evaluation.tool_call_diagnosis import ToolCallDiagnosis
from lfm2_audio.evaluation.trajectory import Trajectory


class TurnTrajectoryBuilder:
    """Builds the ordered steps of one turn from its question and its reply."""

    def __init__(self, *, prompt_text: str, spoken_prompt: bool = False) -> None:
        self._prompt_text = prompt_text
        self._spoken_prompt = spoken_prompt

    def build(self, reply: Reply, *, tool_results: list[dict[str, Any]] | None = None) -> Trajectory:
        """Prompt, the calls the model emitted, their results, then the answer."""

        trajectory = Trajectory()
        trajectory.add("prompt", self._prompt_text, spoken=self._spoken_prompt)

        raw = reply.raw_text or reply.text
        diagnosis = ToolCallDiagnosis.of("turn", raw, [])
        for index, call in enumerate(diagnosis.predicted):
            trajectory.add(
                "tool_call",
                f"{call['name']}({call.get('arguments', {})})",
                name=call["name"],
                arguments=call.get("arguments", {}),
                call_index=index,
            )
        for error in diagnosis.parse_errors:
            trajectory.add("error", error, stage="tool_call_parse")

        for index, result in enumerate(tool_results or []):
            trajectory.add(
                "tool_result",
                str(result.get("result", ""))[:2000],
                name=result.get("name"),
                ok=result.get("ok", True),
                call_index=index,
            )

        metrics = reply.metrics
        trajectory.add(
            "answer",
            spoken_part(raw),
            elapsed_s=metrics.total_s or None,
            ttfa_s=metrics.ttfa_s,
            audio_frames=metrics.audio_frames,
        )
        return trajectory
