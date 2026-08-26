"""``InspectTranscript`` — one of our samples rendered as an Inspect message list.

The viewer shows a turn as a conversation, so the pipeline has to *be* one:
what the user asked, the call the model emitted, what the tool answered, then
the spoken reply. Collapsing all of it into a single assistant bubble — which is
what a naive export does — hides exactly the step a reader is looking for when a
tool call goes wrong.

The spoken audio rides on the final assistant message, which is what makes the
viewer draw a player next to the answer it is scoring.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser, Content, ContentAudio
from inspect_ai.model import ContentText as InspectText
from inspect_ai.tool import ToolCall, ToolCallError

from lfm2_audio.core.prompt import spoken_part
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.tool_call_diagnosis import ToolCallDiagnosis
from lfm2_audio.inspect_bridge.audio import waveform_to_data_uri
from lfm2_audio.scorer.sample import EvalSample

NO_ANSWER = "(no spoken answer)"


class InspectTranscript:
    """Turns one :class:`EvalSample` into the messages the viewer renders."""

    def __init__(self, sample: EvalSample, *, audio: Waveform | None = None) -> None:
        self._sample = sample
        self._audio = audio if audio is not None else sample.predicted_audio

    def messages(self) -> list[ChatMessage]:
        """User turn, tool round-trip, then the spoken answer."""
        messages: list[ChatMessage] = [ChatMessageUser(content=self._user_content())]
        calls = self._tool_calls()
        if calls:
            messages.append(ChatMessageAssistant(content="", tool_calls=calls))
            messages.extend(self._tool_results(calls))
        messages.append(self._answer())
        return messages

    def _user_content(self) -> str:
        return self._sample.prompt_text or "(spoken question, no transcript)"

    def _tool_calls(self) -> list[ToolCall]:
        """Calls the model actually emitted, parsed back from its raw text.

        Read from ``tool_results`` when the run recorded executions, otherwise
        from the diagnosis of the generated text — a campaign that never ran a
        tool still emitted the call, and that is the thing to look at.
        """
        recorded = [r for r in self._sample.tool_results if r.get("name")]
        if recorded:
            return [
                ToolCall(id=f"call_{index}", function=str(row["name"]), arguments=dict(row.get("arguments") or {}))
                for index, row in enumerate(recorded)
            ]
        diagnosis = ToolCallDiagnosis.of(self._sample.sample_id, self._sample.predicted_text, [])
        return [
            ToolCall(id=f"call_{index}", function=str(call["name"]), arguments=dict(call.get("arguments") or {}))
            for index, call in enumerate(diagnosis.predicted)
        ]

    def _tool_results(self, calls: list[ToolCall]) -> list[ChatMessageTool]:
        results: list[ChatMessageTool] = []
        for index, call in enumerate(calls):
            row: dict[str, Any] = self._sample.tool_results[index] if index < len(self._sample.tool_results) else {}
            payload = row.get("result", row.get("result_preview", ""))
            results.append(
                ChatMessageTool(
                    content=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)[:4000],
                    tool_call_id=call.id,
                    function=call.function,
                    error=None if row.get("ok", True) else ToolCallError("unknown", str(row.get("result", ""))),
                )
            )
        return results

    def _answer(self) -> ChatMessageAssistant:
        """The spoken reply: text without the call markers, plus the audio."""
        spoken = spoken_part(self._sample.predicted_text)
        content: list[Content] = [InspectText(text=spoken or NO_ANSWER)]
        if self._audio is not None and not self._audio.is_empty:
            content.append(ContentAudio(audio=waveform_to_data_uri(self._audio), format="wav"))
        return ChatMessageAssistant(content=content)
