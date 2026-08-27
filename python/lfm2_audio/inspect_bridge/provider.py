"""``Lfm2AudioAPI`` — LFM2.5-Audio as an Inspect model provider.

This is the piece Inspect cannot supply: a model that *hears* and *speaks*.
Everything else in a campaign — parallelism, retries, multi-variant runs,
scoring, the log and its viewer — already exists upstream and is better tested
than anything we would write. Once a variant is addressable as
``lfm2/<checkpoint>``, ``inspect eval`` drives it like any other model.

Two things this provider does that a text provider does not:

- the last user message may carry ``ContentAudio``; the waveform is decoded and
  fed to the model rather than described to it;
- the reply carries the generated speech back as ``ContentAudio``, which is what
  makes the viewer draw a player next to the answer being scored.

Concurrency warning: a local checkpoint holds one GPU, so ``--max-samples``
above 1 does not speed it up and will exhaust VRAM. Raise it for endpoints,
which are independent workers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ContentAudio,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.model import ContentText as InspectText
from inspect_ai.tool import ToolChoice, ToolInfo

from lfm2_audio.core.prompt import resolve_system
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply
from lfm2_audio.inspect_bridge.audio import data_uri_to_waveform, waveform_to_data_uri
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)

PROVIDER_NAME = "lfm2"
ENDPOINT_PROVIDER = "lfm2-endpoint"


@modelapi(name=PROVIDER_NAME)
class Lfm2AudioAPI(ModelAPI):
    """Serves a local LFM2.5-Audio checkpoint to Inspect.

    ``--model lfm2/LiquidAI/LFM2.5-Audio-1.5B`` loads the base model;
    ``-M adapter=Rcarvalo/…`` evaluates a fine-tune of it. The model is loaded
    once per provider instance, which is once per ``inspect eval`` variant.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,  # noqa: ANN401 — contrat amont : -M passe des kwargs libres
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars or [], config)
        self._adapter = model_args.pop("adapter", None)
        self._backend = model_args.pop("backend", "auto")
        system = model_args.pop("system", None)
        if isinstance(system, list):
            # Inspect's -M parsing splits values on commas; glue the prompt back.
            system = ",".join(system)
        # A name ('bilingual') resolves to its text; anything else passes through.
        self._system = resolve_system(system)
        self._model_args: dict[str, Any] = model_args
        self._model: LFM2Audio | None = None

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """One turn: the last user message in, text and speech out."""
        text, audio = _last_user_turn(input)
        started = time.perf_counter()
        reply = self._load().reply(text=text, audio=audio, max_tokens=config.max_tokens)
        return _to_output(self.model_name, reply, elapsed=time.perf_counter() - started)

    def _load(self) -> LFM2Audio:
        if self._model is None:
            logger.info("chargement de %s (backend=%s)", self.model_name, self._backend)
            self._model = LFM2Audio.from_pretrained(
                self.model_name,
                adapter=self._adapter,
                backend=self._backend,
                **({"system": self._system} if self._system else {}),  # type: ignore[arg-type]
                **self._model_args,
            )
        # Inspect keeps one provider per variant, so history from the previous
        # sample would leak into this one; every sample must start clean.
        self._model.reset()
        return self._model


def _last_user_turn(messages: list[ChatMessage]) -> tuple[str | None, Waveform | None]:
    """The question to answer: its text, and its audio when the sample carries one."""
    user = next((m for m in reversed(messages) if m.role == "user"), None)
    if user is None:
        return None, None
    if isinstance(user.content, str):
        return user.content, None

    text_parts = [part.text for part in user.content if isinstance(part, InspectText)]
    audio_parts = [part for part in user.content if isinstance(part, ContentAudio)]
    audio = data_uri_to_waveform(audio_parts[0].audio) if audio_parts else None
    text = " ".join(text_parts).strip() or None
    # A spoken question is the question: sending its transcript alongside would
    # let the model read instead of listen, which is not what we measure.
    return (None, audio) if audio is not None else (text, None)


def _to_output(model: str, reply: Reply, *, elapsed: float) -> ModelOutput:
    """The reply as Inspect sees it: raw text kept, speech attached."""
    content: list[Any] = [InspectText(text=reply.raw_text or reply.text)]
    if reply.audio is not None and not reply.audio.is_empty:
        content.append(ContentAudio(audio=waveform_to_data_uri(reply.audio), format="wav"))

    output = ModelOutput.from_message(ChatMessageAssistant(content=content))
    output.model = model
    output.time = elapsed
    # TTFA and frame count are the numbers a voice assistant is judged on, and
    # they exist nowhere else in the log.
    output.metadata = {k: v for k, v in reply.metrics.as_dict().items() if v is not None}
    return output


@modelapi(name=ENDPOINT_PROVIDER)
class Lfm2EndpointAPI(ModelAPI):
    """Serves a deployed variant to Inspect: ``--model lfm2-endpoint/<id>``.

    Same contract as the local provider, opposite concurrency profile: an
    endpoint is an independent worker and the time goes into HTTP, so this is
    where ``--max-samples`` above 1 actually pays.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,  # noqa: ANN401 — contrat amont : -M passe des kwargs libres
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars or [], config)
        self._client = LiquidAudioClient(model_name, api_key=api_key)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        text, audio = _last_user_turn(input)
        started = time.perf_counter()
        reply = self._client.invoke(text=text, audio=audio, max_tokens=config.max_tokens)
        return _to_output(self.model_name, reply, elapsed=time.perf_counter() - started)
