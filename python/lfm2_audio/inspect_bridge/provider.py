"""``Lfm2AudioAPI`` — LFM2.5-Audio as an Inspect model provider.

Built on the evaluation toolkit's provider contract
(:class:`avet.providers.audio_model_api.SyncAudioModelAPI`): the toolkit
reads the last user turn (audio wins over its transcript), times the call
and writes the reply — text, speech, timings — into the log. What is left
here is loading the checkpoint and talking to the endpoint.

Concurrency: a local checkpoint holds one GPU, so ``--max-samples`` above 1
buys nothing and exhausts VRAM; the endpoint is an independent worker and
raises its own connection count.
"""

from __future__ import annotations

import logging
from typing import Any

from avet.providers.audio_model_api import SyncAudioModelAPI
from avet.providers.audio_reply import AudioReply
from avet.providers.audio_turn import AudioTurn
from inspect_ai.model import GenerateConfig, modelapi

from lfm2_audio.core.prompt import resolve_system
from lfm2_audio.ds.reply import Reply
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)

PROVIDER_NAME = "lfm2"
ENDPOINT_PROVIDER = "lfm2-endpoint"


def to_audio_reply(reply: Reply) -> AudioReply:
    """The reply as the toolkit records it: raw text kept, speech and timings attached."""
    metrics = reply.metrics
    audio = reply.audio if reply.audio is not None and not reply.audio.is_empty else None
    return AudioReply(
        text=reply.text,
        raw_text=reply.raw_text or reply.text,
        audio=audio,
        ttfa_s=metrics.ttfa_s,
        audio_frames=int(metrics.audio_frames or 0),
    )


@modelapi(name=PROVIDER_NAME)
class Lfm2AudioAPI(SyncAudioModelAPI):
    """Serves a local LFM2.5-Audio checkpoint to Inspect.

    ``--model lfm2/LiquidAI/LFM2.5-Audio-1.5B`` loads the base model;
    ``-M adapter=Rcarvalo/…`` evaluates a fine-tune of it; ``-M system=bilingual``
    names a frozen system prompt.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        **model_args: Any,  # noqa: ANN401 — upstream contract: -M passes free kwargs
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config, **model_args)
        self._adapter = self.model_args.pop("adapter", None)
        self._backend = self.model_args.pop("backend", "auto")
        system = self.model_args.pop("system", None)
        if isinstance(system, list):
            # Inspect's -M parsing splits values on commas; glue the prompt back.
            system = ",".join(system)
        # A name ('bilingual') resolves to its text; anything else passes through.
        self._system = resolve_system(system)
        self._model: LFM2Audio | None = None

    def reply_sync(self, turn: AudioTurn, config: GenerateConfig) -> AudioReply:
        """One turn: the last user message in, text and speech out."""
        reply = self._load().reply(text=turn.text, audio=turn.audio, max_tokens=config.max_tokens)
        return to_audio_reply(reply)

    def _load(self) -> LFM2Audio:
        if self._model is None:
            logger.info("loading %s (backend=%s)", self.model_name, self._backend)
            self._model = LFM2Audio.from_pretrained(
                self.model_name,
                adapter=self._adapter,
                backend=self._backend,
                **({"system": self._system} if self._system else {}),  # type: ignore[arg-type]
                **self.model_args,
            )
        # Inspect keeps one provider per variant, so history from the previous
        # sample would leak into this one; every sample must start clean.
        self._model.reset()
        return self._model


@modelapi(name=ENDPOINT_PROVIDER)
class Lfm2EndpointAPI(SyncAudioModelAPI):
    """Serves a deployed variant to Inspect: ``--model lfm2-endpoint/<id>``."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] | None = None,
        config: GenerateConfig | None = None,
        **model_args: Any,  # noqa: ANN401 — upstream contract: -M passes free kwargs
    ) -> None:
        super().__init__(model_name, base_url, api_key, api_key_vars, config, **model_args)
        self._client = LiquidAudioClient(model_name, api_key=api_key)

    def reply_sync(self, turn: AudioTurn, config: GenerateConfig) -> AudioReply:
        """One turn through the serverless endpoint."""
        return to_audio_reply(self._client.invoke(text=turn.text, audio=turn.audio, max_tokens=config.max_tokens))

    def max_connections(self) -> int:
        """Independent workers: this is where concurrency actually pays."""
        return 4
