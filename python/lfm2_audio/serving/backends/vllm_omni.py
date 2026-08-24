"""``VllmOmniBackend`` — inférence S2S basse latence sur vLLM-Omni.

Deux entrées de génération, sur la même mécanique :

- :meth:`stream` — dialogue simple, l'historique est tenu par le backend ;
- :meth:`stream_turns` — l'appelant fournit les tours et les points d'arrêt.
  C'est ce dont l'orchestrateur tool-calling a besoin pour interrompre la
  génération sur ``<|tool_call_end|>`` puis reprendre après réinjection.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from typing import Any, Self, cast

from transformers import AutoTokenizer

from lfm2_audio.core.prompt import ChatMLRenderer, strip_special_tokens
from lfm2_audio.core.tokenizer import Tokenizer
from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.ds.checkpoint import ResolvedCheckpoint
from lfm2_audio.ds.conversation import Conversation, ConversationTurn
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.ds.reply import Reply
from lfm2_audio.serving.backends.omni_engine import OmniEngine
from lfm2_audio.serving.model import LFM2Audio
from lfm2_audio.vllm_plugin.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    IM_END_TOKEN_ID,
)

logger = logging.getLogger(__name__)

_TOOL_CALL_END = "<|tool_call_end|>"


class VllmOmniBackend(LFM2Audio):
    """Backend vLLM-Omni 2 stages (AR interleaved → détokeniseur)."""

    backend_name = "vllm"

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        engine: OmniEngine,
        renderer: ChatMLRenderer,
        system: str,
        generation: GenerationConfig | None = None,
    ) -> None:
        super().__init__(checkpoint, system=system, generation=generation)
        self._engine = engine
        self._renderer = renderer
        self._frame_ids = self._load_frame_ids()
        self.im_end_id = self._frame_ids["im_end"]
        self.tool_call_end_id = renderer.single_token_id(_TOOL_CALL_END)

    @classmethod
    def _build(
        cls,
        checkpoint: ResolvedCheckpoint,
        *,
        system: str,
        engine: EngineConfig | None,
        generation: GenerationConfig | None,
    ) -> Self:

        # transformers' return type union does not structurally match our
        # Tokenizer protocol, but every HF tokenizer satisfies it at runtime.
        tokenizer = cast("Tokenizer", AutoTokenizer.from_pretrained(str(checkpoint.path)))
        return cls(
            checkpoint,
            engine=OmniEngine(checkpoint, engine),
            renderer=ChatMLRenderer(tokenizer, audio_placeholder_id=AUDIO_FRAME_PLACEHOLDER_ID),
            system=system,
            generation=generation,
        )

    @staticmethod
    def _load_frame_ids() -> dict[str, int]:

        return {
            "frame": AUDIO_FRAME_PLACEHOLDER_ID,
            "eoa": AUDIO_EOA_PLACEHOLDER_ID,
            "im_end": IM_END_TOKEN_ID,
        }

    # ------------------------------------------------------------------ #
    # Génération
    # ------------------------------------------------------------------ #

    def stream(
        self,
        *,
        text: str | None = None,
        audio: Any = None,
        max_tokens: int | None = None,
    ) -> Iterator[Waveform]:
        """Un tour de dialogue ; l'historique du backend est mis à jour."""
        waveform = self._coerce_audio(audio)
        if waveform is None and not text:
            message = "`text` et/ou `audio` sont requis."
            raise ValueError(message)

        # L'audio va à l'encodeur : 16 kHz obligatoire (le mel y est calibré).
        self.conversation.add("user", text=text or "", audio=waveform.for_encoder() if waveform else None)
        yield from self._run(list(self.conversation), stop_token_ids=[self.im_end_id], max_tokens=max_tokens)

        # L'audio user est consommé : les tours suivants le porteront en texte.
        self.conversation.release_audio()
        reply = self._last_reply
        self.conversation.add("assistant", text=reply.text if reply else "")

    def stream_turns(
        self,
        turns: Sequence[ConversationTurn],
        *,
        stop_token_ids: Sequence[int],
        max_tokens: int | None = None,
    ) -> Iterator[Waveform]:
        """Génération pilotée par l'appelant (orchestrateur tool-calling).

        L'historique du backend n'est pas touché : c'est l'orchestrateur qui
        décide quels tours composent le prompt de chaque passe.
        """
        Conversation.from_turns(turns)  # valide l'invariant « un seul audio »
        yield from self._run(turns, stop_token_ids=list(stop_token_ids), max_tokens=max_tokens)

    def _run(
        self,
        turns: Sequence[ConversationTurn],
        *,
        stop_token_ids: list[int],
        max_tokens: int | None,
    ) -> Iterator[Waveform]:
        """Boucle de génération commune : rendu, appel engine, collecte."""
        generation = self.generation.with_max_tokens(max_tokens)
        prompt = self._renderer.render(turns, system=self.system)
        sampling = self._engine.sampling_pair(
            max_tokens=generation.max_tokens,
            temperature=generation.temperature,
            stop_token_ids=stop_token_ids,
        )

        started = time.time()
        raw_text, frames, ttfa = "", 0, None

        for output in self._engine.generate(prompt.as_vllm_prompt(), sampling):
            if output.final_output_type == "text":
                raw_text, frames = self._read_text(output, fallback=raw_text)
            elif output.final_output_type == "audio":
                chunk = self._read_audio(output)
                if chunk is not None:
                    ttfa = ttfa if ttfa is not None else time.time() - started
                    yield chunk

        self._last_reply = Reply(
            text=strip_special_tokens(raw_text),
            metrics=self._elapsed(started, frames, ttfa),
            raw_text=raw_text,
        )
        self._log_outcome(frames, ttfa)

    # ------------------------------------------------------------------ #
    # Lecture des sorties
    # ------------------------------------------------------------------ #

    def _read_text(self, output: Any, *, fallback: str) -> tuple[str, int]:
        """Texte du stage 0 + nombre de frames audio qu'il a émises."""
        request_output = output.request_output
        if not request_output or not request_output.outputs:
            return fallback, 0
        completion = request_output.outputs[0]
        token_ids = list(completion.token_ids or [])
        frames = sum(1 for token in token_ids if token in (self._frame_ids["frame"], self._frame_ids["eoa"]))
        return (completion.text or fallback), frames

    @staticmethod
    def _read_audio(output: Any) -> Waveform | None:
        """Chunk audio du stage 1, s'il en porte un."""
        payload = getattr(output, "multimodal_output", None) or getattr(
            output.request_output, "multimodal_output", None
        )
        samples = _extract_samples(payload)
        if samples is None or samples.size == 0:
            return None
        return Waveform.of(samples, OUTPUT_SAMPLE_RATE)

    @staticmethod
    def _log_outcome(frames: int, ttfa: float | None) -> None:
        """Distingue les deux pannes qui se ressemblent en sortie."""
        if frames and ttfa is None:
            logger.warning(
                "%d frames audio émises par le stage 0 mais aucun chunk reçu du stage 1 "
                "→ plomberie connector/stage 1, pas le modèle.",
                frames,
            )
        elif not frames:
            logger.info("aucune frame audio émise (pas de <|text_end|>) → prompt ou modèle, pas la plomberie.")

    def close(self) -> None:
        self._engine.close()


def _extract_samples(payload: Any) -> Any:
    """Déballe le waveform d'un ``multimodal_output`` (dict imbriqué, tenseur, tableau).

    ``torch`` n'est pas importé : un tenseur est reconnu par duck typing.
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        for key in ("model_outputs", "audio", "waveform", "wav"):
            if key in payload:
                return _extract_samples(payload[key])
        return None
    if hasattr(payload, "detach"):
        return payload.detach().float().cpu().numpy().reshape(-1)
    return payload
