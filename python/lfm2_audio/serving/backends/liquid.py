"""``LiquidAudioBackend`` — implémentation de référence (PyTorch, batch = 1).

Plus lente que vLLM-Omni (re-prefill de tout le contexte à chaque tour), mais
c'est l'implémentation **de référence** : elle sert d'étalon de parité numérique
et reste utilisable là où vLLM-Omni ne s'installe pas.

Contrairement au backend vLLM, l'historique est tenu par le ``ChatState`` de
liquid-audio, qui garde son KV cache d'un tour à l'autre.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, Self

from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.ds.checkpoint import ResolvedCheckpoint
from lfm2_audio.ds.config import EngineConfig, GenerationConfig
from lfm2_audio.ds.reply import Reply
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)

_END_OF_AUDIO_CODE = 2048
_MIMI_WARMUP_STEPS = 5
_TEXT_END = "<|text_end|>"


class LiquidAudioBackend(LFM2Audio):
    """Backend liquid-audio : speech → speech natif, une requête à la fois."""

    backend_name = "liquid"

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        model: Any,
        processor: Any,
        system: str,
        generation: GenerationConfig | None = None,
    ) -> None:
        super().__init__(checkpoint, system=system, generation=generation)
        self._model = model
        self._processor = processor
        self._mimi = processor.mimi.eval()
        self._chat: Any = None
        self._warmup_detokenizer()
        self.reset()

    @classmethod
    def _build(
        cls,
        checkpoint: ResolvedCheckpoint,
        *,
        system: str,
        engine: EngineConfig | None,
        generation: GenerationConfig | None,
    ) -> Self:
        import torch
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor

        source = str(checkpoint.path)
        started = time.time()
        model = LFM2AudioModel.from_pretrained(source, device="cuda", dtype=torch.bfloat16).eval()
        processor = LFM2AudioProcessor.from_pretrained(source, device="cuda")

        if checkpoint.adapter is not None:
            cls._merge_adapter(model, checkpoint.adapter)

        logger.info("modèle liquid-audio prêt en %.0fs", time.time() - started)
        return cls(checkpoint, model=model, processor=processor, system=system, generation=generation)

    @staticmethod
    def _merge_adapter(model: Any, adapter: Any) -> None:
        """Injecte puis fusionne un adaptateur LoRA dans les poids de base."""
        from pathlib import Path

        from lfm2_audio.training.lora import (
            inject_lora,
            load_lora,
            load_lora_settings,
            merge_lora,
        )

        settings = load_lora_settings(str(adapter))
        inject_lora(model, settings)
        load_lora(model, Path(adapter) / "adapter_model.safetensors")
        merge_lora(model)
        logger.info("adaptateur LoRA fusionné depuis %s", adapter)

    def _warmup_detokenizer(self) -> None:
        """Chauffe le détokeniseur Mimi en mode streaming.

        Comme la démo officielle : décoder frame par frame pendant la génération
        rend la première frame audible en ~80 ms, au lieu d'attendre un décodage
        en bloc à la fin du tour.
        """
        import torch

        with self._mimi.streaming(1), torch.no_grad():
            for _ in range(_MIMI_WARMUP_STEPS):
                self._mimi.decode(torch.randint(_END_OF_AUDIO_CODE, (1, 8, 1), device="cuda"))

    # ------------------------------------------------------------------ #
    # Génération
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Repart d'un ``ChatState`` neuf, amorcé par le system prompt."""
        from liquid_audio import ChatState

        super().reset()
        self._chat = ChatState(self._processor)
        self._chat.new_turn("system")
        self._chat.add_text(self.system)
        self._chat.end_turn()

    def stream(
        self,
        *,
        text: str | None = None,
        audio: Any = None,
        max_tokens: int | None = None,
    ) -> Iterator[Waveform]:
        import torch

        waveform = self._coerce_audio(audio)
        if waveform is None and not text:
            message = "`text` et/ou `audio` sont requis."
            raise ValueError(message)

        self._open_user_turn(text, waveform)
        generation = self.generation.with_max_tokens(max_tokens)
        generate = self._model.generate_sequential if generation.text_only else self._model.generate_interleaved

        started = time.time()
        text_ids: list[Any] = []
        frames, ttfa = 0, None

        with torch.no_grad(), self._mimi.streaming(1):
            for token in generate(
                **self._chat,
                max_new_tokens=generation.max_tokens,
                audio_temperature=generation.audio_temperature,
                audio_top_k=generation.audio_top_k,
            ):
                if token.numel() == 1:
                    text_ids.append(token.detach().cpu())
                    continue
                frames += 1
                if bool((token == _END_OF_AUDIO_CODE).any()):
                    continue
                decoded = self._mimi.decode(token[None, :, None])[0]
                ttfa = ttfa if ttfa is not None else time.time() - started
                yield Waveform.of(decoded.float().cpu().numpy(), OUTPUT_SAMPLE_RATE)

        self._close_assistant_turn(text_ids, started, frames, ttfa)

    def _open_user_turn(self, text: str | None, waveform: Waveform | None) -> None:
        import torch

        self._chat.new_turn("user")
        if waveform is not None:
            samples, sample_rate = waveform.as_model_input()
            self._chat.add_audio(torch.as_tensor(samples, dtype=torch.float32).reshape(1, -1), sample_rate)
        if text:
            self._chat.add_text(text)
        self._chat.end_turn()
        self._chat.new_turn("assistant")

    def _close_assistant_turn(self, text_ids: list[Any], started: float, frames: int, ttfa: float | None) -> None:
        """Décode le texte et le réinjecte dans l'historique pour le tour suivant.

        Seul le texte est réinjecté — les frames audio sont omises, exactement
        comme côté vLLM, pour que les deux backends voient le même contexte.
        """
        raw_text = self._processor.text.decode([int(token) for token in text_ids])
        clean = raw_text.replace(_TEXT_END, "").strip()
        self._chat.add_text(clean)
        self._chat.end_turn()
        self._last_reply = Reply(
            text=clean,
            metrics=self._elapsed(started, frames, ttfa),
            raw_text=raw_text,
        )
