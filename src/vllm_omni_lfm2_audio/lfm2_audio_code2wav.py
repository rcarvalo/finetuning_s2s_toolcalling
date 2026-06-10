"""Stage 1 — code2wav : détokeniseur audio LFM2 (codes Mimi → PCM 24 kHz).

Réutilise ``liquid_audio.detokenizer.LFM2AudioDetokenizer`` (mêmes poids, dans
``audio_detokenizer/`` du checkpoint). Stage non-AR (LLM_GENERATION) alimenté
en chunks par le framework async_chunk : chaque payload contient
``codec_left_context_frames`` de contexte gauche pour des frontières de chunks
propres ; seuls les échantillons des nouvelles frames sont émis (sémantique
delta, cf. mimo_audio_code2wav).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vllm_omni_lfm2_audio.constants import CODEBOOKS, END_OF_AUDIO_CODE, SAMPLES_PER_FRAME

logger = logging.getLogger(__name__)


class Lfm2AudioCode2Wav(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.detokenizer = None  # chargé dans load_weights (chemin checkpoint requis)
        self._model_path: Path | None = None

    def set_model_path(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)

    def load_weights(self, weights, model_path: str | Path | None = None) -> set[str]:
        """Le détokeniseur vit dans ``audio_detokenizer/`` du checkpoint (config
        Lfm2 + safetensors propres) — chargé tel quel, comme le fait
        ``LFM2AudioProcessor.audio_detokenizer``."""
        from liquid_audio.detokenizer import LFM2AudioDetokenizer
        from safetensors.torch import load_file
        from transformers import Lfm2Config

        path = Path(model_path) if model_path is not None else self._model_path
        if path is None:
            raise RuntimeError("code2wav stage needs the checkpoint path (audio_detokenizer/)")
        detok_dir = path / "audio_detokenizer"

        detok_config = Lfm2Config.from_pretrained(detok_dir / "config.json")
        # compat layer_types llama.cpp → transformers (cf. LFM2AudioProcessor)
        detok_config.layer_types = [
            "full_attention" if layer == "sliding_attention" else layer for layer in detok_config.layer_types
        ]
        detok = LFM2AudioDetokenizer(detok_config).eval()
        detok.load_state_dict(load_file(detok_dir / "model.safetensors"))
        self.detokenizer = detok
        # contrat track_weights_loading : noms de paramètres du module
        return {f"detokenizer.{name}" for name, _ in detok.named_parameters()}

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        codes: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Signature alignée sur le runner vLLM (vu sur Colab : il appelle avec
        input_ids/positions/...) — les codes arrivent dans ``input_ids`` (ids
        plats, frames × codebooks) ou en kwarg ``codes`` (frames (8, T)).

        ``additional_information["left_context_size"]`` (frames) : préfixe de
        contexte décodé mais NON émis (sémantique delta du async_chunk).
        """
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        if self.detokenizer is None:
            raise RuntimeError("detokenizer not loaded")

        if codes is None:
            codes = input_ids
        if codes is None or codes.numel() == 0:
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": None})

        frames = self._to_frames(codes)
        # retire les frames EOA (2048 hors vocabulaire du détokeniseur)
        keep = frames[0] != END_OF_AUDIO_CODE
        frames = frames[:, keep]
        if frames.shape[1] == 0:
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": None})

        info = additional_information or {}
        left_context = int(info.get("left_context_size", 0))

        wav = self.detokenizer(frames.unsqueeze(0).to(next(self.detokenizer.parameters()).device))  # (1, T')
        if left_context > 0:
            wav = wav[:, left_context * SAMPLES_PER_FRAME :]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": wav.reshape(1, -1).float().cpu()},
        )

    @staticmethod
    def _to_frames(codes: torch.Tensor) -> torch.Tensor:
        """Normalise vers (codebooks, T)."""
        codes = codes.to(torch.long)
        if codes.dim() == 2 and codes.shape[0] == CODEBOOKS:
            return codes
        flat = codes.reshape(-1)
        if flat.numel() % CODEBOOKS != 0:
            raise ValueError(f"flat codes length {flat.numel()} is not a multiple of {CODEBOOKS}")
        return flat.view(-1, CODEBOOKS).T.contiguous()
