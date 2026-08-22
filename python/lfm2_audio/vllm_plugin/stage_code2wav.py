"""Stage 1 — code2wav : détokeniseur audio LFM2 (codes Mimi → PCM 24 kHz).

Réutilise ``liquid_audio.detokenizer.LFM2AudioDetokenizer`` (mêmes poids, dans
``audio_detokenizer/`` du checkpoint). Stage non-AR (LLM_GENERATION) alimenté
en chunks par le framework async_chunk : chaque payload contient
``codec_left_context_frames`` de contexte gauche pour des frontières de chunks
propres ; seuls les échantillons des nouvelles frames sont émis (sémantique
delta, cf. mimo_audio_code2wav).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from liquid_audio.detokenizer import LFM2AudioDetokenizer
from safetensors.torch import load_file
from torch import nn
from transformers import Lfm2Config
from vllm_omni.model_executor.models.output_templates import OmniOutput

from lfm2_audio.vllm_plugin.constants import (
    CODEBOOKS,
    END_OF_AUDIO_CODE,
    LEFT_CONTEXT_HEADER_MAGIC,
    SAMPLES_PER_FRAME,
)

logger = logging.getLogger(__name__)


class Lfm2AudioCode2Wav(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        # Construit dans load_weights (le chemin du checkpoint est requis).
        self.detokenizer: Any = None
        self._model_path: Path | None = None

    def set_model_path(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)

    def load_weights(self, weights, model_path: str | Path | None = None) -> set[str]:
        """Le détokeniseur vit dans ``audio_detokenizer/`` du checkpoint (config
        Lfm2 + safetensors propres) — chargé tel quel, comme le fait
        ``LFM2AudioProcessor.audio_detokenizer``."""

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
        # float32 : torch.polar (rotary freqs du forward) n'existe pas en Half
        # sur CPU, et le coût mémoire du détokeniseur est marginal.
        self.detokenizer = detok.float()
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

        if self.detokenizer is None:
            raise RuntimeError("detokenizer not loaded")

        if codes is None:
            codes = input_ids
        if codes is None or codes.numel() == 0:
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": None})

        # Découpe le batch par requête (comme mimo_audio_code2wav) : sous
        # concurrence, le runner concatène les tokens de plusieurs requêtes ;
        # un wav unique serait attribué à toutes les requêtes du batch.
        seq_token_counts = kwargs.get("seq_token_counts")
        if codes.dim() != 2 and isinstance(seq_token_counts, (list, tuple)) and len(seq_token_counts) > 1:
            flat, parts, off = codes.reshape(-1), [], 0
            for count in seq_token_counts:
                parts.append(flat[off : off + int(count)])
                off += int(count)
            wavs_per_req = [self._decode_request(p) for p in parts]
            return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": wavs_per_req})

        wav = self._decode_request(codes)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": wav if wav.numel() else None},
        )

    def _decode_request(self, codes: torch.Tensor) -> torch.Tensor:
        """Décode tous les segments (chunks) d'une requête en un wav (1, T) CPU."""
        wavs = []
        for segment, left_context in self._to_segments(codes):
            # retire les frames EOA (2048 hors vocabulaire du détokeniseur)
            keep = segment[0] != END_OF_AUDIO_CODE
            frames = segment[:, keep]
            if frames.shape[1] == 0:
                continue
            # garde-fou dummy/profile run de vLLM : ids arbitraires → clamp dans
            # le vocabulaire Mimi (sans effet sur de vrais codes, toujours < 2048)
            frames = frames.clamp_(0, END_OF_AUDIO_CODE - 1)

            # le détokeniseur est construit dans load_weights (CPU) : suit le
            # device des codes au premier appel
            if next(self.detokenizer.parameters()).device != frames.device:
                self.detokenizer = self.detokenizer.to(frames.device)
            wav = self.detokenizer(frames.unsqueeze(0))  # (1, T')
            if left_context > 0:
                wav = wav[:, left_context * SAMPLES_PER_FRAME :]
            if os.environ.get("LFM2_DEBUG_CHUNK"):
                logger.warning(
                    "[c2w] frames_in=%d left=%d frames_out=%d",
                    frames.shape[1],
                    left_context,
                    wav.shape[1] // SAMPLES_PER_FRAME,
                )
            wavs.append(wav)

        if not wavs:
            return torch.zeros((1, 0), dtype=torch.float32)
        return torch.cat(wavs, dim=1).reshape(1, -1).float().cpu()

    @staticmethod
    def _to_segments(codes: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
        """Normalise vers une liste de segments ((codebooks, T), left_context).

        Chaque payload de _build_payload est préfixé de deux tokens-en-tête
        hors vocabulaire Mimi (>= LEFT_CONTEXT_HEADER_MAGIC) : left_context_size
        puis nombre de frames du body. Sous charge, vLLM-Omni concatène
        plusieurs payloads d'une même requête avant que le stage 1 ne les
        consomme — la longueur encodée permet de re-découper chunk par chunk
        (chacun avec son propre left_context à tronquer).

        Compat : header à 1 token (ancien format, left seul) et flux sans
        header (left=0). Tolérant aux longueurs non multiples de 8
        (dummy/profile run de vLLM)."""
        codes = codes.to(torch.long)
        if codes.dim() == 2 and codes.shape[0] == CODEBOOKS:
            return [(codes, 0)]
        flat = codes.reshape(-1)
        segments: list[tuple[torch.Tensor, int]] = []
        i, n = 0, flat.numel()
        while i < n:
            left_context = 0
            if int(flat[i].item()) >= LEFT_CONTEXT_HEADER_MAGIC:
                left_context = int(flat[i].item()) - LEFT_CONTEXT_HEADER_MAGIC
                i += 1
                if i < n and int(flat[i].item()) >= LEFT_CONTEXT_HEADER_MAGIC:
                    n_frames = int(flat[i].item()) - LEFT_CONTEXT_HEADER_MAGIC
                    i += 1
                    end = min(i + n_frames * CODEBOOKS, n)
                else:  # ancien format : tout le reste appartient à ce segment
                    end = n
            else:
                end = n
            body = flat[i:end]
            i = end
            usable = (body.numel() // CODEBOOKS) * CODEBOOKS
            if usable != body.numel():
                logger.debug("trimming %d trailing codes (not a full frame)", body.numel() - usable)
            if usable:
                segments.append((body[:usable].view(-1, CODEBOOKS).T.contiguous(), left_context))
        return segments
