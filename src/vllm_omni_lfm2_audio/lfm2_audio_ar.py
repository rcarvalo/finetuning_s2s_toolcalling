"""Stage 0 — AR interleaved (« thinker-talker fusionné ») de LFM2.5-Audio.

Intégration vLLM suivant le pattern MiMo-Audio (``vllm_omni/model_executor/
models/mimo_audio``), adapté à l'architecture LFM2.5-Audio :

- backbone texte = ``Lfm2ForCausalLM`` de vLLM core (cache conv hybride géré
  par vLLM), instancié via ``init_vllm_registered_model`` ;
- entrée audio (micro) = mel-128 → ConformerEncoder + adaptateur (modules
  liquid-audio), exposée par l'interface multimodale standard ;
- steps audio : le token échantillonné est un PLACEHOLDER (cf. constants) —
  forcé via ``OmniOutput.next_token_id`` + masquage des logits ; les 8 codes
  Mimi sont produits par le depthformer (``audio_head.sample_frame``) et
  exportés step par step via ``multimodal_outputs["codes"]["audio"]`` ;
- l'embedding du step suivant d'une frame = somme des 8 embeddings, servi par
  un cache par requête (décode) ou reconstruit depuis
  ``multi_modal_data["audio_out_codes"]`` (prefill / recompute après
  préemption) ;
- la décision texte/audio de chaque step est REJOUÉE depuis la séquence d'ids
  du tour assistant courant (``modality.replay`` — fonction pure), jamais
  stockée seule : robuste à la préemption, compatible prefix caching.

NOTE runtime : les signatures suivent vllm-omni v0.22.0 (vérifiées sur le
package PyPI). La validation finale (parité greedy vs liquid-audio, phase P2
du plan) s'exécute sur GPU via tests/test_omni_parity.py.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import torch
from torch import nn

from vllm_omni_lfm2_audio.audio_head import Lfm2AudioHead
from vllm_omni_lfm2_audio.constants import TEXT_VOCAB_SIZE
from vllm_omni_lfm2_audio.modality import Modality, ModalityConfig, replay

logger = logging.getLogger(__name__)


class Lfm2AudioARForConditionalGeneration(nn.Module):
    """Stage AR interleaved. Interfaces vLLM ajoutées au runtime via le wrapper
    (SupportsPP/SupportsMultiModal sont déclarées sur la classe wrapper)."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

        config = vllm_config.model_config.hf_config
        self.config = config
        self.have_multimodal_outputs = True
        self.requires_raw_input_tokens = True

        # --- machine à états (ratio lu DANS le checkpoint, source unique) --- #
        self.modality_cfg = ModalityConfig(
            n_text=getattr(config, "interleaved_n_text", 6),
            n_audio=getattr(config, "interleaved_n_audio", 12),
            frame_placeholder_id=getattr(config, "audio_frame_token_id", 128),
            eoa_placeholder_id=getattr(config, "audio_eoa_token_id", 129),
        )

        # --- backbone Lfm2 (implémentation vLLM core, KV/conv cache géré) --- #
        lfm_cfg = config.lfm if hasattr(config, "lfm") else config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=lfm_cfg,
            prefix=maybe_prefix(prefix, "lfm"),
            architectures=["Lfm2ForCausalLM"],
        )

        # --- encodeur audio-in (modules liquid-audio, mêmes poids) --- #
        from liquid_audio.model.conformer.encoder import ConformerEncoder
        from liquid_audio.model.mlp import MLP

        encoder_cfg = dict(config.encoder) if isinstance(config.encoder, dict) else vars(config.encoder)
        self.conformer = ConformerEncoder(**encoder_cfg)
        hidden = lfm_cfg.hidden_size if hasattr(lfm_cfg, "hidden_size") else lfm_cfg["hidden_size"]
        self.audio_adapter = MLP(self.conformer._feat_out, hidden, [hidden])

        # --- tête audio-out (depthformer) --- #
        depth_cfg = dict(config.depthformer) if isinstance(config.depthformer, dict) else vars(config.depthformer)
        self.audio_head = Lfm2AudioHead(
            lfm_hidden_size=hidden,
            depthformer_layers=depth_cfg["layers"],
            depthformer_dim=depth_cfg["dim"],
            depthformer_tie=depth_cfg["tie"],
            codebooks=getattr(config, "codebooks", 8),
        )

        # sampling audio (prosodie) — défauts du démo liquid-audio
        self.audio_temperature: float | None = getattr(config, "audio_temperature", 1.0)
        self.audio_top_k: int | None = getattr(config, "audio_top_k", 4)

        # --- état par requête --- #
        # ids du tour assistant courant (pour replay de la modalité)
        self._turn_ids_by_req: dict[str, list[int]] = {}
        # embedding (somme des 8 codebooks) de la dernière frame générée,
        # à utiliser comme embedding d'entrée du prochain step de la requête
        self._frame_emb_by_req: dict[str, torch.Tensor] = {}
        # codes de la frame produite à CE step (exportés via OmniOutput)
        self._frame_codes_by_req: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Multimodal : audio-in (mel) et audio-out passé (codes)
    # ------------------------------------------------------------------ #

    def embed_multimodal(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """mel (128, T) → embeddings conformer+adapter pour les placeholders
        d'entrée audio du prompt (pattern Qwen2-Audio)."""
        mel = kwargs.get("input_audio_features")
        mel_lens = kwargs.get("input_audio_lens")
        if mel is None:
            return {}
        audio_enc, enc_lens = self.conformer(mel, mel_lens)
        len_mask = torch.arange(audio_enc.shape[-1], device=audio_enc.device).unsqueeze(0) < enc_lens.unsqueeze(1)
        return {"audio": self.audio_adapter(audio_enc.mT[len_mask])}

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
        request_ids: list[str] | None = None,
        audio_out_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embeddings d'entrée. Les positions placeholder audio sont remplacées :
        - prefill / recompute : depuis ``audio_out_codes`` (mm data, frames (N, 8)) ;
        - décode : depuis le cache de la dernière frame de la requête.
        """
        embeds = self.language_model.model.embed_tokens(input_ids)

        placeholder_mask = (input_ids == self.modality_cfg.frame_placeholder_id) | (
            input_ids == self.modality_cfg.eoa_placeholder_id
        )
        if placeholder_mask.any():
            if audio_out_codes is not None:  # prefill : reconstruit depuis les codes passés
                frames = audio_out_codes.to(embeds.device)
                frame_embeds = torch.stack([self.audio_head.embed_frame(f) for f in frames])
                embeds[placeholder_mask] = frame_embeds.to(embeds.dtype)
            elif request_ids is not None:  # décode : cache de la frame du step précédent
                for row, req_id in enumerate(request_ids):
                    if placeholder_mask[row].any() and req_id in self._frame_emb_by_req:
                        embeds[row][placeholder_mask[row]] = self._frame_emb_by_req[req_id].to(embeds.dtype)
            else:
                raise RuntimeError("audio placeholder without codes nor per-request cache")

        return embeds

    # ------------------------------------------------------------------ #
    # Forward / logits
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        request_ids: list[str] | None = None,
        **kwargs: Any,
    ):
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids, request_ids=request_ids)

        hidden = self.language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        # Décision de modalité du PROCHAIN token, par requête (replay pur).
        codes_payload: dict[str, torch.Tensor] = {}
        next_token_ids: dict[str, int] = {}
        if request_ids is not None and input_ids.dim() >= 1:
            self._track_turn_ids(input_ids, request_ids)
            for row, req_id in enumerate(request_ids):
                state = replay(self._turn_ids_by_req.get(req_id, []), self.modality_cfg)
                if state.current is Modality.AUDIO:
                    frame = self.audio_head.sample_frame(
                        hidden[row, -1].float(),
                        temperature=self.audio_temperature,
                        top_k=self.audio_top_k,
                    )
                    is_eoa = bool(frame[0].item() == 2048)
                    if is_eoa:
                        frame = torch.full_like(frame, 2048)
                    self._frame_codes_by_req[req_id] = frame
                    self._frame_emb_by_req[req_id] = self.audio_head.embed_frame(frame)
                    codes_payload[req_id] = frame
                    next_token_ids[req_id] = (
                        self.modality_cfg.eoa_placeholder_id if is_eoa else self.modality_cfg.frame_placeholder_id
                    )
                else:
                    self._frame_emb_by_req.pop(req_id, None)

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"codes": {"audio": codes_payload}} if codes_payload else None,
            next_token_id=next_token_ids or None,
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata=None) -> torch.Tensor:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def mask_logits_for_audio_steps(self, logits: torch.Tensor, request_ids: list[str]) -> torch.Tensor:
        """Défense en profondeur : pendant un step audio, seul le placeholder est
        échantillonnable (en plus de ``next_token_id`` qui dicte déjà l'id)."""
        for row, req_id in enumerate(request_ids):
            state = replay(self._turn_ids_by_req.get(req_id, []), self.modality_cfg)
            if state.current is Modality.AUDIO:
                masked = torch.full_like(logits[row], -float("inf"))
                masked[self.modality_cfg.frame_placeholder_id] = 0.0
                masked[self.modality_cfg.eoa_placeholder_id] = 0.0
                logits[row] = masked
        return logits

    # ------------------------------------------------------------------ #

    def _track_turn_ids(self, input_ids: torch.Tensor, request_ids: list[str]) -> None:
        """Maintient les ids du tour assistant courant par requête.

        Prefill (plusieurs ids) : reconstruit depuis le suffixe après le dernier
        début de tour assistant. Décode (1 id) : append. Toute requête inconnue
        (préemption/recompute) repart du prompt complet — l'état reste correct
        car ``replay`` est une fonction pure des ids.
        """
        for row, req_id in enumerate(request_ids):
            ids = input_ids[row] if input_ids.dim() > 1 else input_ids
            ids_list = ids.tolist() if ids.dim() > 0 else [int(ids.item())]
            if len(ids_list) > 1 or req_id not in self._turn_ids_by_req:
                self._turn_ids_by_req[req_id] = self._extract_current_assistant_turn(
                    self._turn_ids_by_req.get(req_id, []) + ids_list
                )
            else:
                self._turn_ids_by_req[req_id].append(ids_list[0])

    @staticmethod
    def _extract_current_assistant_turn(ids: list[int]) -> list[int]:
        """Suffixe après le dernier ``<|im_end|>`` (id 7) : le tour en cours."""
        from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID

        try:
            last_end = len(ids) - 1 - ids[::-1].index(IM_END_TOKEN_ID)
            return ids[last_end + 1 :]
        except ValueError:
            return ids

    def free_request(self, request_id: str) -> None:
        """Libère l'état d'une requête terminée/annulée (appelé par le runner)."""
        self._turn_ids_by_req.pop(request_id, None)
        self._frame_emb_by_req.pop(request_id, None)
        self._frame_codes_by_req.pop(request_id, None)

    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Répartit le state dict liquid-audio (préfixes lfm./conformer./
        audio_adapter./depthformer./depth_*/audio_embedding.)."""
        weights = dict(weights)
        loaded: set[str] = set()

        lfm_weights = [("model." + k[len("lfm.") :], v) for k, v in weights.items() if k.startswith("lfm.")]
        self.language_model.load_weights(lfm_weights)
        loaded.update(k for k in weights if k.startswith("lfm."))

        enc_state = {k[len("conformer.") :]: v for k, v in weights.items() if k.startswith("conformer.")}
        self.conformer.load_state_dict(enc_state, strict=True)
        loaded.update(k for k in weights if k.startswith("conformer."))

        adapter_state = {k[len("audio_adapter.") :]: v for k, v in weights.items() if k.startswith("audio_adapter.")}
        self.audio_adapter.load_state_dict(adapter_state, strict=True)
        loaded.update(k for k in weights if k.startswith("audio_adapter."))

        loaded.update(self.audio_head.load_weights(weights))
        return loaded
