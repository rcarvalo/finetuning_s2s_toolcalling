"""Stage 0 — AR interleaved (« thinker-talker fusionné ») de LFM2.5-Audio.

Architecture v2, calée sur les hooks RÉELS du runner vllm-omni 0.22.0
(vérifiés dans gpu_model_runner/gpu_ar_model_runner, validés sur Colab) :

- ``preprocess_batch(req_ids, …)``   : appelé 1×/step AVANT le forward avec la
  liste ordonnée des requêtes du batch → fixe le mapping ligne→requête ;
- ``preprocess(input_ids, input_embeds, request_id, …)`` : appelé par requête
  avec son span de tokens → construit les embeddings du span et OVERLAYE
  l'embedding de frame (somme des 8 codebooks, cache par requête) sur les
  positions placeholder — c'est ici que vit l'identité de requête (§3bis.3) ;
- ``compute_logits(hidden)``         : stash du hidden (aligné 1:1 avec les
  lignes de sample) + logits texte du backbone Lfm2 ;
- ``sample(logits, sampling_metadata)`` (``prefer_model_sampler=True``) :
  modalité rejouée par ligne depuis ``output_token_ids`` (fonction pure de
  ``modality.replay``) ; lignes TEXT → sampling texte standard ; lignes AUDIO
  → rollout du depthformer sur le hidden stashé → 8 codes Mimi, id émis =
  placeholder (frame ou EOA), embedding mis en cache pour le preprocess du
  step suivant ;
- ``forward``                        : backbone Lfm2 pur ; exporte les frames
  en attente via ``OmniOutput.multimodal_outputs`` (payload sparse par req_id).

Les mécanismes de la v1 (``OmniOutput.next_token_id``, kwarg ``request_ids``)
n'existent pas dans le runtime — voir §3bis de docs/vllm_omni_integration.md.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

import torch
from torch import nn

from vllm_omni_lfm2_audio.audio_head import Lfm2AudioHead
from vllm_omni_lfm2_audio.constants import IM_END_TOKEN_ID
from vllm_omni_lfm2_audio.modality import Modality, ModalityConfig, replay

logger = logging.getLogger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TRACKED_REQUESTS = 1024  # garde-fou mémoire si free_request n'est pas appelé


class Lfm2AudioARForConditionalGeneration(nn.Module):
    """Stage AR interleaved (hooks runner exposés via le wrapper)."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

        config = vllm_config.model_config.hf_config
        self.config = config

        # Garde-fou : vLLM 0.22 active l'async scheduling PAR DÉFAUT, et notre
        # sample() rejoue la modalité depuis output_token_ids — en async, le
        # token in-flight remplace le placeholder forcé → replay désynchronisé
        # (« text token encountered during an AUDIO step » en pleine
        # génération). On échoue AU DÉMARRAGE avec la consigne, pas au 8e step.
        if getattr(getattr(vllm_config, "scheduler_config", None), "async_scheduling", False):
            raise RuntimeError(
                "Lfm2AudioOmniModel : async_scheduling doit être désactivé sur le "
                "stage 0 (replay de modalité tronqué d'un token in-flight — écart 5, "
                "action 6 de docs/optimization_audit.md). Ajouter "
                "`async_scheduling: false` au stage 0 du deploy YAML."
            )

        # Même logique pour les FULL CUDA graphs : la capture englobe notre
        # forward externe → pendant un replay le Python ne tourne plus et
        # multimodal_outputs reste le dict vide de la capture (aucun code
        # audio n'atteint le stage 1). PIECEWISE garde le forward externe en
        # Python à chaque step (seuls les segments compilés du backbone
        # rejouent en graph).
        cg_mode = getattr(getattr(vllm_config, "compilation_config", None), "cudagraph_mode", None)
        if cg_mode is not None and "FULL" in getattr(cg_mode, "name", str(cg_mode)).upper():
            raise RuntimeError(
                f"Lfm2AudioOmniModel : cudagraph_mode={cg_mode} capture le forward "
                "externe — l'export audio (multimodal_outputs) est gelé au contenu "
                "de la capture (vide). Mettre dans le stage 0 du deploy YAML :\n"
                "    compilation_config:\n"
                '      cudagraph_mode: "PIECEWISE"\n'
                "ou enforce_eager: true."
            )

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
            audio_tie=getattr(config, "tie_audio_embeddings", False),
        )

        # sampling audio (prosodie) — défaut GREEDY (None) pour la parité avec
        # generate_interleaved (qui est greedy par défaut) ; un sampling trop
        # chaud empêche l'émission de l'EOA → boucle/sur-génération. Surchargeable
        # via le config (audio_temperature/audio_top_k) une fois la parité tenue.
        self.audio_temperature: float | None = getattr(config, "audio_temperature", None)
        self.audio_top_k: int | None = getattr(config, "audio_top_k", None)

        # --- état par step (fixé par preprocess_batch/preprocess) --- #
        self._step_req_ids: list[str] = []
        self._step_sampling_reqs: list[str] = []
        self._pending_hidden: torch.Tensor | None = None  # stash de compute_logits

        # --- état par requête --- #
        # embedding (somme des 8 codebooks) de la dernière frame générée,
        # consommé par preprocess() au step suivant
        self._frame_emb_by_req: dict[str, torch.Tensor] = {}
        # frames générées non encore exportées vers le stage 1
        self._pending_codes_by_req: dict[str, list[torch.Tensor]] = {}
        # historique COMPLET par requête, pour debug (dump si LFM2_DUMP_FRAMES)
        self._all_frames_by_req: dict[str, list[torch.Tensor]] = {}
        # compteurs de profiling du sampler (LFM2_DEBUG_TIMING)
        self._tim_n = self._tim_rows = 0
        self._tim_replay = self._tim_audio = self._tim_text = 0.0

    # ------------------------------------------------------------------ #
    # Hooks runner : preprocess (identité de requête + embeddings)
    # ------------------------------------------------------------------ #

    def preprocess_batch(self, *, req_ids: list[str], model_intermediate_buffer=None, device=None) -> None:
        """1×/step, avant les preprocess par requête : ordre du batch."""
        self._step_req_ids = list(req_ids)
        self._step_sampling_reqs = []

    def preprocess(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        request_id: str = "",
        _omni_is_prefill: bool = False,
        _omni_prompt_len: int = 0,
        _omni_num_computed_tokens: int = 0,
        **infos: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Par requête : embeddings du span + overlay des placeholders audio.

        Cette requête échantillonnera ce step sauf si c'est un chunk de prefill
        partiel — c'est ce qui aligne ``_step_sampling_reqs`` avec les lignes
        de ``sample()``.
        """
        span_len = int(input_ids.shape[0])
        samples_this_step = (not _omni_is_prefill) or (_omni_num_computed_tokens + span_len >= _omni_prompt_len)
        if samples_this_step:
            self._step_sampling_reqs.append(request_id)

        # Audio-in : en prefill le runner fournit input_embeds DÉJÀ mergés par
        # le chemin mm standard (embed_input_ids aux positions placeholder) —
        # ré-embedder écraserait les embeddings du conformer.
        if _omni_is_prefill and input_embeds is not None and input_embeds.shape[0] == span_len:
            embeds = input_embeds
        else:
            embeds = self.language_model.model.embed_tokens(input_ids)

        if span_len == 1:  # décode : un placeholder porte l'embedding de SA frame
            token_id = int(input_ids[0].item())
            if token_id in (self.modality_cfg.frame_placeholder_id, self.modality_cfg.eoa_placeholder_id):
                frame_emb = self._frame_emb_by_req.get(request_id)
                if frame_emb is not None:
                    embeds[0] = frame_emb.to(dtype=embeds.dtype, device=embeds.device)
                else:  # préemption/recompute : cache perdu (cf. risques de la spec)
                    logger.warning("frame embedding cache miss for request %s", request_id)

        return input_ids, embeds, {}

    # ------------------------------------------------------------------ #
    # Multimodal : audio-in (mel) — pattern standard vLLM
    # ------------------------------------------------------------------ #

    def embed_multimodal(self, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """mm_kwargs du processor → embeddings par item audio.

        Contrat ``MultiModalEmbeddings`` (vLLM 0.22) : une séquence de tenseurs
        ``(n_i, hidden)``, un par audio, dans l'ordre des items. Chemin de
        référence : pad → conformer (B, 128, T) → masque par longueurs →
        adapter — identique à ``LFM2AudioModel`` (liquid)."""
        mel = kwargs.get("input_audio_features")
        mel_lens = kwargs.get("input_audio_lens")
        if mel is None:
            return ()

        device = self.depthformer_device()
        dtype = self.audio_adapter_dtype()
        if isinstance(mel, torch.Tensor) and mel.dim() == 2:
            mel = [mel]
        items = [m.to(device=device, dtype=dtype) for m in mel]
        if mel_lens is None:
            lens = torch.tensor([m.shape[-1] for m in items], device=device, dtype=torch.long)
        else:
            lens = torch.as_tensor(mel_lens, device=device, dtype=torch.long).reshape(-1)

        # pad (B, 128, T_max) — pad_sequence travaille sur (T, 128)
        padded = torch.nn.utils.rnn.pad_sequence([m.mT for m in items], batch_first=True).mT
        audio_enc, enc_lens = self.conformer(padded, lens)  # (B, D, T_out)
        len_mask = torch.arange(audio_enc.shape[-1], device=device).unsqueeze(0) < enc_lens.unsqueeze(1)
        flat = self.audio_adapter(audio_enc.mT[len_mask])  # (sum n_i, hidden)
        return tuple(flat.split([int(n) for n in enc_lens.tolist()]))

    def depthformer_device(self) -> torch.device:
        return self.audio_head.depth_linear.weight.device

    def audio_adapter_dtype(self) -> torch.dtype:
        return next(self.audio_adapter.parameters()).dtype

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        embeds = self.language_model.model.embed_tokens(input_ids)
        if multimodal_embeddings is not None and is_multimodal is not None:
            if isinstance(multimodal_embeddings, (list, tuple)):
                if not multimodal_embeddings:
                    return embeds
                multimodal_embeddings = torch.cat(list(multimodal_embeddings))
            embeds[is_multimodal] = multimodal_embeddings.to(dtype=embeds.dtype, device=embeds.device)
        return embeds

    # ------------------------------------------------------------------ #
    # Forward / logits / sampling
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden = self.language_model.model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=self._drain_pending_codes(),
        )

    def _drain_pending_codes(self) -> dict[str, Any]:
        """Frames produites par sample() au(x) step(s) précédent(s), exportées
        en payload sparse par req_id (consommé par stage_input_processors).

        Toujours un dict ({} si rien) : avec enable_prefix_caching, le chemin
        omni_prefix_cache du runner fait ``multimodal_outputs.keys()`` sans
        garde None (prefix_cache.get_merged_multimodal_states, vllm-omni
        0.22.0) — None crash le step de prefill."""
        req_ids, codes = [], []
        for req_id in self._step_req_ids:
            frames = self._pending_codes_by_req.pop(req_id, None)
            if frames:
                req_ids.append(req_id)
                codes.append(torch.stack(frames))  # (n_frames, codebooks)
        if not req_ids:
            return {}
        return {"codes": {"audio": codes}, "meta": {"req_id": req_ids, "sparse_audio": True}}

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata=None) -> torch.Tensor:
        # hidden gathered aux logits_indices : alignement 1:1 avec sample()
        self._pending_hidden = hidden_states
        return self.language_model.compute_logits(hidden_states)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Sampler custom (``prefer_model_sampler``) : machine à états interleaved.

        Retourner ``None`` délègue tout au sampler standard (utilisé si le
        mapping ligne→requête n'est pas fiable ce step).
        """
        from vllm.v1.outputs import SamplerOutput

        if logits is None or logits.numel() == 0:
            return None

        n_rows = int(logits.shape[0])
        # mapping ligne→requête : selon le scheduler, les logits couvrent soit
        # toutes les requêtes du batch, soit seulement celles qui échantillonnent
        if len(self._step_req_ids) == n_rows:
            reqs = self._step_req_ids
        elif len(self._step_sampling_reqs) == n_rows:
            reqs = self._step_sampling_reqs
        else:
            # dummy run / profiling, ou divergence de plomberie : sampler standard
            if self._step_req_ids:
                logger.warning(
                    "sample(): %d rows vs %d batch reqs / %d sampling reqs — falling back",
                    n_rows, len(self._step_req_ids), len(self._step_sampling_reqs),
                )
            return None

        timing = os.environ.get("LFM2_DEBUG_TIMING")
        t0 = time.perf_counter() if timing else 0.0

        audio_rows: list[int] = []
        text_rows: list[int] = []
        for row in range(n_rows):
            history = (
                sampling_metadata.output_token_ids[row]
                if row < len(sampling_metadata.output_token_ids)
                else []
            )
            state = replay(self._turn_suffix(history), self.modality_cfg)
            (audio_rows if state.current is Modality.AUDIO else text_rows).append(row)
        t1 = time.perf_counter() if timing else 0.0

        sampled: list[int] = [0] * n_rows
        if audio_rows:
            sampled_audio = self._sample_audio_rows(audio_rows, [reqs[r] for r in audio_rows])
            for row, tok in zip(audio_rows, sampled_audio):
                sampled[row] = tok
        t2 = time.perf_counter() if timing else 0.0
        for row in text_rows:
            sampled[row] = self._sample_text_row(logits[row], row, sampling_metadata)

        if timing:
            t3 = time.perf_counter()
            self._tim_n += 1
            self._tim_rows += n_rows
            self._tim_replay += t1 - t0
            self._tim_audio += t2 - t1
            self._tim_text += t3 - t2
            if self._tim_n % 100 == 0:
                logger.warning(
                    "[timing] steps=%d rows/step=%.1f replay=%.2fms audio=%.2fms text=%.2fms (moy/step)",
                    self._tim_n, self._tim_rows / self._tim_n,
                    1e3 * self._tim_replay / self._tim_n,
                    1e3 * self._tim_audio / self._tim_n,
                    1e3 * self._tim_text / self._tim_n,
                )

        out = torch.tensor(sampled, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=out.unsqueeze(-1), logprobs_tensors=None)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _turn_suffix(history: list[int]) -> list[int]:
        """Ids du tour assistant courant : suffixe après le dernier <|im_end|>."""
        try:
            last_end = len(history) - 1 - history[::-1].index(IM_END_TOKEN_ID)
            return history[last_end + 1 :]
        except ValueError:
            return history

    def _sample_audio_rows(self, rows: list[int], request_ids: list[str]) -> list[int]:
        """Rollout depthformer batché sur les lignes audio du step ; émet les placeholders."""
        assert self._pending_hidden is not None, "compute_logits must run before sample"
        frames = self.audio_head.sample_frames(
            self._pending_hidden[rows],
            temperature=self.audio_temperature,
            top_k=self.audio_top_k,
        )  # (B, codebooks)
        is_eoa = frames[:, 0] == 2048
        if bool(is_eoa.any()):
            frames[is_eoa] = 2048  # frame EOA pleine (rejouable par replay)
        embs = self.audio_head.embed_frames(frames)
        frames_cpu = frames.cpu()

        dump = bool(os.environ.get("LFM2_DUMP_FRAMES"))
        sampled: list[int] = []
        for j, request_id in enumerate(request_ids):
            self._frame_emb_by_req[request_id] = embs[j]
            self._pending_codes_by_req.setdefault(request_id, []).append(frames_cpu[j])
            if dump:
                self._all_frames_by_req.setdefault(request_id, []).append(frames_cpu[j])
            sampled.append(
                self.modality_cfg.eoa_placeholder_id
                if bool(is_eoa[j])
                else self.modality_cfg.frame_placeholder_id
            )
        self._evict_stale_requests()
        return sampled

    def _sample_text_row(self, row_logits: torch.Tensor, row: int, md) -> int:
        """Sampling texte standard pour une ligne (greedy / temp / top-k / top-p)."""
        temperature = self._req_scalar(md.temperature, row, 0.0)
        if getattr(md, "all_greedy", False) or temperature < _SAMPLING_EPS:
            return int(row_logits.argmax().item())

        logits = row_logits.to(torch.float32) / temperature
        top_k = int(self._req_scalar(md.top_k, row, 0))
        if 0 < top_k < logits.shape[-1]:
            min_kept = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < min_kept, -float("inf"))
        top_p = float(self._req_scalar(md.top_p, row, 1.0))
        if top_p < 1.0 - _SAMPLING_EPS:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cumprobs = sorted_logits.softmax(-1).cumsum(-1)
            cutoff = int((cumprobs <= top_p).sum().item()) + 1
            keep = sorted_idx[:cutoff]
            mask = torch.full_like(logits, -float("inf"))
            mask[keep] = logits[keep]
            logits = mask
        generator = md.generators.get(row) if getattr(md, "generators", None) else None
        probs = logits.softmax(-1)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    @staticmethod
    def _req_scalar(values, row: int, default: float) -> float:
        if values is None:
            return default
        if isinstance(values, torch.Tensor):
            return float(values[row].item()) if values.numel() > row else default
        return float(values)

    def _evict_stale_requests(self) -> None:
        """Garde-fou si free_request n'est jamais appelé par le runner."""
        for cache in (self._frame_emb_by_req, self._pending_codes_by_req):
            while len(cache) > _MAX_TRACKED_REQUESTS:
                cache.pop(next(iter(cache)))

    def on_requests_finished(self, req_ids: Iterable[str]) -> None:
        """Hook appelé par GPUARModelRunner (``scheduler_output.finished_req_ids``)
        — c'est le point de nettoyage d'état par requête du runtime, pas
        ``free_request`` (jamais invoqué côté worker). Délègue au cleanup +
        dump existant."""
        for req_id in list(req_ids):
            self.free_request(req_id)

    def free_request(self, request_id: str) -> None:
        self._frame_emb_by_req.pop(request_id, None)
        self._pending_codes_by_req.pop(request_id, None)
        frames = self._all_frames_by_req.pop(request_id, None)
        dump = os.environ.get("LFM2_DUMP_FRAMES")
        if dump and frames:
            safe = request_id.replace("/", "_")
            torch.save(torch.stack(frames), f"{dump}_{safe}.pt")
            logger.info("dumped %d frames for %s", len(frames), request_id)

    # ------------------------------------------------------------------ #

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Répartit le state dict liquid-audio (préfixes lfm./conformer./
        audio_adapter./depthformer./depth_*/audio_embedding.).

        Retourne les noms de PARAMÈTRES de CE module (contrat
        ``track_weights_loading`` du loader vLLM).
        """
        weights = dict(weights)
        loaded: set[str] = set()

        lfm_weights = [("model." + k[len("lfm.") :], v) for k, v in weights.items() if k.startswith("lfm.")]
        lfm_loaded = self.language_model.load_weights(lfm_weights) or set()
        loaded.update(f"language_model.{name}" for name in lfm_loaded)

        enc_state = {k[len("conformer.") :]: v for k, v in weights.items() if k.startswith("conformer.")}
        self.conformer.load_state_dict(enc_state, strict=True)
        loaded.update(f"conformer.{name}" for name, _ in self.conformer.named_parameters())

        adapter_state = {k[len("audio_adapter.") :]: v for k, v in weights.items() if k.startswith("audio_adapter.")}
        self.audio_adapter.load_state_dict(adapter_state, strict=True)
        loaded.update(f"audio_adapter.{name}" for name, _ in self.audio_adapter.named_parameters())

        self.audio_head.load_weights(weights)
        loaded.update(f"audio_head.{name}" for name in self.audio_head.loaded_param_names())
        return loaded
