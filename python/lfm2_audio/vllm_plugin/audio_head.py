"""Tête audio : depthformer (code predictor) + embeddings de frames.

Réutilise les modules de ``liquid_audio`` (mêmes classes, mêmes poids → parité
numérique). Deux opérations :

- ``sample_frame``  : hidden state du backbone → 8 codes Mimi (rollout AR du
  depthformer sur les 8 codebooks) — port de
  ``LFM2AudioModel._sample_audio_frame`` ;
- ``embed_frame``   : 8 codes → embedding d'entrée du step suivant (somme des
  8 embeddings offsetés) — port de la branche audio de ``generate_interleaved``.
"""

from __future__ import annotations

import math

import torch
from liquid_audio.model.transformer import (
    MHA,
    RawLMBackbone,
    SharedEmbedding,
    StandardBlock,
)
from torch import nn


class Lfm2AudioHead(nn.Module):
    """Sous-ensemble audio-out de LFM2AudioModel (poids partagés à l'identique)."""

    def __init__(
        self,
        *,
        lfm_hidden_size: int,
        depthformer_layers: int,
        depthformer_dim: int,
        depthformer_tie: bool,
        codebooks: int = 8,
        audio_vocab_size: int = 2049,
        audio_tie: bool = False,
    ) -> None:
        super().__init__()
        # SharedEmbedding vit dans model.transformer (vérifié sur liquid-audio 1.3.0)

        self.codebooks = codebooks
        self.audio_vocab_size = audio_vocab_size
        self.depthformer_dim = depthformer_dim

        # Mêmes constructions que LFM2AudioModel.__init__ (noms de poids identiques).
        self.audio_embedding = SharedEmbedding(
            dim=lfm_hidden_size,
            vocab_size=audio_vocab_size * codebooks,
            embed_init_scale=1.0,
            norm_eps=0.00001,
            tie_embedding=audio_tie,
        )
        scale = 1 / math.sqrt(2 * depthformer_layers)
        layers = [
            StandardBlock(MHA(depthformer_dim, out_init_scale=scale), out_init_scale=scale)
            for _ in range(depthformer_layers)
        ]
        self.depthformer = RawLMBackbone(layers, has_embedding=False)
        self.depth_linear = nn.Linear(lfm_hidden_size, depthformer_dim * codebooks)
        self.depth_embeddings = nn.ModuleList(
            [
                SharedEmbedding(dim=depthformer_dim, vocab_size=audio_vocab_size, tie_embedding=depthformer_tie)
                for _ in range(codebooks)
            ]
        )
        self.register_buffer("codebook_offsets", torch.arange(codebooks) * audio_vocab_size, persistent=False)

    @torch.no_grad()
    def sample_frame(
        self,
        hidden: torch.Tensor,  # (hidden_size,)
        *,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Rollout AR du depthformer → (codebooks,) codes Mimi.

        Port fidèle de ``LFM2AudioModel._sample_audio_frame``.
        """
        greedy = temperature is None or temperature <= 0 or top_k == 1
        hidden = hidden.to(self.depth_linear.weight.dtype)  # suit le dtype du modèle (fp16/bf16)
        depthformer_in = self.depth_linear(hidden).view(self.codebooks, self.depthformer_dim)
        depthformer_token = torch.zeros_like(depthformer_in[0])
        cache = None

        out_tokens: list[torch.Tensor] = []
        for i in range(self.codebooks):
            cur_input = depthformer_in[i] + depthformer_token
            depthformer_out, cache = self.depthformer.forward_cached(cur_input[None, None, :], cache)
            logits = self.depth_embeddings[i].get_logits(depthformer_out.squeeze())

            if greedy:
                next_token = logits.argmax(keepdim=True)
            else:
                logits = logits / float(temperature or 1.0)
                if top_k is not None:
                    min_score = torch.topk(logits, top_k).values[-1]
                    logits = torch.masked_fill(logits, logits < min_score, -float("inf"))
                next_token = torch.multinomial(logits.softmax(0), 1)

            out_tokens.append(next_token)
            depthformer_token = self.depth_embeddings[i](next_token).squeeze()

        return torch.cat(out_tokens)

    @torch.no_grad()
    def sample_frames(
        self,
        hidden: torch.Tensor,  # (batch, hidden_size)
        *,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Rollout AR du depthformer batché sur les lignes → (batch, codebooks).

        Même calcul que ``sample_frame`` mais en un seul rollout pour toutes
        les requêtes du step : 8 passes depthformer au total au lieu de
        8 × batch (le coût d'échantillonnage audio ne croît plus avec la
        concurrence)."""
        greedy = temperature is None or temperature <= 0 or top_k == 1
        hidden = hidden.to(self.depth_linear.weight.dtype)
        bsz = hidden.shape[0]
        depthformer_in = self.depth_linear(hidden).view(bsz, self.codebooks, self.depthformer_dim)
        depthformer_token = torch.zeros_like(depthformer_in[:, 0])  # (B, D)
        cache = None

        out_tokens: list[torch.Tensor] = []
        for i in range(self.codebooks):
            cur_input = (depthformer_in[:, i] + depthformer_token).unsqueeze(1)  # (B, 1, D)
            depthformer_out, cache = self.depthformer.forward_cached(cur_input, cache)
            logits = self.depth_embeddings[i].get_logits(depthformer_out[:, -1])  # (B, V)

            if greedy:
                next_token = logits.argmax(dim=-1)  # (B,)
            else:
                logits = logits / float(temperature or 1.0)
                if top_k is not None:
                    min_score = torch.topk(logits, top_k, dim=-1).values[..., -1:]
                    logits = torch.masked_fill(logits, logits < min_score, -float("inf"))
                next_token = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)

            out_tokens.append(next_token)
            depthformer_token = self.depth_embeddings[i](next_token)  # (B, D)

        return torch.stack(out_tokens, dim=1)  # (B, codebooks)

    @torch.no_grad()
    def embed_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """(codebooks,) codes → (hidden_size,) embedding d'entrée du step suivant."""
        return self.audio_embedding(frame.to(self.codebook_offsets.device) + self.codebook_offsets).sum(0)

    @torch.no_grad()
    def embed_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """(batch, codebooks) codes → (batch, hidden_size) embeddings."""
        return self.audio_embedding(frames.to(self.codebook_offsets.device) + self.codebook_offsets).sum(1)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> set[str]:
        """Charge le sous-ensemble audio depuis un state dict liquid-audio complet
        (clés ``audio_embedding.*``, ``depthformer.*``, ``depth_linear.*``,
        ``depth_embeddings.*`` — identiques aux noms de modules ici).

        Retourne les noms de PARAMÈTRES du module chargés (contrat
        ``track_weights_loading`` de vLLM)."""
        own = {
            k: v
            for k, v in weights.items()
            if k.split(".")[0] in ("audio_embedding", "depthformer", "depth_linear", "depth_embeddings")
        }
        missing, unexpected = self.load_state_dict(own, strict=False)
        real_missing = [k for k in missing if not k.endswith("codebook_offsets")]
        if real_missing:
            raise RuntimeError(f"audio head is missing weights: {real_missing[:5]}...")
        if unexpected:
            raise RuntimeError(f"unexpected audio head weights: {unexpected[:5]}...")
        return set(own)

    def loaded_param_names(self) -> set[str]:
        """Noms de paramètres du module (pour le tracking du loader vLLM)."""
        return {name for name, _ in self.named_parameters()}
