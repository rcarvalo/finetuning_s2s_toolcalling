"""Wrapper CUDA graph du rollout depthformer (action 5 de l'audit, pattern MiMo).

Le rollout eager de ``Lfm2AudioHead.sample_frames`` coûte ~16 kernel launchs
Python par frame audio (8 passes depthformer + sampling) — c'est l'overhead
dominant du step audio, ~5-15 ms/frame selon le GPU. Pattern in-tree :
``mimo_audio/cuda_graph_decoder_wrapper.py`` et ``qwen3_tts`` — buffers
statiques par bucket de batch, capture une fois, replay ensuite (1 launch).

Spécificités ici :
- capture LAZY au premier step audio de chaque bucket (le runner ne fournit
  pas de hook post-chargement des poids) — coût one-shot absorbé par les
  tours de warmup, comme le JIT Triton ;
- le sampling par défaut est greedy (argmax, déterministe) → capture sûre ;
  en mode stochastique, ``torch.cuda.graph`` enregistre le générateur RNG
  (philox) et chaque replay tire un aléa frais ;
- fallback eager systématique : batch > bucket max, device CPU, ou capture
  échouée (bucket blacklisté, pas de nouvelle tentative) ;
- kill-switch : env ``LFM2_DEPTHFORMER_EAGER=1`` (câblé dans lfm2_audio_ar).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# max_num_seqs=4 dans le YAML de déploiement ; 8 couvre une marge de config.
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8)


class CudaGraphDepthformer:
    """Replay CUDA graph de ``sample_frames`` par bucket de taille de batch."""

    def __init__(
        self,
        head,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES,
    ) -> None:
        self.head = head
        self.temperature = temperature
        self.top_k = top_k
        self.capture_sizes = sorted(capture_sizes)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._hidden_in: dict[int, torch.Tensor] = {}
        self._frames_out: dict[int, torch.Tensor] = {}
        self._failed: set[int] = set()
        self._pool = None  # pool mémoire partagé entre les buckets

    def _bucket_for(self, batch: int) -> int | None:
        for size in self.capture_sizes:
            if batch <= size:
                return size
        return None

    def _eager(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head.sample_frames(
            hidden, temperature=self.temperature, top_k=self.top_k
        )

    def _capture(self, bucket: int, hidden: torch.Tensor) -> None:
        dtype = self.head.depth_linear.weight.dtype
        static_in = torch.zeros(
            bucket, hidden.shape[1], dtype=dtype, device=hidden.device
        )
        # warmup hors capture sur un stream dédié (recette torch.cuda.graph :
        # les allocations de chauffe ne polluent pas le pool du graph)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._eager(static_in)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            static_out = self._eager(static_in)
        if self._pool is None:
            self._pool = graph.pool()

        self._graphs[bucket] = graph
        self._hidden_in[bucket] = static_in
        self._frames_out[bucket] = static_out
        logger.info("[depthformer-graph] bucket=%d capturé (1 replay/frame)", bucket)

    @torch.no_grad()
    def sample_frames(self, hidden: torch.Tensor) -> torch.Tensor:
        """(batch, hidden_size) → (batch, codebooks) — replay ou fallback eager."""
        batch = hidden.shape[0]
        if hidden.device.type != "cuda":
            return self._eager(hidden)
        bucket = self._bucket_for(batch)
        if bucket is None or bucket in self._failed:
            return self._eager(hidden)
        if bucket not in self._graphs:
            try:
                self._capture(bucket, hidden)
            except Exception:
                logger.warning(
                    "[depthformer-graph] capture bucket=%d échouée — fallback eager définitif",
                    bucket, exc_info=True,
                )
                self._failed.add(bucket)
                return self._eager(hidden)

        static_in = self._hidden_in[bucket]
        static_in[:batch].copy_(hidden)
        if batch < bucket:
            static_in[batch:].zero_()  # lignes de padding (sorties ignorées)
        self._graphs[bucket].replay()
        return self._frames_out[bucket][:batch].clone()
