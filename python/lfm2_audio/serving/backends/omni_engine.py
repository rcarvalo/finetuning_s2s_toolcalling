"""``OmniEngine`` — cycle de vie de l'engine vLLM-Omni et contournements 0.22.

Isolé du backend pour une raison simple : ce fichier concentre tout ce qui est
**imposé par le runtime** et n'a rien à voir avec LFM2.5-Audio. Chaque
contournement ci-dessous a été mesuré ; les retirer casse le flux audio de façon
silencieuse (le texte continue de sortir, plus aucun chunk n'arrive).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm_omni import Omni

from lfm2_audio.core.env import load_vllm_omni_plugins
from lfm2_audio.ds.inference_config import EngineConfig

if TYPE_CHECKING:
    from lfm2_audio.ds.checkpoint import ResolvedCheckpoint

logger = logging.getLogger(__name__)


class OmniEngine:
    """Enveloppe de ``vllm_omni.Omni`` configurée pour le streaming in-process."""

    def __init__(self, checkpoint: ResolvedCheckpoint, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self._closed = False

        load_vllm_omni_plugins()

        started = time.time()
        self._omni = Omni(**self.config.to_omni_kwargs(str(checkpoint.path)))
        self._patch_for_streaming()
        logger.info(
            "engine prêt en %.0fs (async_chunk=%s)",
            time.time() - started,
            getattr(self._omni, "async_chunk", None),
        )

    def _patch_for_streaming(self) -> None:
        """Deux neutralisations nécessaires au streaming in-process.

        1. ``Omni.generate`` force ``FINAL_ONLY`` sur tous les stages LLM : aucun
           chunk audio ne sortirait, ni pendant ni à la fin. On respecte le DELTA
           demandé pour le stage 1.
        2. ``py_generator`` ferme l'engine quand le générateur s'épuise, ce qui
           tuerait le tour suivant. La vraie fermeture passe par :meth:`close`.
        """
        self._omni._set_final_only_for_llm_stages = list
        self._real_close = self._omni.close
        self._omni.close = lambda: None

    def sampling_pair(
        self,
        *,
        max_tokens: int,
        temperature: float,
        stop_token_ids: list[int],
    ) -> list[Any]:
        """Paramètres d'échantillonnage des 2 stages.

        Stage 0 en ``FINAL_ONLY`` (le texte complet en fin de tour), stage 1 en
        ``DELTA`` (les chunks audio au fil de l'eau). ``skip_special_tokens=False``
        car vLLM retirerait sinon ``<|tool_call_start|>`` / ``<|tool_call_end|>``
        du texte, rendant les tool calls indétectables.
        """

        return [
            SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
            SamplingParams(
                max_tokens=1,
                detokenize=False,
                output_kind=RequestOutputKind.DELTA,
            ),
        ]

    def generate(self, prompt: dict[str, Any], sampling: list[Any]) -> Iterator[Any]:
        """Sorties des 2 stages au fil de la génération."""
        yield from self._omni.generate(prompt, sampling, py_generator=True, use_tqdm=False)

    def close(self) -> None:
        """Ferme réellement l'engine. Idempotent."""
        if not self._closed:
            self._real_close()
            self._closed = True
