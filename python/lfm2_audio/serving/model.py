"""``LFM2Audio`` — l'API publique du serving.

C'est le seul objet que l'utilisateur manipule :

>>> from lfm2_audio import LFM2Audio                     # doctest: +SKIP
>>> model = LFM2Audio.from_pretrained("Rcarvalo/lfm25-tc-en-s2s")  # doctest: +SKIP
>>> text, audio = model.reply(audio="question.wav")      # doctest: +SKIP

Classe abstraite plutôt que façade délégante : ``from_pretrained`` est une
*fabrique* qui retourne la sous-classe correspondant au backend choisi, sans
couche d'indirection supplémentaire. Les sous-classes n'implémentent que
:meth:`stream` — :meth:`reply` en dérive par *template method*.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import numpy.typing as npt

from lfm2_audio.core.prompt import DEFAULT_SYSTEM
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.checkpoint import CheckpointRequest, ResolvedCheckpoint
from lfm2_audio.ds.conversation import Conversation
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.ds.reply import Reply, TurnMetrics
from lfm2_audio.serving.checkpoint.resolver import CheckpointResolver
from lfm2_audio.serving.registry import BACKENDS

logger = logging.getLogger(__name__)

type AudioInput = Waveform | str | Path | tuple[npt.ArrayLike, int]
"""Formes acceptées en entrée audio : value object, chemin, ou (signal, fréquence)."""


class LFM2Audio(ABC):
    """Modèle LFM2.5-Audio chargé, prêt pour un dialogue speech-to-speech."""

    backend_name: str = ""
    """Renseigné par chaque sous-classe — sert aux logs et aux messages d'erreur."""

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        system: str = DEFAULT_SYSTEM,
        generation: GenerationConfig | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.system = system
        self.generation = generation or GenerationConfig()
        self.conversation = Conversation()
        self._last_reply: Reply | None = None

    # ------------------------------------------------------------------ #
    # Fabrique
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        backend: str = "auto",
        adapter: str | Path | None = None,
        system: str = DEFAULT_SYSTEM,
        engine: EngineConfig | None = None,
        generation: GenerationConfig | None = None,
        interleaved_ratio: tuple[int, int] | None = None,
        cache_dir: str | Path | None = None,
    ) -> LFM2Audio:
        """Charge un modèle depuis n'importe quelle forme de checkpoint.

        ``model`` accepte un répertoire local (layout liquid-audio ou déjà
        converti), un repo Hugging Face, ou un répertoire d'adaptateur LoRA —
        dont la base est alors lue dans ``adapter_config.json``. Ce qui manque
        (fusion LoRA, conversion) est produit une fois puis mis en cache.

        ``backend`` : ``"vllm"``, ``"liquid"``, ou ``"auto"`` (premier installé).
        """
        spec = BACKENDS.get(backend)
        request = CheckpointRequest(
            spec=model,
            backend=spec.name,
            adapter=adapter,
            interleaved_ratio=interleaved_ratio,
        )
        resolved = CheckpointResolver(cache_dir=cache_dir).resolve(request)
        logger.info("checkpoint %s (%s) → backend %s", resolved.path, resolved.layout, spec.name)

        backend_class = spec.load()
        return backend_class._build(resolved, system=system, engine=engine, generation=generation)

    @classmethod
    @abstractmethod
    def _build(
        cls,
        checkpoint: ResolvedCheckpoint,
        *,
        system: str,
        engine: EngineConfig | None,
        generation: GenerationConfig | None,
    ) -> Self:
        """Construit l'instance à partir d'un checkpoint résolu.

        Chaque backend consomme les options qui le concernent — ``engine`` n'a de
        sens que pour vLLM-Omni, par exemple.
        """

    # ------------------------------------------------------------------ #
    # Génération — template method
    # ------------------------------------------------------------------ #

    @abstractmethod
    def stream(
        self,
        *,
        text: str | None = None,
        audio: AudioInput | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Waveform]:
        """Génère un tour en rendant les chunks audio au fil de l'eau.

        Doit renseigner :attr:`_last_reply` avant de s'épuiser, et pousser le
        tour user puis le tour assistant dans :attr:`conversation`.
        """

    def reply(
        self,
        *,
        text: str | None = None,
        audio: AudioInput | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """Génère un tour complet et retourne texte + audio concaténé.

        Déballable : ``text, audio = model.reply(...)``.
        """
        chunks = list(self.stream(text=text, audio=audio, max_tokens=max_tokens))
        return self._finish(Waveform.concat(chunks))

    # ------------------------------------------------------------------ #
    # État
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Vide l'historique — le tour suivant repart d'un contexte neuf."""
        self.conversation.clear()
        self._last_reply = None

    @property
    def last_reply(self) -> Reply | None:
        """Dernière réponse générée (renseignée même en mode streaming)."""
        return self._last_reply

    @property
    def last_text(self) -> str:
        """Texte BRUT du dernier tour, marqueurs compris.

        C'est ce dont l'orchestrateur a besoin : ``<|tool_call_start|>`` et son
        pendant fermant doivent survivre pour que le span soit parsable.
        """
        return self._last_reply.raw_text if self._last_reply else ""

    def close(self) -> None:  # noqa: B027 — hook optionnel, pas un contrat abstrait
        """Libère les ressources du backend. Idempotent.

        Volontairement concrète et vide : tous les backends n'ont pas de
        ressource à libérer (liquid-audio s'appuie sur le GC de torch), donc
        l'imposer comme abstraite obligerait à écrire des implémentations vides.
        """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(checkpoint={self.checkpoint.name!r})"

    # ------------------------------------------------------------------ #
    # Outils pour les sous-classes
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_audio(audio: AudioInput | None) -> Waveform | None:
        """Normalise une entrée audio en :class:`Waveform` (chemin, tuple, tableau)."""
        if audio is None or isinstance(audio, Waveform):
            return audio
        if isinstance(audio, (str, Path)):
            return Waveform.from_file(audio)
        if isinstance(audio, tuple):
            samples, sample_rate = audio
            return Waveform.of(samples, int(sample_rate))
        message = f"entrée audio non supportée : {type(audio).__name__}"
        raise TypeError(message)

    def _finish(self, audio: Waveform | None) -> Reply:
        """Recompose la réponse finale à partir du dernier tour streamé."""
        streamed = self._last_reply
        if streamed is None:  # stream() n'a rien renseigné : tour vide
            return Reply(text="", audio=audio)
        completed = Reply(
            text=streamed.text,
            audio=audio,
            metrics=streamed.metrics,
            raw_text=streamed.raw_text,
        )
        self._last_reply = completed
        return completed

    @staticmethod
    def _elapsed(start: float, frames: int, ttfa: float | None) -> TurnMetrics:
        return TurnMetrics(ttfa_s=ttfa, total_s=time.time() - start, audio_frames=frames)
