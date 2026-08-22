"""``CheckpointResolver`` — orchestre sources, détection et préparation.

Collaborateurs injectables (source chain, détecteur, stratégies, cache) : le
résolveur ne connaît que leurs interfaces, ce qui le rend testable sans réseau,
sans GPU et sans écrire sur le disque de l'utilisateur.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from lfm2_audio.core.errors import CheckpointError
from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout, ResolvedCheckpoint
from lfm2_audio.serving.checkpoint.detector import LayoutDetector
from lfm2_audio.serving.checkpoint.preparers import (
    DEFAULT_PREPARERS,
    READY_MARKER,
    CheckpointPreparer,
)
from lfm2_audio.serving.checkpoint.sources import SourceChain

logger = logging.getLogger(__name__)

CACHE_ENV_VAR = "LFM2_SERVE_CACHE"
_DEFAULT_CACHE = Path.home() / ".cache" / "lfm2_audio" / "checkpoints"


class CheckpointResolver:
    """Transforme une :class:`CheckpointRequest` en :class:`ResolvedCheckpoint`."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        sources: SourceChain | None = None,
        detector: LayoutDetector | None = None,
        preparers: tuple[CheckpointPreparer, ...] = DEFAULT_PREPARERS,
    ) -> None:
        self._cache_dir = Path(cache_dir or os.environ.get(CACHE_ENV_VAR) or _DEFAULT_CACHE)
        self._sources = sources or SourceChain()
        self._detector = detector or LayoutDetector()
        self._preparers = preparers

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def resolve(self, request: CheckpointRequest) -> ResolvedCheckpoint:
        """Répertoire prêt à charger pour le backend demandé."""
        source, layout, adapter = self._locate(request)
        self._reject_backbone(source, layout)

        if request.backend == "liquid":
            # liquid-audio charge le layout natif et fusionne le LoRA en mémoire :
            # aucun export sur disque n'est nécessaire.
            return ResolvedCheckpoint(path=source, layout=layout, adapter=adapter)

        return self._prepare_for_omni(source, layout, adapter, request)

    # ------------------------------------------------------------------ #
    # Étapes
    # ------------------------------------------------------------------ #

    def _locate(self, request: CheckpointRequest) -> tuple[Path, Layout, Path | None]:
        """Matérialise la source et son adaptateur, et détecte le layout."""
        source = self._sources.materialize(request.spec)
        layout = self._detector.detect(source)
        adapter = self._sources.materialize(request.adapter) if request.adapter else None

        # Une spécification qui EST un adaptateur porte sa base dans son config.
        if layout is Layout.ADAPTER:
            if adapter is not None:
                message = (
                    f"{request.spec} est un adaptateur : `adapter=` ne peut pas être "
                    "fourni en plus. Passer la base en premier argument."
                )
                raise CheckpointError(message)
            adapter = source
            base = self._detector.base_model_of_adapter(source)
            logger.info("base lue dans l'adaptateur : %s", base)
            source = self._sources.materialize(base)
            layout = self._detector.detect(source)

        return source, layout, adapter

    @staticmethod
    def _reject_backbone(source: Path, layout: Layout) -> None:
        if layout is Layout.BACKBONE:
            message = (
                f"{source} est un backbone texte seul (Lfm2ForCausalLM) : servable par "
                "`vllm serve`, mais sans audio. Utiliser un export `--mode full` pour le S2S."
            )
            raise CheckpointError(message)

    def _prepare_for_omni(
        self,
        source: Path,
        layout: Layout,
        adapter: Path | None,
        request: CheckpointRequest,
    ) -> ResolvedCheckpoint:
        preparer = self._select_preparer(layout, adapter)
        target = self._cache_path(source, adapter, request.interleaved_ratio)

        if (target / READY_MARKER).exists():
            logger.info("checkpoint Omni réutilisé depuis le cache : %s", target)
            return ResolvedCheckpoint(path=target, layout=Layout.OMNI)

        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("préparation via %s", preparer.name)
        prepared = preparer.prepare(source, target, request, adapter)
        return ResolvedCheckpoint(path=prepared, layout=Layout.OMNI)

    def _select_preparer(self, layout: Layout, adapter: Path | None) -> CheckpointPreparer:
        has_adapter = adapter is not None
        for preparer in self._preparers:
            if preparer.handles(layout, has_adapter=has_adapter):
                return preparer
        message = (
            f"aucune stratégie de préparation pour layout={layout} (adaptateur={'oui' if has_adapter else 'non'})."
        )
        raise CheckpointError(message)

    def _cache_path(self, source: Path, adapter: Path | None, ratio: tuple[int, int] | None) -> Path:
        key = f"{source}|{adapter}|{ratio}"
        digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return self._cache_dir / f"{source.name}-omni-{digest}"
