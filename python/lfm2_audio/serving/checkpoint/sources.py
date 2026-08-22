"""Sources de checkpoint — *Chain of Responsibility*.

Une spécification utilisateur (``"exports/mon_modele"``, ``"Rcarvalo/mon-repo"``)
est présentée à chaque source dans l'ordre ; la première qui l'accepte la
matérialise en répertoire local. Ajouter un stockage (S3, GCS…) = ajouter une
classe, sans toucher au résolveur.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from huggingface_hub import snapshot_download

from lfm2_audio.core.errors import CheckpointError

logger = logging.getLogger(__name__)


class CheckpointSource(ABC):
    """Maillon de la chaîne : sait reconnaître et matérialiser une spécification."""

    @abstractmethod
    def accepts(self, spec: str | Path) -> bool:
        """Vrai si cette source sait traiter ``spec``."""

    @abstractmethod
    def materialize(self, spec: str | Path) -> Path:
        """Retourne le répertoire local correspondant, en téléchargeant si besoin."""

    @property
    def name(self) -> str:
        return type(self).__name__


class LocalPathSource(CheckpointSource):
    """Répertoire déjà présent sur le disque."""

    def accepts(self, spec: str | Path) -> bool:
        return Path(spec).expanduser().exists()

    def materialize(self, spec: str | Path) -> Path:
        path = Path(spec).expanduser().resolve()
        if not path.is_dir():
            message = f"{path} n'est pas un répertoire de checkpoint."
            raise CheckpointError(message)
        return path


class HuggingFaceSource(CheckpointSource):
    """Repo du Hub, au format ``org/nom``."""

    def accepts(self, spec: str | Path) -> bool:
        text = str(spec)
        return "/" in text and not text.startswith((".", "/", "~"))

    def materialize(self, spec: str | Path) -> Path:

        logger.info("téléchargement du repo Hugging Face %s", spec)
        try:
            return Path(snapshot_download(str(spec))).resolve()
        except Exception as exc:
            message = f"téléchargement de {str(spec)!r} impossible : {exc}"
            raise CheckpointError(message) from exc


class SourceChain:
    """Chaîne ordonnée de sources. Première acceptante = gagnante."""

    def __init__(self, sources: list[CheckpointSource] | None = None) -> None:
        # Le local d'abord : un répertoire nommé `org/nom` doit gagner sur le Hub.
        self._sources = sources or [LocalPathSource(), HuggingFaceSource()]

    def materialize(self, spec: str | Path) -> Path:
        for source in self._sources:
            if source.accepts(spec):
                logger.debug("%s prend en charge %s", source.name, spec)
                return source.materialize(spec)
        message = (
            f"{spec!r} n'existe pas localement et n'est pas un identifiant de repo Hugging Face (attendu `org/nom`)."
        )
        raise CheckpointError(message)
