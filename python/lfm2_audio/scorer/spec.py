"""``ScorerSpec`` — de quoi décrire un scorer sans l'importer."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfm2_audio.scorer.base import BaseScorer


@dataclass(frozen=True, slots=True)
class ScorerSpec:
    """Chemin d'import et prérequis d'un scorer.

    Décrire plutôt qu'importer : lister les scorers disponibles ne doit charger
    ni torch, ni onnxruntime, ni le client Gemini. Le module concret porte ses
    imports en tête et n'est chargé qu'à la construction.
    """

    name: str
    module: str
    class_name: str
    requires: tuple[str, ...] = ()
    """Modules tiers sans lesquels ce scorer ne peut pas tourner."""

    extra: str = "eval"
    """Extra pyproject à installer pour obtenir ``requires``."""

    description: str = ""

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        absent = []
        for module in self.requires:
            try:
                found = find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                absent.append(module)
        return tuple(absent)

    @property
    def is_installable(self) -> bool:
        return not self.missing_requirements

    def unavailable_reason(self) -> str | None:
        missing = self.missing_requirements
        if not missing:
            return None
        return f"{', '.join(missing)} non installé(s) — `uv sync --extra {self.extra}`"

    def load(self) -> type[BaseScorer]:
        """Classe du scorer, importée à la demande."""
        scorer_class: type[BaseScorer] = getattr(import_module(self.module), self.class_name)
        return scorer_class
