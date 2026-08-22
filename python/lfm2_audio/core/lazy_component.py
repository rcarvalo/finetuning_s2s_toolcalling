"""``LazyComponent`` — résolution différée d'une classe par chemin d'import.

Les modules concrets (Whisper, juge Gemini, scorers ONNX) portent leurs imports
lourds **en tête**. Pour qu'une machine sans torch puisse tout de même lire une
config et lister les métriques, la résolution se fait par chaîne : c'est de la
répartition dynamique, pas un import différé caché dans une fonction.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any


@dataclass(frozen=True, slots=True)
class LazyComponent:
    """Classe désignée par son module et son nom, chargée à la construction."""

    module: str
    class_name: str
    requires: tuple[str, ...] = ()
    extra: str = "eval"

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

    def unavailable_reason(self) -> str | None:
        missing = self.missing_requirements
        if not missing:
            return None
        return f"{', '.join(missing)} non installé(s) — `uv sync --extra {self.extra}`"

    def load(self) -> type[Any]:
        loaded: type[Any] = getattr(import_module(self.module), self.class_name)
        return loaded

    def build(self, **kwargs: Any) -> Any:  # noqa: ANN401 — signature du composant cible
        return self.load()(**kwargs)
