"""Sonde de dépendances optionnelles des scorers.

``find_spec`` plutôt qu'un ``import`` : on veut savoir si une dépendance lourde
(torch, onnxruntime) est installée sans payer son chargement tant que personne
ne score.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.util import find_spec


def missing_modules(modules: Iterable[str]) -> list[str]:
    """Modules non importables parmi ceux passés."""
    absent: list[str] = []
    for module in modules:
        try:
            found = find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            absent.append(module)
    return absent


def describe_missing(modules: Iterable[str], *, extra: str) -> str | None:
    """Message d'indisponibilité prêt à afficher, ou ``None`` si tout est là."""
    absent = missing_modules(modules)
    if not absent:
        return None
    return f"{', '.join(absent)} non installé(s) — `uv sync --extra {extra}`"
