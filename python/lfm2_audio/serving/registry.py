"""``BackendRegistry`` — *Factory* des backends d'inférence.

Les backends sont enregistrés par **chemin d'import**, pas par classe : importer
``lfm2_audio`` ne doit jamais tirer torch, vLLM ou liquid-audio. La classe n'est
chargée qu'au moment de l'instanciation, ce qui permet aussi au mode ``"auto"``
de choisir le premier backend réellement installé sur la machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

from lfm2_audio.core.errors import BackendUnavailableError

if TYPE_CHECKING:
    from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)

AUTO = "auto"


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Ce qu'il faut pour charger un backend sans l'importer tout de suite."""

    name: str
    module: str
    class_name: str
    requires: tuple[str, ...]
    """Modules tiers dont l'absence rend ce backend inutilisable."""

    description: str = ""

    @property
    def is_available(self) -> bool:
        return all(find_spec(module) is not None for module in self.requires)

    def load(self) -> type[LFM2Audio]:
        backend_class: type[LFM2Audio] = getattr(import_module(self.module), self.class_name)
        return backend_class


class BackendRegistry:
    """Catalogue des backends. Instancié une fois dans ``serving`` (singleton léger)."""

    def __init__(self, specs: tuple[BackendSpec, ...] = ()) -> None:
        self._specs: dict[str, BackendSpec] = {spec.name: spec for spec in specs}

    def register(self, spec: BackendSpec) -> None:
        self._specs[spec.name] = spec

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def available(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self._specs.items() if spec.is_available)

    def describe(self, name: str) -> BackendSpec:
        """Spec d'un backend, **sans** vérifier qu'il est installé.

        Sert à l'introspection (aide des CLIs, diagnostics) là où :meth:`get`
        refuserait un backend absent de la machine.
        """
        spec = self._specs.get(name)
        if spec is None:
            message = f"backend inconnu : {name!r} (connus : {', '.join(self.names)})"
            raise BackendUnavailableError(message)
        return spec

    def get(self, name: str) -> BackendSpec:
        """Spec d'un backend, ``"auto"`` compris."""
        resolved = self._resolve_auto() if name == AUTO else name
        spec = self.describe(resolved)
        if not spec.is_available:
            missing = [module for module in spec.requires if find_spec(module) is None]
            message = (
                f"backend {spec.name!r} indisponible : {', '.join(missing)} non installé(s). "
                "Installer avec `uv sync --extra serving` (GPU requis)."
            )
            raise BackendUnavailableError(message)
        return spec

    def load(self, name: str) -> type[LFM2Audio]:
        """Classe du backend, importée à la demande."""
        return self.get(name).load()

    def _resolve_auto(self) -> str:
        for name, spec in self._specs.items():
            if spec.is_available:
                logger.info("backend choisi automatiquement : %s", name)
                return name
        message = (
            f"aucun backend d'inférence installé — `uv sync --extra serving` (candidats : {', '.join(self.names)})."
        )
        raise BackendUnavailableError(message)


BACKENDS = BackendRegistry(
    (
        # L'ordre fixe la préférence du mode "auto" : vLLM-Omni d'abord (basse
        # latence), liquid-audio en repli (référence PyTorch, batch=1).
        BackendSpec(
            name="vllm",
            module="lfm2_audio.serving.backends.vllm_omni",
            class_name="VllmOmniBackend",
            requires=("vllm", "vllm_omni", "liquid_audio"),
            description="vLLM-Omni 2 stages — streaming basse latence (TTFA ~300 ms)",
        ),
        BackendSpec(
            name="liquid",
            module="lfm2_audio.serving.backends.liquid",
            class_name="LiquidAudioBackend",
            requires=("liquid_audio",),
            description="liquid-audio PyTorch — implémentation de référence, batch=1",
        ),
    )
)
