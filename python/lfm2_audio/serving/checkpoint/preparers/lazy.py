"""``LazyPreparer`` — déclare une stratégie sans charger son module.

*Proxy* : il répond à :meth:`handles` à partir d'une déclaration statique, et ne
résout la vraie classe qu'au moment de préparer. La chaîne de stratégies reste
donc uniforme, alors que la seule stratégie lourde (fusion LoRA, qui tire peft et
torch) n'est chargée que si un adaptateur est réellement fourni.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lfm2_audio.core.lazy_component import LazyComponent
from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout
from lfm2_audio.serving.checkpoint.preparers.base import CheckpointPreparer


class LazyPreparer(CheckpointPreparer):
    """Proxy d'un préparateur désigné par son chemin d'import."""

    def __init__(
        self,
        component: LazyComponent,
        predicate: Callable[[Layout, bool], bool],
        *,
        label: str,
    ) -> None:
        self._component = component
        self._predicate = predicate
        self._label = label
        self._delegate: CheckpointPreparer | None = None

    @property
    def name(self) -> str:
        return self._label

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        """Répond sans rien importer — c'est tout l'intérêt du proxy."""
        return self._predicate(layout, has_adapter)

    def prepare(
        self,
        source: Path,
        target: Path,
        request: CheckpointRequest,
        adapter: Path | None,
    ) -> Path:
        return self._resolve().prepare(source, target, request, adapter)

    def _resolve(self) -> CheckpointPreparer:
        if self._delegate is None:
            resolved: CheckpointPreparer = self._component.build()
            self._delegate = resolved
        return self._delegate
