"""``LayoutDetector`` — reconnaît le layout d'un répertoire de checkpoint.

Responsabilité unique : lire ``config.json`` et répondre. Aucune écriture, aucun
téléchargement — ce qui le rend testable sans GPU ni réseau.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lfm2_audio.core.errors import CheckpointError
from lfm2_audio.ds.checkpoint import Layout
from lfm2_audio.vllm_plugin.convert_checkpoint import (
    OMNI_ARCHITECTURE,
    OMNI_MODEL_TYPE,
    REQUIRED_LIQUID_SECTIONS,
)

_BACKBONE_ARCHITECTURE = "Lfm2ForCausalLM"
_BACKBONE_MODEL_TYPE = "lfm2"


class LayoutDetector:
    """Déduit un :class:`Layout` d'un répertoire local."""

    def detect(self, directory: Path) -> Layout:
        """Lève ``CheckpointError`` si le répertoire n'est pas un checkpoint connu."""
        if (directory / "adapter_config.json").exists():
            return Layout.ADAPTER

        config = self._read_config(directory)
        architectures = config.get("architectures") or []
        model_type = config.get("model_type")

        if OMNI_ARCHITECTURE in architectures or model_type == OMNI_MODEL_TYPE:
            return Layout.OMNI
        if all(section in config for section in REQUIRED_LIQUID_SECTIONS):
            return Layout.LIQUID
        if _BACKBONE_ARCHITECTURE in architectures or model_type == _BACKBONE_MODEL_TYPE:
            return Layout.BACKBONE

        message = (
            f"layout non reconnu dans {directory} : architectures={architectures}, "
            f"model_type={model_type!r}. Attendu un checkpoint LFM2.5-Audio "
            "(liquid ou Omni)."
        )
        raise CheckpointError(message)

    def base_model_of_adapter(self, adapter_dir: Path) -> str:
        """Modèle de base déclaré par peft dans ``adapter_config.json``."""
        config_path = adapter_dir / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = config.get("base_model_name_or_path")
        if not base:
            message = (
                f"{adapter_dir} est un adaptateur sans `base_model_name_or_path` : "
                "préciser explicitement le modèle de base."
            )
            raise CheckpointError(message)
        return str(base)

    @staticmethod
    def _read_config(directory: Path) -> dict[str, Any]:
        config_path = directory / "config.json"
        if not config_path.exists():
            message = f"{directory} ne contient ni config.json ni adapter_config.json — ce n'est pas un checkpoint."
            raise CheckpointError(message)
        try:
            parsed: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            message = f"config.json illisible dans {directory} : {exc}"
            raise CheckpointError(message) from exc
        return parsed
