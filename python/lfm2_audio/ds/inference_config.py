"""Configuration typée du serving (modèles pydantic — validation à la frontière).

Les défauts encodent des contournements **mesurés** sur vLLM-Omni 0.22, pas des
préférences. Les changer sans lire ``docs/optimization_audit.md`` casse le flux
audio de façon silencieuse (texte correct, zéro chunk vers le stage 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Racine du repo depuis python/lfm2_audio/ds/config.py → ../../../
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPLOY_CONFIG = _REPO_ROOT / "configs" / "serving" / "vllm_omni.yaml"

Backend = Literal["vllm", "liquid", "auto"]


class EngineConfig(BaseModel):
    """Paramètres de démarrage de l'engine vLLM-Omni.

    ``deploy_config`` (le YAML par stage) est le chemin **recommandé** : il active
    les CUDA graphs PIECEWISE du stage 0 et ``initial_codec_chunk_frames``, soit
    un TTFA de 250-350 ms contre ~750 ms en tout-eager. Les champs scalaires
    ci-dessous ne servent que sur le chemin legacy (``deploy_config=None``) — le
    YAML les porte par stage et ne doit pas être écrasé globalement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    deploy_config: Path | None = Field(
        default=DEFAULT_DEPLOY_CONFIG if DEFAULT_DEPLOY_CONFIG.exists() else None,
        description="YAML de déploiement par stage ; None = kwargs legacy tout-eager",
    )
    gpu_memory_utilization: float = Field(default=0.42, gt=0.0, le=1.0)
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    enforce_eager: bool = True
    async_chunk: bool = True
    stage_init_timeout: int = Field(default=1200, gt=0)
    init_timeout: int = Field(default=1800, gt=0)

    # APC actif par défaut sur vLLM 0.22 → le chemin omni_prefix_cache du runner
    # refait le payload en découpant par tokens schedulés et PERD notre export
    # sparse codes.audio : texte correct mais ZÉRO chunk vers le stage 1
    # (mesuré le 12/06). À réactiver quand le merge mm sparse sera corrigé.
    enable_prefix_caching: bool = False
    # L'async scheduling tronque d'un token in-flight l'historique que notre
    # sampler rejoue pour reconstruire la modalité (écart 5 de l'audit runtime).
    async_scheduling: bool = False

    @field_validator("deploy_config")
    @classmethod
    def _deploy_config_must_exist(cls, value: Path | None) -> Path | None:
        if value is not None and not Path(value).exists():
            message = f"deploy_config introuvable : {value}"
            raise ValueError(message)
        return value

    def to_omni_kwargs(self, model: str) -> dict[str, Any]:
        """kwargs de ``vllm_omni.Omni(...)`` pour ce checkpoint."""
        kwargs: dict[str, Any] = {
            "model": model,
            "async_chunk": self.async_chunk,
            "stage_init_timeout": self.stage_init_timeout,
            "init_timeout": self.init_timeout,
        }
        if self.deploy_config is not None:
            kwargs["deploy_config"] = str(self.deploy_config)
        else:
            kwargs.update(
                enforce_eager=self.enforce_eager,
                gpu_memory_utilization=self.gpu_memory_utilization,
                dtype=self.dtype,
                async_scheduling=self.async_scheduling,
                enable_prefix_caching=self.enable_prefix_caching,
            )
        return kwargs

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Charge une surcharge depuis un YAML (clés = champs de ce modèle)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**data)
