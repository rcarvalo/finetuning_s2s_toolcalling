"""``TrainingConfig`` — recette d'entraînement complète, validée à la lecture.

Modèle pydantic parce que cette configuration vient d'un YAML. Elle décrit
**aussi** les métriques à suivre en cours de route (:class:`ScoringConfig`) :
c'est ce qui permet de changer les scorers suivis sans toucher au code du
lanceur.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from lfm2_audio.ds.scoring_config import ScoringConfig


class LoraConfig(BaseModel):
    """Adaptateurs LoRA injectés dans le backbone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)


class FreezeConfig(BaseModel):
    """Ce qui reste gelé pendant l'entraînement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    freeze_encoder: bool = True
    freeze_audio_heads: bool = True
    freeze_backbone: bool = False


class EvaluationScheduleConfig(BaseModel):
    """Quand et sur quoi faire tourner les scorers pendant l'entraînement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    question_set: str = ""
    """JSONL du petit jeu de suivi — quelques dizaines de cas, pas plus."""

    audio_root: str | None = None
    interval: int = Field(default=500, gt=0)
    at_start: bool = True
    """Mesure de référence avant le premier pas : sans elle, rien n'est lisible."""

    max_questions: int | None = 32
    max_new_tokens: int = Field(default=256, gt=0)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


class TrainingConfig(BaseModel):
    """Recette complète : données, optimisation, instrumentation, suivi métrique."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = "LiquidAI/LFM2.5-Audio-1.5B"
    train_dataset: str = ""
    val_dataset: str | None = None
    context_length: int = Field(default=4096, gt=0)

    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    max_steps: int = Field(default=2000, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    batch_size: int = Field(default=2, gt=0)
    dataloader_num_workers: int = Field(default=2, ge=0)

    logging_interval: int = Field(default=10, gt=0)
    save_interval: int = Field(default=500, gt=0)
    val_interval: int = Field(default=200, gt=0)
    output_dir: str = "outputs/sft"

    grad_clip: float = Field(default=1.0, ge=0)
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    hub_repo: str | None = None
    push_interval: int = Field(default=0, ge=0)
    """Push de l'adaptateur tous les N pas. 0 = jamais (le push final reste)."""

    lora: LoraConfig = Field(default_factory=LoraConfig)
    freeze: FreezeConfig = Field(default_factory=FreezeConfig)
    evaluation: EvaluationScheduleConfig = Field(default_factory=EvaluationScheduleConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        """Aplatie pour wandb — la recette exacte accompagne le run."""
        return self.model_dump(mode="json")
