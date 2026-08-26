"""``CampaignConfig`` — the versions to evaluate, and how, in one file.

Comparing v3 to v4 today means running two commands with hand-matched flags and
hoping nothing differed. One file instead: the same dataset, the same scorers
and the same decoding for every variant, so a difference between two runs can
only come from the variant itself.

The file is a frontier (a human writes it), so it is validated by pydantic and
refuses unknown keys — a mistyped ``adaptor:`` must fail loudly rather than be
silently ignored and produce an unlabelled run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lfm2_audio.ds.scoring_config import ScoringConfig


class VariantConfig(BaseModel):
    """One thing to evaluate: a checkpoint, an adapter, or a live endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Short label; becomes the run directory and the name shown in reports."""

    checkpoint: str | None = None
    adapter: str | None = None
    backend: str = "auto"
    endpoint: str | None = None
    """Serverless endpoint id — mutually exclusive with ``checkpoint``."""

    max_tokens: int = Field(default=400, gt=0)

    @model_validator(mode="after")
    def _require_one_source(self) -> VariantConfig:
        if bool(self.checkpoint) == bool(self.endpoint):
            message = f"variante {self.name!r} : renseigner soit checkpoint, soit endpoint (exactement un)"
            raise ValueError(message)
        return self

    @property
    def is_remote(self) -> bool:
        return self.endpoint is not None


class CampaignConfig(BaseModel):
    """A dataset, a scoring setup, and the variants to run against them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: str
    """JSONL of the evaluation set — identical for every variant, by construction."""

    audio_root: str | None = None
    tool_definitions: str | None = None
    variants: tuple[VariantConfig, ...] = ()
    scoring: ScoringConfig = ScoringConfig()
    runs_root: str = "reports/runs"

    max_parallel: int = Field(default=1, ge=1)
    """Variants evaluated at once.

    Local checkpoints share one GPU, so the default is sequential; raise it for
    endpoints, which are independent workers and where waiting on HTTP dominates.
    """

    limit: int | None = Field(default=None, gt=0)
    """Cap on cases, for a smoke run."""

    @model_validator(mode="after")
    def _require_variants(self) -> CampaignConfig:
        if not self.variants:
            message = "aucune variante déclarée : la campagne n'aurait rien à évaluer"
            raise ValueError(message)
        names = [v.name for v in self.variants]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            # Two runs under one name overwrite each other's directory, and the
            # comparison would silently read the survivor twice.
            message = f"noms de variantes dupliqués : {sorted(duplicates)}"
            raise ValueError(message)
        return self

    @property
    def local_variants(self) -> tuple[VariantConfig, ...]:
        return tuple(v for v in self.variants if not v.is_remote)

    @property
    def remote_variants(self) -> tuple[VariantConfig, ...]:
        return tuple(v for v in self.variants if v.is_remote)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CampaignConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**raw)

    def as_dict(self) -> dict[str, Any]:
        """The exact recipe, to travel with every run it produced."""
        return self.model_dump(mode="json")
