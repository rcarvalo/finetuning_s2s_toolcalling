"""``GenerationConfig`` — paramètres d'échantillonnage d'un tour."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Paramètres d'échantillonnage d'un tour.

    Le texte est greedy (``temperature=0``) : les tool calls doivent être
    déterministes. La prosodie audio est échantillonnée **à l'intérieur** du
    modèle (``audio_temperature`` / ``audio_top_k`` du config du checkpoint), pas
    par le sampler de vLLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(default=400, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    audio_temperature: float = Field(default=1.0, ge=0.0)
    audio_top_k: int = Field(default=4, gt=0)

    text_only: bool = False
    """Génération texte seule, sans interleave audio forcé.

    Indispensable pour évaluer le tool calling sur le backend liquid : l'audio
    interleavé shredde un span structuré ``[fn(arg="…")]``. Sans effet côté
    vLLM-Omni, où le tool call sort déjà dans le flux texte.
    """

    def with_max_tokens(self, max_tokens: int | None) -> GenerationConfig:
        """Copie avec ``max_tokens`` surchargé (no-op si ``None``)."""
        if max_tokens is None:
            return self
        return self.model_copy(update={"max_tokens": max_tokens})
