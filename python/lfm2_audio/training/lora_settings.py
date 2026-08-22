"""``LoraSettings`` — hyperparamètres d'un adaptateur LoRA.

Séparé de :mod:`lfm2_audio.training.lora`, qui importe peft : une config
d'entraînement se lit et se valide sans installer peft, et le callback de push
Hub n'a besoin que de ces valeurs pour écrire l'``adapter_config.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TARGET_MODULES = [
    # Attention (q/k/v/out), GLU (w1/w2/w3), convolution LIV (in_proj).
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "w1",
    "w2",
    "w3",
    "in_proj",
]


@dataclass(slots=True)
class LoraSettings:
    """Rang, échelle et modules ciblés de l'injection."""

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
