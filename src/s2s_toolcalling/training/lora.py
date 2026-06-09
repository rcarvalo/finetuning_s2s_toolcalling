"""Injection LoRA dans le backbone LFM2.5 de LFM2AudioModel (Phase 2).

Utilise ``peft.inject_adapter_in_model`` (API bas niveau) : les couches Linear
ciblées sont remplacées in place par des LoraLayer SANS wrapper PeftModel,
donc ``LFM2AudioModel.forward`` et le Trainer liquid-audio restent inchangés.

Cibles par défaut (modules documentés pour LFM2 : attention GQA + GLU + blocs
conv LIV) — restreintes au préfixe ``lfm.`` pour ne jamais toucher l'encodeur
ni les têtes audio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Attention (q/k/v/out), GLU (w1/w2/w3), conv LIV (in_proj/out_proj).
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"]


@dataclass(slots=True)
class LoraSettings:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))


def inject_lora(model, settings: LoraSettings):
    """Injecte les adaptateurs dans ``model.lfm`` (in place). Retourne la LoraConfig."""
    from peft import LoraConfig, inject_adapter_in_model

    config = LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=settings.target_modules,
        bias="none",
    )
    inject_adapter_in_model(config, model.lfm)

    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    if n_lora == 0:
        raise RuntimeError(
            "no LoRA parameters injected — check target_modules against the LFM2.5 module names"
        )
    logger.info("LoRA injected: %.2fM adapter parameters (r=%d, alpha=%d)", n_lora / 1e6, settings.r, settings.alpha)
    return config


def lora_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model, output_dir: str | Path) -> Path:
    from safetensors.torch import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "adapter_model.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in lora_state_dict(model).items()}, str(path))
    return path


def load_lora(model, adapter_path: str | Path) -> None:
    from safetensors.torch import load_file

    weights = load_file(str(adapter_path))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    unexpected = [k for k in unexpected if "lora_" in k]
    if unexpected:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}...")


def merge_lora(model) -> int:
    """Fusionne les adaptateurs dans les poids de base (export GGUF / inférence).

    Retourne le nombre de couches fusionnées.
    """
    from peft.tuners.lora import LoraLayer

    merged = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
            merged += 1
    logger.info("merged %d LoRA layers into base weights", merged)
    return merged
