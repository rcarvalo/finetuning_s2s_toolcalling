"""Injection LoRA dans le backbone LFM2.5 de LFM2AudioModel (Phase 2).

Utilise ``peft.inject_adapter_in_model`` (API bas niveau) : les couches Linear
ciblées sont remplacées in place par des LoraLayer SANS wrapper PeftModel,
donc ``LFM2AudioModel.forward`` et le Trainer liquid-audio restent inchangés.

Cibles par défaut (modules documentés pour LFM2 : attention GQA + GLU + blocs
conv LIV) — restreintes au préfixe ``lfm.`` pour ne jamais toucher l'encodeur
ni les têtes audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file

from lfm2_audio.training.lora_settings import LoraSettings

logger = logging.getLogger(__name__)


def inject_lora(model, settings: LoraSettings):
    """Injecte les adaptateurs dans ``model.lfm`` (in place). Retourne la LoraConfig."""

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
        raise RuntimeError("no LoRA parameters injected — check target_modules against the LFM2.5 module names")
    logger.info("LoRA injected: %.2fM adapter parameters (r=%d, alpha=%d)", n_lora / 1e6, settings.r, settings.alpha)
    return config


def lora_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model, output_dir: str | Path, settings: LoraSettings | None = None) -> Path:
    """Sauve les poids de l'adaptateur + sa config (nécessaire pour réinjecter
    le LoRA à l'identique au moment du merge/export)."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "adapter_model.safetensors"
    save_file({k: v.contiguous().cpu() for k, v in lora_state_dict(model).items()}, str(path))
    if settings is not None:
        (out / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": settings.r,
                    "alpha": settings.alpha,
                    "dropout": settings.dropout,
                    "target_modules": settings.target_modules,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return path


def load_lora_settings(adapter_dir: str | Path) -> LoraSettings:
    """Relit la config d'adaptateur écrite par ``save_lora``.

    Repli sur les DÉFAUTS (r16/alpha32/dropout0.05/cibles standard) si
    ``adapter_config.json`` est absent : les ``DEFAULT_TARGET_MODULES`` recréent
    la même structure LoRA que ``save_lora`` par défaut, donc le ``.safetensors``
    se recharge correctement même sans config.
    """

    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        logger.warning(
            "adapter_config.json absent dans %s — défauts LoRA (r=16, alpha=32, cibles standard)", adapter_dir
        )
        return LoraSettings(enabled=True)
    return LoraSettings(enabled=True, **json.loads(cfg_path.read_text(encoding="utf-8")))


def load_lora(model, adapter_path: str | Path) -> None:

    weights = load_file(str(adapter_path))
    _missing, unexpected = model.load_state_dict(weights, strict=False)
    unexpected = [k for k in unexpected if "lora_" in k]
    if unexpected:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}...")


def warm_start_lora(model, source: str) -> None:
    """Load adapter weights from a Hub repo id or a local directory.

    Used to resume a run that a preempted VM cut short: the adapter must
    already be injected (same r/alpha/targets), only its weights are replaced.
    """
    local = Path(source)
    if local.is_dir():
        # An existing directory is a local adapter, never a repo id: falling
        # through to the Hub would report a confusing network error instead.
        path = local / "adapter_model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"no adapter_model.safetensors in {local}")
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(source, "adapter_model.safetensors"))
    load_lora(model, path)
    logger.info("LoRA warm-started from %s", source)


def merge_lora(model) -> int:
    """Fusionne les adaptateurs dans les poids de base (export GGUF / inférence).

    Retourne le nombre de couches fusionnées.
    """

    merged = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.merge()
            merged += 1
    logger.info("merged %d LoRA layers into base weights", merged)
    return merged
