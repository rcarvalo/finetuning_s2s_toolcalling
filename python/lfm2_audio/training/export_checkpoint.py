"""Export du modèle fine-tuné : merge LoRA → checkpoints déployables (P0 vLLM-Omni).

Deux modes :

- ``--mode full`` : checkpoint LFM2.5-Audio complet (LoRA fusionné) au layout
  liquid-audio — rechargeable par ``LFM2AudioModel.from_pretrained`` et base de
  la conversion vers le stage 0 vLLM-Omni. Le ratio interleaved FR calibré est
  écrit dans le ``config.json`` exporté (**source unique** entraînement/serving).

- ``--mode backbone`` : extrait le backbone ``lfm.*`` fusionné et le remappe au
  layout HF standard ``Lfm2ForCausalLM`` → servable immédiatement par
  ``vllm serve <dir>`` (voie hybride audio-in/text-out, et test de parité P0).

Usage :

    python -m lfm2_audio.training.export_checkpoint \\
        --base LiquidAI/LFM2.5-Audio-1.5B \\
        --adapter outputs/phase2b_sft/final_adapter \\
        --output exports/lfm25_audio_fr --mode full \\
        --interleaved-text-tokens 6 --interleaved-audio-tokens 10

    python -m lfm2_audio.training.export_checkpoint \\
        --base LiquidAI/LFM2.5-Audio-1.5B \\
        --adapter outputs/phase2b_sft/final_adapter \\
        --output exports/lfm25_backbone_fr --mode backbone

Les fonctions de remapping/config sont pures (testables sans GPU) ; les imports
lourds restent dans les fonctions d'exécution.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from liquid_audio import LFM2AudioModel
from liquid_audio.utils import get_model_dir
from safetensors.torch import save_file

from lfm2_audio.core.model_ref import model_ref
from lfm2_audio.training.checkpoint_layout import (
    BACKBONE_AUX_FILES,
    FULL_AUX_DIRS,
    FULL_AUX_FILES,
    build_backbone_config,
    merged_full_mapping,
    remap_backbone_keys,
    strip_lora_keys,
    update_interleaved_ratio,
)
from lfm2_audio.training.lora import inject_lora, load_lora, load_lora_settings, merge_lora


def _load_merged_model(base: str | Path, adapter: str | None, device: str):

    model = LFM2AudioModel.from_pretrained(model_ref(base), device=device, dtype=torch.bfloat16).eval()
    if adapter:
        settings = load_lora_settings(adapter)
        inject_lora(model, settings)
        load_lora(model, Path(adapter) / "adapter_model.safetensors")
        merge_lora(model)
    return model


def _base_cache_dir(base: str | Path) -> Path:

    return Path(get_model_dir(model_ref(base)))


def export_full(
    base: str | Path, adapter: str | None, output: Path, device: str, n_text: int | None, n_audio: int | None
) -> None:

    model = _load_merged_model(base, adapter, device)
    state = model.state_dict()
    # {nom_export: nom_source} : retire lora_* ET désencapsule base_layer (sinon
    # le serving cherche ...weight et ne trouve que ...base_layer.weight → KeyError).
    mapping = merged_full_mapping(state.keys())
    save_file(
        {dst: state[src].contiguous().cpu() for dst, src in mapping.items()},
        str(output / "model.safetensors"),
    )

    cache = _base_cache_dir(base)
    liquid_config = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    liquid_config = update_interleaved_ratio(liquid_config, n_text, n_audio)
    (output / "config.json").write_text(json.dumps(liquid_config, indent=2), encoding="utf-8")

    for name in FULL_AUX_FILES:
        src = cache / name
        if src.exists():
            shutil.copy2(src, output / name)
    for name in FULL_AUX_DIRS:
        src = cache / name
        if src.is_dir():
            shutil.copytree(src, output / name, dirs_exist_ok=True)

    print(f"checkpoint complet exporté : {output} ({len(mapping)} tenseurs)")
    if n_text or n_audio:
        ratio = f"{liquid_config.get('interleaved_n_text')}:{liquid_config.get('interleaved_n_audio')}"
        print(f"ratio interleaved écrit : {ratio}")


def export_backbone(base: str, adapter: str | None, output: Path, device: str) -> None:

    model = _load_merged_model(base, adapter, device)
    state = model.state_dict()
    mapping = remap_backbone_keys(strip_lora_keys(state.keys()))
    save_file(
        {dst: state[src].contiguous().cpu() for src, dst in mapping.items()},
        str(output / "model.safetensors"),
    )

    cache = _base_cache_dir(base)
    liquid_config = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    config = build_backbone_config(liquid_config)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for name in BACKBONE_AUX_FILES:
        src = cache / name
        if src.exists():
            shutil.copy2(src, output / name)

    print(f"backbone exporté : {output} ({len(mapping)} tenseurs) — servable par `vllm serve {output}`")
