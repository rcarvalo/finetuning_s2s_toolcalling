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

    python -m s2s_toolcalling.training.export_checkpoint \\
        --base LiquidAI/LFM2.5-Audio-1.5B \\
        --adapter outputs/phase2b_sft/final_adapter \\
        --output exports/lfm25_audio_fr --mode full \\
        --interleaved-text-tokens 6 --interleaved-audio-tokens 10

    python -m s2s_toolcalling.training.export_checkpoint \\
        --base LiquidAI/LFM2.5-Audio-1.5B \\
        --adapter outputs/phase2b_sft/final_adapter \\
        --output exports/lfm25_backbone_fr --mode backbone

Les fonctions de remapping/config sont pures (testables sans GPU) ; les imports
lourds restent dans les fonctions d'exécution.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

BACKBONE_PREFIX = "lfm."

# Fichiers annexes copiés tels quels du checkpoint de base (mode full).
FULL_AUX_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer-e351c8d8-checkpoint125.safetensors",  # poids Mimi
)
FULL_AUX_DIRS = ("audio_detokenizer",)

# Tokenizer seul pour le backbone texte.
BACKBONE_AUX_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


class ExportError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Fonctions pures (testées sans GPU)
# --------------------------------------------------------------------------- #


def _unwrap_lora(key: str) -> str:
    """Désencapsule le wrapper peft : ``....base_layer.weight`` → ``....weight``.

    ``merge_lora`` (peft ``LoraLayer.merge``) fole le delta DANS ``base_layer.weight``
    mais laisse le module wrappé → le state_dict garde l'infixe ``.base_layer.``.
    Le serving (vLLM / from_pretrained) attend le nom nu : on le retire ici.
    """
    return key.replace(".base_layer.", ".")


def remap_backbone_keys(keys: Iterable[str]) -> dict[str, str]:
    """``lfm.X`` → ``model.X`` (layout HF Lfm2ForCausalLM). Ignore les autres modules.

    Refuse les clés LoRA résiduelles : l'export exige un état FUSIONNÉ. Le wrapper
    peft (``base_layer``) est désencapsulé côté destination.
    """
    mapping: dict[str, str] = {}
    for key in keys:
        if "lora_" in key:
            raise ExportError(f"unmerged LoRA key found: {key!r} — call merge_lora before exporting")
        if key.startswith(BACKBONE_PREFIX):
            mapping[key] = "model." + _unwrap_lora(key[len(BACKBONE_PREFIX) :])
    if not mapping:
        raise ExportError("no backbone keys found (expected keys prefixed with 'lfm.')")
    return mapping


def strip_lora_keys(keys: Iterable[str]) -> list[str]:
    """Clés à conserver dans l'export full : tout sauf les adaptateurs (déjà fusionnés)."""
    return [k for k in keys if "lora_" not in k]


def merged_full_mapping(keys: Iterable[str]) -> dict[str, str]:
    """``{nom_export: nom_source}`` pour l'export full mergé : retire les adaptateurs
    (``lora_*``) et désencapsule le wrapper peft (``.base_layer.`` → ``.``).

    Le nom d'export peut différer du nom source (``base_layer``) → on retourne le
    mapping pour réindexer le state_dict à la sauvegarde.
    """
    return {_unwrap_lora(k): k for k in keys if "lora_" not in k}


def build_backbone_config(liquid_config: dict[str, Any]) -> dict[str, Any]:
    """config.json HF Lfm2ForCausalLM depuis la section ``lfm`` du config liquid-audio."""
    lfm = liquid_config.get("lfm")
    if not isinstance(lfm, dict):
        raise ExportError("liquid config has no 'lfm' section")
    config = dict(lfm)
    config["architectures"] = ["Lfm2ForCausalLM"]
    config["model_type"] = "lfm2"
    config.setdefault("tie_word_embeddings", True)
    return config


def update_interleaved_ratio(liquid_config: dict[str, Any], n_text: int | None, n_audio: int | None) -> dict[str, Any]:
    """Écrit le ratio calibré dans le config exporté (source unique du serving)."""
    config = dict(liquid_config)
    if n_text is not None:
        if n_text < 1:
            raise ExportError("interleaved_n_text must be >= 1")
        config["interleaved_n_text"] = n_text
    if n_audio is not None:
        if n_audio < 1:
            raise ExportError("interleaved_n_audio must be >= 1")
        config["interleaved_n_audio"] = n_audio
    return config


# --------------------------------------------------------------------------- #
# Exécution (GPU / IO)
# --------------------------------------------------------------------------- #


def _load_merged_model(base: str, adapter: str | None, device: str):
    import torch
    from liquid_audio import LFM2AudioModel

    from s2s_toolcalling.training.lora import inject_lora, load_lora, load_lora_settings, merge_lora

    model = LFM2AudioModel.from_pretrained(base, device=device, dtype=torch.bfloat16).eval()
    if adapter:
        settings = load_lora_settings(adapter)
        inject_lora(model, settings)
        load_lora(model, Path(adapter) / "adapter_model.safetensors")
        merge_lora(model)
    return model


def _base_cache_dir(base: str) -> Path:
    from liquid_audio.utils import get_model_dir

    return Path(get_model_dir(base))


def export_full(base: str, adapter: str | None, output: Path, device: str, n_text: int | None, n_audio: int | None) -> None:
    from safetensors.torch import save_file

    model = _load_merged_model(base, adapter, device)
    state = model.state_dict()
    # {nom_export: nom_source} : retire lora_* ET désencapsule base_layer (sinon
    # le serving cherche ...weight et ne trouve que ...base_layer.weight → KeyError).
    mapping = merged_full_mapping(state.keys())
    save_file({dst: state[src].contiguous().cpu() for dst, src in mapping.items()},
              str(output / "model.safetensors"))

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
        print(f"ratio interleaved écrit : {liquid_config.get('interleaved_n_text')}:{liquid_config.get('interleaved_n_audio')}")


def export_backbone(base: str, adapter: str | None, output: Path, device: str) -> None:
    from safetensors.torch import save_file

    model = _load_merged_model(base, adapter, device)
    state = model.state_dict()
    mapping = remap_backbone_keys(strip_lora_keys(state.keys()))
    save_file({dst: state[src].contiguous().cpu() for src, dst in mapping.items()}, str(output / "model.safetensors"))

    cache = _base_cache_dir(base)
    liquid_config = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    config = build_backbone_config(liquid_config)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for name in BACKBONE_AUX_FILES:
        src = cache / name
        if src.exists():
            shutil.copy2(src, output / name)

    print(f"backbone exporté : {output} ({len(mapping)} tenseurs) — servable par `vllm serve {output}`")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Audio-1.5B", help="modèle de base (repo HF ou chemin)")
    parser.add_argument("--adapter", default=None, help="répertoire final_adapter (poids + adapter_config.json)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["full", "backbone"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interleaved-text-tokens", type=int, default=None, help="ratio calibré (mode full)")
    parser.add_argument("--interleaved-audio-tokens", type=int, default=None, help="ratio calibré (mode full)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        export_full(args.base, args.adapter, output, args.device, args.interleaved_text_tokens, args.interleaved_audio_tokens)
    else:
        export_backbone(args.base, args.adapter, output, args.device)


if __name__ == "__main__":
    main()
