"""Correspondance des clés et des configs entre layouts de checkpoint.

Python pur : c'est la logique qui décide quel poids atterrit sous quel nom, donc
celle qui casse silencieusement un export si elle se trompe. Elle doit être
testable sans torch, et elle l'est.

Trois cibles :
- ``remap_backbone_keys`` — sous-arbre ``lfm.`` vers un ``Lfm2ForCausalLM``
  servable par ``vllm serve`` (texte seul) ;
- ``merged_full_mapping`` — export complet, adaptateurs LoRA déjà fusionnés ;
- ``build_backbone_config`` / ``update_interleaved_ratio`` — le ``config.json``
  qui accompagne les poids. Le ratio interleaved y est écrit : c'est la source
  unique entre entraînement et serving.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from lfm2_audio.core.errors import ExportError

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
