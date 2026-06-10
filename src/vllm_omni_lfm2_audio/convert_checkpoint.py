"""Conversion du checkpoint exporté (s2s_toolcalling export --mode full) vers
le layout attendu par le plugin vLLM-Omni.

Le checkpoint liquid-audio mergé est déjà au bon format de poids (préfixes
``lfm.``, ``conformer.``, ``audio_adapter.``, ``depthformer.``, ``depth_*``,
``audio_embedding.`` + ``audio_detokenizer/``) ; la conversion ne touche que le
``config.json`` :

- ``architectures: ["Lfm2AudioOmniModel"]`` + ``model_type: "lfm2_audio"``
  (clé de routage du registre et du pipeline) ;
- ids placeholders du flux audio (cf. ``constants``) ;
- conservation des sections liquid (lfm/encoder/depthformer/preprocessor) et
  du ratio interleaved calibré.

Usage :
    python -m vllm_omni_lfm2_audio.convert_checkpoint \\
        --checkpoint exports/lfm25_audio_fr --output exports/lfm25_audio_fr_omni
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from vllm_omni_lfm2_audio.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    DEFAULT_INTERLEAVED_N_AUDIO,
    DEFAULT_INTERLEAVED_N_TEXT,
)

OMNI_ARCHITECTURE = "Lfm2AudioOmniModel"
OMNI_MODEL_TYPE = "lfm2_audio"

REQUIRED_LIQUID_SECTIONS = ("lfm", "encoder", "depthformer", "preprocessor")


class ConversionError(ValueError):
    pass


def build_omni_config(
    liquid_config: dict[str, Any],
    *,
    audio_frame_token_id: int = AUDIO_FRAME_PLACEHOLDER_ID,
    audio_eoa_token_id: int = AUDIO_EOA_PLACEHOLDER_ID,
) -> dict[str, Any]:
    """config.json vLLM-Omni depuis le config liquid-audio (fonction pure)."""
    missing = [s for s in REQUIRED_LIQUID_SECTIONS if s not in liquid_config]
    if missing:
        raise ConversionError(f"liquid config is missing sections: {missing} — is this a full export?")

    config = dict(liquid_config)
    config["architectures"] = [OMNI_ARCHITECTURE]
    config["model_type"] = OMNI_MODEL_TYPE
    config.setdefault("interleaved_n_text", DEFAULT_INTERLEAVED_N_TEXT)
    config.setdefault("interleaved_n_audio", DEFAULT_INTERLEAVED_N_AUDIO)
    if audio_frame_token_id == audio_eoa_token_id:
        raise ConversionError("audio_frame_token_id and audio_eoa_token_id must differ")
    config["audio_frame_token_id"] = audio_frame_token_id
    config["audio_eoa_token_id"] = audio_eoa_token_id
    return config


def convert(checkpoint: Path, output: Path, *, frame_id: int, eoa_id: int) -> None:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise ConversionError(f"{config_path} not found")
    liquid_config = json.loads(config_path.read_text(encoding="utf-8"))
    omni_config = build_omni_config(liquid_config, audio_frame_token_id=frame_id, audio_eoa_token_id=eoa_id)

    output.mkdir(parents=True, exist_ok=True)
    for item in checkpoint.iterdir():
        if item.name == "config.json":
            continue
        if item.is_dir():
            shutil.copytree(item, output / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, output / item.name)
    (output / "config.json").write_text(json.dumps(omni_config, indent=2), encoding="utf-8")

    ratio = f"{omni_config['interleaved_n_text']}:{omni_config['interleaved_n_audio']}"
    print(f"checkpoint vLLM-Omni écrit : {output} (ratio interleaved {ratio})")
    print(f"servir avec : vllm-omni serve {output}  (plugin vllm_omni_lfm2_audio installé)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="export complet (s2s_toolcalling export --mode full)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-frame-token-id", type=int, default=AUDIO_FRAME_PLACEHOLDER_ID)
    parser.add_argument("--audio-eoa-token-id", type=int, default=AUDIO_EOA_PLACEHOLDER_ID)
    args = parser.parse_args()

    convert(
        Path(args.checkpoint),
        Path(args.output),
        frame_id=args.audio_frame_token_id,
        eoa_id=args.audio_eoa_token_id,
    )


if __name__ == "__main__":
    main()
