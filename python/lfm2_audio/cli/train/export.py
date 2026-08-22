"""Export du modèle fine-tuné : merge LoRA → checkpoint déployable.

Point d'entrée : ``lfm2-export-checkpoint``.
La logique vit dans :mod:`lfm2_audio.training.export_checkpoint` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lfm2_audio.training.export_checkpoint import (
    export_backbone,
    export_full,
)


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
        export_full(
            args.base,
            args.adapter,
            output,
            args.device,
            args.interleaved_text_tokens,
            args.interleaved_audio_tokens,
        )
    else:
        export_backbone(args.base, args.adapter, output, args.device)


if __name__ == "__main__":
    main()
