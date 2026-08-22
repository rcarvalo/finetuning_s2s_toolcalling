"""Conversion d'un checkpoint liquid-audio vers le layout vLLM-Omni.

Point d'entrée : ``lfm2-convert-checkpoint``.
La logique vit dans :mod:`lfm2_audio.vllm_plugin.convert_checkpoint` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lfm2_audio.vllm_plugin.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
)
from lfm2_audio.vllm_plugin.convert_checkpoint import (
    convert,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="export complet (lfm2_audio export --mode full)")
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
