"""Packing d'un JSONL de dialogues en dataset HF prêt pour le SFT.

Point d'entrée : ``lfm2-preprocess-sft``.
La logique vit dans :mod:`lfm2_audio.data_prep.preprocess_sft` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from lfm2_audio.core.chat_format import verify_special_tokens
from lfm2_audio.data_prep.liquid_adapter import DialogueChatMessages
from lfm2_audio.tools.schemas import RECEPTION_TOOL_DEFINITIONS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dialogues", required=True, help="JSONL de dialogues")
    parser.add_argument("--audio-root", required=True, help="Racine des fichiers audio référencés")
    parser.add_argument("--output", required=True, help="Répertoire de sortie (save_to_disk)")
    parser.add_argument("--model-id", default="LiquidAI/LFM2.5-Audio-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--interleaved-text-tokens", type=int, default=6)
    parser.add_argument("--interleaved-audio-tokens", type=int, default=12)
    parser.add_argument("--assistant-audio-mode", choices=["interleaved", "sequential"], default="interleaved")
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument(
        "--tool-definitions",
        default=None,
        help="JSON des définitions d'outils à injecter dans le system prompt "
        "(défaut : les outils de l'agent d'accueil, cf. tools.schemas)",
    )
    args = parser.parse_args(argv)



    if args.tool_definitions:
        tool_definitions = json.loads(Path(args.tool_definitions).read_text(encoding="utf-8"))
    else:

        tool_definitions = RECEPTION_TOOL_DEFINITIONS

    processor = LFM2AudioProcessor.from_pretrained(args.model_id, device=args.device).eval()

    # Garde-fou : les marqueurs de tool calling doivent être des tokens uniques.
    checks = verify_special_tokens(processor.text)
    bad = [tok for tok, ok in checks.items() if not ok]
    if bad:
        print(f"ERREUR : tokens spéciaux absents du vocabulaire (multi-token) : {bad}", file=sys.stderr)
        print("Le tokenizer chargé n'est pas celui de LFM2.5 — abandon.", file=sys.stderr)
        raise SystemExit(1)

    mapper = LFM2AudioChatMapper(
        processor,
        interleaved_text_tokens=args.interleaved_text_tokens,
        interleaved_audio_tokens=args.interleaved_audio_tokens,
    )

    # Itérable picklable (≠ générateur) : preprocess_dataset pickle la closure
    # pour le fingerprint datasets — un générateur lèverait « cannot pickle ».
    data = DialogueChatMessages(
        args.dialogues,
        audio_root=args.audio_root,
        tool_definitions=tool_definitions,
        assistant_audio_mode=args.assistant_audio_mode,
    )

    preprocess_dataset(
        data,
        output_path=args.output,
        mapper=mapper,
        max_context_length=args.max_context_length,
    )
    print(f"Dataset pré-packé écrit dans {args.output}")


if __name__ == "__main__":
    main()
