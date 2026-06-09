"""CLI Phase 2 : prépare le dataset SFT au format disque liquid-audio.

Convertit un JSONL de dialogues (cf. ``dialogue_schema``) en dataset HF
pré-packé via ``LFM2AudioChatMapper`` (tokens texte + mels + codes Mimi +
masques de supervision), prêt pour ``LFM2DataLoader``.

Usage :

    python -m s2s_toolcalling.data.preprocess_sft \\
        --dialogues data/dialogues_train.jsonl \\
        --audio-root data/audio \\
        --output datasets/sft_train \\
        --interleaved-text-tokens 6 --interleaved-audio-tokens 10

Le ratio interleaved FR se calibre avec ``scripts/calibrate_interleaved_ratio.py``
(recette du variant japonais : 6:9 ; anglais : 6:12).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

    from liquid_audio import LFM2AudioProcessor
    from liquid_audio.data.mapper import LFM2AudioChatMapper
    from liquid_audio.data.preprocess import preprocess_dataset

    from s2s_toolcalling.data.chat_format import verify_special_tokens
    from s2s_toolcalling.data.dialogue_schema import load_dialogues
    from s2s_toolcalling.data.liquid_adapter import dialogues_to_chat_messages

    if args.tool_definitions:
        tool_definitions = json.loads(Path(args.tool_definitions).read_text(encoding="utf-8"))
    else:
        from s2s_toolcalling.tools.schemas import RECEPTION_TOOL_DEFINITIONS

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

    data = dialogues_to_chat_messages(
        load_dialogues(args.dialogues),
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
