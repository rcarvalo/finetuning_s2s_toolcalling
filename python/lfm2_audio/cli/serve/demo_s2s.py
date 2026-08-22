"""Démo speech-to-speech : audio et/ou texte en entrée, texte + WAV en sortie.

    lfm2-demo --checkpoint exports/lfm25_audio_fr --audio-in question.wav
    lfm2-demo --checkpoint Rcarvalo/lfm25-tc-en-s2s --text "Hello, who are you?"
    lfm2-demo --checkpoint exports/lfm25_audio_fr --interactive

En interactif : tape du texte, ``@/chemin/audio.wav`` pour parler, ``/reset``
pour vider l'historique, Ctrl-D pour sortir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.evaluation.latency import format_ms
from lfm2_audio.serving import LFM2Audio

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="répertoire local, repo Hugging Face, ou répertoire d'adaptateur LoRA",
    )
    parser.add_argument("--adapter", default=None, help="adaptateur LoRA à fusionner")
    parser.add_argument("--backend", choices=["auto", "vllm", "liquid"], default="auto")
    parser.add_argument("--audio-in", type=Path, default=None, help="WAV d'entrée")
    parser.add_argument("--text", default=None, help="texte d'entrée (seul ou en complément)")
    parser.add_argument("--out", type=Path, default=Path("out/reply.wav"))
    parser.add_argument("--system", default="Respond with interleaved text and audio.")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="génération texte seule (tool calls : évite que l'audio shredde le span)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="tours de chauffe avant le tour mesuré (JIT Triton, autotuning CUDA graphs)",
    )
    parser.add_argument("--interactive", action="store_true", help="boucle multi-tours")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def run_turn(model: LFM2Audio, text: str | None, audio: Path | None, out: Path) -> None:
    """Un tour, puis affichage du texte, de l'audio et des latences."""
    reply = model.reply(text=text, audio=audio)
    print(f"\n🤖 {reply.text}")

    if reply.audio is None or reply.audio.is_empty:
        print("(pas d'audio généré)")
        return

    saved = reply.audio.save(out)
    ttfa = f"TTFA={format_ms(reply.metrics.ttfa_s)}  " if reply.metrics.ttfa_s else ""
    rtf = reply.real_time_factor
    print(
        f"🔊 {saved}  ({reply.audio.duration_s:.1f}s d'audio — "
        f"{ttfa}total={reply.metrics.total_s:.2f}s" + (f", RTF={rtf:.2f})" if rtf is not None else ")")
    )


def interactive_loop(model: LFM2Audio, out: Path) -> None:
    print("Interactif — texte, `@/chemin/audio.wav`, `/reset`, ou Ctrl-D pour quitter.")
    for index, turn in enumerate(_read_lines()):
        if turn == "/reset":
            model.reset()
            print("(historique vidé)")
            continue
        text, audio = (None, Path(turn[1:])) if turn.startswith("@") else (turn, None)
        try:
            run_turn(model, text, audio, out.with_stem(f"{out.stem}_{index:02d}"))
        except (Lfm2AudioError, ValueError, FileNotFoundError) as error:
            print(f"⚠️  {error}")


def _read_lines() -> Iterator[str]:
    while True:
        try:
            line = input("\n👤 > ").strip()
        except EOFError:
            return
        if line:
            yield line


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.interactive and args.text is None and args.audio_in is None:
        print("--text et/ou --audio-in requis (ou --interactive)", file=sys.stderr)
        return 2

    try:
        model = LFM2Audio.from_pretrained(
            args.checkpoint,
            backend=args.backend,
            adapter=args.adapter,
            system=args.system,
            generation=GenerationConfig(max_tokens=args.max_tokens, text_only=args.text_only),
        )
    except Lfm2AudioError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    with model:
        for index in range(args.warmup):
            model.reply(text=args.text, audio=args.audio_in)
            model.reset()
            print(f"[warmup {index + 1}/{args.warmup}]")

        if args.interactive:
            interactive_loop(model, args.out)
        else:
            run_turn(model, args.text, args.audio_in, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
