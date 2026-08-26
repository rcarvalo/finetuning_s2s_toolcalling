"""Campagne d'évaluation : qualité audio et qualité des réponses.

    lfm2-evaluate --checkpoint exports/tc_en --questions benchmark/toolcalling_en/cases.jsonl
    lfm2-evaluate --list-scorers

Métriques disponibles : WER (audio re-transcrit), DNSMOS et NISQA (MOS prédit
sans référence), tool calling (BFCL-style), raisonnement (juge LLM). Celles dont
les dépendances manquent sont reportées ``unavailable`` avec la marche à suivre —
la campagne n'échoue pas pour autant.

La logique vit dans :mod:`lfm2_audio.evaluation` et :mod:`lfm2_audio.scorer` —
ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
import logging
import sys

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.evaluation.model_generator import ModelResponseGenerator
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.evaluation.sample_archive import SampleArchive
from lfm2_audio.evaluation.tool_prompt import resolve_system
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.scorer.registry import SCORERS
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="chemin local, repo HF, ou adaptateur LoRA")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["auto", "vllm", "liquid"], default="auto")
    parser.add_argument("--questions", help="JSONL du jeu d'évaluation")
    parser.add_argument("--audio-root", default=None, help="racine des WAV référencés")
    parser.add_argument("--out", default=None, help="rapport JSON à écrire")
    parser.add_argument("--limit", type=int, default=None, help="n'évaluer que les N premiers cas")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--scorers",
        default=None,
        help=f"liste séparée par des virgules (défaut : tous). Connus : {', '.join(SCORERS.names)}",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="échouer si un scorer demandé est indisponible, au lieu de dégrader le rapport",
    )
    parser.add_argument(
        "--tool-definitions",
        default=None,
        help=(
            "JSON des outils à DÉCLARER dans le system prompt (comme à l'entraînement). "
            "Sans lui le modèle ignore que des outils existent et ne peut pas les appeler. "
            "Raccourci : 'en' pour le set web_search + db_query."
        ),
    )
    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "répertoire où conserver l'audio et les textes produits. Permet de renoter la "
            "campagne SANS GPU quand une métrique change (`lfm2-eval-rescore`)."
        ),
    )
    parser.add_argument("--list-scorers", action="store_true", help="lister les métriques et sortir")
    return parser


def print_scorers() -> int:
    """Catalogue des métriques et leur disponibilité sur cette machine."""
    width = max(len(name) for name in SCORERS.names)
    for spec in SCORERS.specs():
        reason = spec.unavailable_reason()
        state = "disponible" if reason is None else reason
        print(f"  {spec.name:<{width}}  {state}")
        print(f"  {'':<{width}}  {spec.description}")
    return 0


def build_scoring_config(args: argparse.Namespace) -> ScoringConfig:
    """Config de scoring : tous les scorers, ou la sélection demandée."""
    if args.scorers is None:
        base = ScoringConfig.with_defaults()
        return base.model_copy(update={"fail_on_unavailable": args.fail_on_unavailable})

    names = [name.strip() for name in args.scorers.split(",") if name.strip()]
    return ScoringConfig(
        scorers=tuple(ScorerConfig(name=name) for name in names),
        fail_on_unavailable=args.fail_on_unavailable,
    )


AUDIO_SCORERS = frozenset({"wer", "dnsmos", "utmos", "nisqa"})
"""Scorers that grade the generated audio — they need interleaved generation."""


def build_generation(config: ScoringConfig) -> GenerationConfig:
    """Decoding mode of the campaign, derived from what the scorers measure.

    On the liquid backend, interleaved audio placeholders SHRED a structured
    tool-call span — `web_search(query="…")` comes out mangled and every
    argument reads as wrong. A campaign that grades no audio therefore decodes
    text-only; one that grades audio keeps interleaving and accepts that any
    tool-call metric in the same run is unreliable.
    """
    text_only = not (set(config.enabled_names) & AUDIO_SCORERS)
    if text_only:
        logger.info("no audio scorer requested — decoding text-only to keep tool-call spans intact")
    return GenerationConfig(text_only=text_only)


def build_system(args: argparse.Namespace) -> str:
    """System prompt of the campaign — same resolver as the training callback."""
    return resolve_system(args.tool_definitions)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list_scorers:
        return print_scorers()

    if not args.checkpoint or not args.questions:
        print("--checkpoint et --questions sont requis", file=sys.stderr)
        return 2

    questions = QuestionSet.from_jsonl(args.questions, audio_root=args.audio_root).take(args.limit)
    logger.info("%d questions (%d avec appel attendu)", len(questions), questions.positives)

    try:
        scoring_config = build_scoring_config(args)
        scorers = ScorerFactory(scoring_config).build_all()
        model = LFM2Audio.from_pretrained(
            args.checkpoint,
            backend=args.backend,
            adapter=args.adapter,
            system=build_system(args),
            generation=build_generation(scoring_config),
        )
    except Lfm2AudioError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    with model:
        report = EvaluationPipeline(scorers).run(
            questions,
            ModelResponseGenerator(model, max_tokens=args.max_tokens),
            context={
                "checkpoint": args.checkpoint,
                "backend": model.backend_name,
                "adapter": args.adapter,
                # Two campaigns that declared different tools are not comparable.
                "tool_definitions": args.tool_definitions,
            },
            archive=SampleArchive(args.archive) if args.archive else None,
        )

    print()
    print(report.render())
    if args.out:
        logger.info("rapport écrit : %s", report.write_json(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
