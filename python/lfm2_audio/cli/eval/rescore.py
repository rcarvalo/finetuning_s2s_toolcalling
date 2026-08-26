"""``lfm2-eval-rescore`` — renoter une campagne archivée, sans GPU ni régénération.

Une métrique fausse ne doit pas coûter une seconde campagne. Quand
``lfm2-evaluate --archive DIR`` a conservé les réponses, cette commande leur
applique les scorers d'aujourd'hui : c'est le chemin pour refaire une baseline
après une correction de métrique (le cas DNSMOS du 24/08/2026, où l'audio
manquait et les chiffres étaient devenus incomparables).

    lfm2-eval-rescore --archive reports/baseline_en_samples \\
        --scorers dnsmos,utmos --out reports/baseline_en_audio_v2.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.sample_archive import SampleArchive
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.scorer.registry import SCORERS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True, help="répertoire écrit par `lfm2-evaluate --archive`")
    parser.add_argument("--out", default=None, help="rapport JSON à écrire")
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
    return parser


def build_scoring_config(args: argparse.Namespace) -> ScoringConfig:
    if args.scorers is None:
        return ScoringConfig.with_defaults().model_copy(update={"fail_on_unavailable": args.fail_on_unavailable})
    names = [name.strip() for name in args.scorers.split(",") if name.strip()]
    return ScoringConfig(
        scorers=tuple(ScorerConfig(name=name) for name in names),
        fail_on_unavailable=args.fail_on_unavailable,
    )


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    archive = SampleArchive(args.archive)
    if len(archive) == 0:
        print(f"❌ archive vide ou introuvable : {archive.root}", file=sys.stderr)
        return 1

    try:
        scorers = ScorerFactory(build_scoring_config(args)).build_all()
    except Lfm2AudioError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    samples = list(archive.load())
    logger.info("%d échantillons relus depuis %s", len(samples), archive.root)
    report = EvaluationPipeline(scorers).score(
        samples,
        # Rappeler d'où viennent les réponses : un rapport renoté ne doit pas
        # se confondre avec une campagne fraîche.
        context={"rescored_from": str(archive.root), "cases": len(samples)},
    )

    print()
    print(report.render())
    if args.out:
        logger.info("rapport écrit : %s", report.write_json(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
