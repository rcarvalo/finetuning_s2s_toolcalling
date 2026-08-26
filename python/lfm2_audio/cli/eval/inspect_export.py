"""``lfm2-eval-inspect`` — export an archived campaign to an Inspect log, then view it.

    lfm2-eval-inspect --archive reports/baseline_en_samples \\
        --scorers dnsmos,utmos,tool_call --out logs/baseline.eval
    inspect view --log-dir logs

Scores the archive with today's scorers (the same code the reports use) and
writes a log whose assistant messages carry the generated audio, so the viewer
plays each reply next to the score it earned. Needs the ``inspect`` extra.
"""

from __future__ import annotations

import argparse
import logging
import sys

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.scoring_config import ScorerConfig, ScoringConfig
from lfm2_audio.evaluation.sample_archive import SampleArchive
from lfm2_audio.inspect_bridge.log_exporter import InspectLogExporter
from lfm2_audio.scorer.factory import ScorerFactory
from lfm2_audio.scorer.registry import SCORERS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", required=True, help="répertoire écrit par `lfm2-evaluate --archive`")
    parser.add_argument("--out", required=True, help="fichier .eval à écrire")
    parser.add_argument("--task", default=None, help="nom de la tâche affiché (défaut : nom de l'archive)")
    parser.add_argument("--model", default="lfm2.5-audio", help="modèle affiché dans le viewer")
    parser.add_argument(
        "--scorers",
        default=None,
        help=f"liste séparée par des virgules (défaut : tous). Connus : {', '.join(SCORERS.names)}",
    )
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="échouer si un scorer demandé est indisponible, au lieu de dégrader l'export",
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

    scored = ((sample, [scorer.score(sample) for scorer in scorers]) for sample in archive.load())
    exporter = InspectLogExporter(
        task=args.task or archive.root.name,
        model=args.model,
        dataset=str(archive.root),
    )
    target = exporter.write(scored, args.out)
    print(f"\n{target}\n\n  inspect view --log-dir {target.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
