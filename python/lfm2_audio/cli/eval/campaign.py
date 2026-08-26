"""``lfm2-campaign`` — run every variant of a config: collect, score, report.

    lfm2-campaign --config configs/eval/versions.yaml

Each variant lands in its own run directory (samples, audio, trajectories,
scores, report), and the campaign ends with the comparison table. One file
describes the dataset, the scorers and the variants, so a difference between
two runs can only come from the variant itself.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.ds.campaign_config import CampaignConfig, VariantConfig
from lfm2_audio.evaluation.campaign import Campaign, GeneratorFactory, VariantOutcome
from lfm2_audio.evaluation.comparison import compare_reports
from lfm2_audio.evaluation.endpoint_generator import EndpointResponseGenerator
from lfm2_audio.evaluation.generator import ResponseGenerator
from lfm2_audio.evaluation.model_generator import ModelResponseGenerator
from lfm2_audio.evaluation.tool_prompt import resolve_system
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML de campagne (dataset, scorers, variantes)")
    parser.add_argument("--dry-run", action="store_true", help="valider la config et lister les variantes, sans lancer")
    return parser


def _describe(variant: VariantConfig) -> str:
    """What a variant points at, for the dry run."""
    if variant.endpoint:
        return f"endpoint:{variant.endpoint}"
    return f"{variant.checkpoint} (+{variant.adapter})" if variant.adapter else str(variant.checkpoint)


def make_generator(config: CampaignConfig) -> GeneratorFactory:
    """Builds the generator a variant needs — endpoint or loaded checkpoint."""

    def factory(variant: VariantConfig) -> ResponseGenerator:
        if variant.endpoint or variant.checkpoint is None:
            # VariantConfig guarantees exactly one source; the None check is what
            # tells the type checker which branch we are in.
            client = LiquidAudioClient(str(variant.endpoint))
            return EndpointResponseGenerator(client, max_tokens=variant.max_tokens)
        model = LFM2Audio.from_pretrained(
            variant.checkpoint,
            adapter=variant.adapter,
            backend=variant.backend,
            system=resolve_system(config.tool_definitions),
        )
        return ModelResponseGenerator(model, max_tokens=variant.max_tokens)

    return factory


def render(outcomes: list[VariantOutcome]) -> None:
    """The campaign's summary: what each variant scored, and what broke."""
    for outcome in outcomes:
        if not outcome.succeeded:
            print(f"\n❌ {outcome.name} : {outcome.error}")
            continue
        print(f"\n=== {outcome.name}")
        print(outcome.report.render() if outcome.report else "")

    scored = [o for o in outcomes if o.succeeded and o.report]
    if len(scored) < 2:
        return
    baseline, *candidates = scored
    for candidate in candidates:
        comparison = compare_reports(baseline.report.as_dict(), candidate.report.as_dict())  # type: ignore[union-attr]
        print(f"\n=== {baseline.name} → {candidate.name}")
        print(comparison.to_markdown())


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    try:
        config = CampaignConfig.from_yaml(args.config)
    except (ValueError, OSError) as error:
        print(f"❌ config illisible : {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"{len(config.variants)} variantes sur {config.questions} (parallélisme {config.max_parallel})")
        for variant in config.variants:
            print(f"  {variant.name:<20} {_describe(variant)}")
        return 0

    try:
        outcomes = Campaign(config, make_generator(config)).run()
    except Lfm2AudioError as error:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    render(outcomes)
    failed = [o.name for o in outcomes if not o.succeeded]
    if failed:
        print(f"\n⚠️  variantes en échec : {', '.join(failed)}", file=sys.stderr)
    print(f"\nruns écrits dans {Path(config.runs_root).resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
