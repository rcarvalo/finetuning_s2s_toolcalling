"""``Campaign`` — every variant of a config, collected, scored and reported.

One config, one command: each variant produces its own run directory holding
the samples, the audio, the trajectories and the scores, and the campaign ends
with a comparison table. Two runs of the same campaign differ only by the
variant, because everything else came from the same file.

Parallelism is per variant and bounded by ``max_parallel``. Threads, not
processes: a local variant spends its time in torch and a remote one in HTTP,
and both release the GIL — while a process pool would reload the model, or ship
audio through a pickle for nothing. The default of 1 is deliberate: local
variants share a GPU, and running two at once mostly makes both slower.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lfm2_audio.ds.campaign_config import CampaignConfig, VariantConfig
from lfm2_audio.evaluation.generator import ResponseGenerator
from lfm2_audio.evaluation.pipeline import EvaluationPipeline
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.evaluation.report import EvaluationReport
from lfm2_audio.evaluation.sample_archive import SampleArchive
from lfm2_audio.scorer.factory import ScorerFactory

logger = logging.getLogger(__name__)

GeneratorFactory = Callable[[VariantConfig], ResponseGenerator]


@dataclass(frozen=True, slots=True)
class VariantOutcome:
    """What one variant produced — or why it produced nothing."""

    name: str
    archive: Path
    report: EvaluationReport | None = None
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.report is not None


class Campaign:
    """Runs every variant of a :class:`CampaignConfig` and keeps their outcomes."""

    def __init__(self, config: CampaignConfig, generator_factory: GeneratorFactory) -> None:
        self._config = config
        self._generator_factory = generator_factory

    def run(self) -> list[VariantOutcome]:
        """Evaluate every variant; a variant that fails does not take the others down."""
        questions = self._questions()
        logger.info(
            "campagne : %d variantes sur %d cas (parallélisme %d)",
            len(self._config.variants),
            len(questions),
            self._config.max_parallel,
        )
        if self._config.max_parallel == 1:
            return [self._run_variant(variant, questions) for variant in self._config.variants]

        with ThreadPoolExecutor(max_workers=self._config.max_parallel) as pool:
            futures = [pool.submit(self._run_variant, variant, questions) for variant in self._config.variants]
            return [future.result() for future in futures]

    def _questions(self) -> QuestionSet:
        questions = QuestionSet.from_jsonl(self._config.questions, audio_root=self._config.audio_root)
        return questions.take(self._config.limit) if self._config.limit else questions

    def _run_variant(self, variant: VariantConfig, questions: QuestionSet) -> VariantOutcome:
        archive_root = Path(self._config.runs_root) / variant.name / "samples"
        try:
            # Scorers are built per variant: the judge and the transcriber hold
            # models, and sharing them across threads is not part of their contract.
            scorers = ScorerFactory(self._config.scoring).build_all()
            generator = self._generator_factory(variant)
            report = EvaluationPipeline(scorers).run(
                questions,
                generator,
                context=self._context(variant),
                archive=SampleArchive(archive_root),
            )
        except Exception as error:  # une variante qui casse ne perd pas les autres
            logger.warning("variante %s en échec : %s", variant.name, error)
            return VariantOutcome(variant.name, archive_root, error=f"{type(error).__name__}: {error}")

        failures, first_error = self._generation_failures(archive_root)
        if failures and failures == len(questions):
            # The pipeline turns a per-question crash into an empty sample so one
            # bad case cannot lose the rest. A variant where EVERY case did that
            # produced no measurement at all, and must not appear in the
            # comparison as a run that merely scored badly.
            reason = f"aucune génération n'a abouti ({failures}/{len(questions)} cas) : {first_error}"
            logger.warning("variante %s : %s", variant.name, reason)
            return VariantOutcome(variant.name, archive_root, error=reason)

        report.write_json(Path(self._config.runs_root) / variant.name / "report.json")
        logger.info(
            "variante %s terminée (%d/%d cas générés) → %s",
            variant.name,
            len(questions) - failures,
            len(questions),
            archive_root.parent,
        )
        return VariantOutcome(variant.name, archive_root, report=report)

    @staticmethod
    def _generation_failures(archive_root: Path) -> tuple[int, str]:
        """How many cases failed to generate, and the first reason why.

        The count alone reads as a verdict without a cause; the underlying
        exception is what a reader actually needs to fix the variant.
        """
        errors = [
            str(sample.metadata["generation_error"])
            for sample in SampleArchive(archive_root).load()
            if "generation_error" in sample.metadata
        ]
        return len(errors), errors[0] if errors else ""

    def _context(self, variant: VariantConfig) -> dict[str, object]:
        """Everything a later reader needs to know this run was comparable."""
        return {
            "variant": variant.name,
            "checkpoint": variant.checkpoint,
            "adapter": variant.adapter,
            "backend": variant.backend,
            "endpoint": variant.endpoint,
            "max_tokens": variant.max_tokens,
            "questions": self._config.questions,
            "tool_definitions": self._config.tool_definitions,
        }
