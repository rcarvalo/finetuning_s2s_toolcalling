"""``InspectLogExporter`` — a scored archive becomes a log the Inspect viewer opens.

Offline by construction: it reads what a campaign already produced (a
:class:`~lfm2_audio.evaluation.sample_archive.SampleArchive` plus the scores)
and writes an ``.eval`` file. No model, no GPU, no network — so a run can be
re-examined long after the machine that produced it is gone, which is precisely
what the archive exists for.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
    write_eval_log,
)

from lfm2_audio.inspect_bridge.scores import to_inspect_score, unmeasured_note
from lfm2_audio.inspect_bridge.transcript import InspectTranscript
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample as OurSample

logger = logging.getLogger(__name__)


class InspectLogExporter:
    """Writes samples and their scores as an Inspect ``.eval`` log."""

    def __init__(self, *, task: str, model: str, dataset: str = "") -> None:
        self._task = task
        self._model = model
        self._dataset = dataset

    def write(
        self,
        scored: Iterable[tuple[OurSample, Sequence[ScoreResult]]],
        destination: str | Path,
    ) -> Path:
        """Write the log and return where it landed."""
        samples = [self._sample(sample, results) for sample, results in scored]
        now = datetime.now(UTC).isoformat()
        log = EvalLog(
            version=2,
            status="success",
            eval=EvalSpec(
                created=now,
                task=self._task,
                dataset=EvalDataset(name=self._dataset or self._task, samples=len(samples)),
                model=self._model,
                config=EvalConfig(),
            ),
            plan=EvalPlan(),
            results=EvalResults(total_samples=len(samples), completed_samples=len(samples)),
            stats=EvalStats(started_at=now, completed_at=now),
            samples=samples,
        )
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(target))
        logger.info("%d échantillons exportés vers %s", len(samples), target)
        return target

    def _sample(self, sample: OurSample, results: Sequence[ScoreResult]) -> EvalSample:
        scores = {}
        unmeasured: dict[str, str] = {}
        for result in results:
            score = to_inspect_score(result)
            if score is not None:
                scores[result.scorer] = score
            else:
                unmeasured[result.scorer] = unmeasured_note(result)

        metadata: dict[str, Any] = dict(sample.metadata)
        if sample.expected_calls:
            metadata["expected_calls"] = sample.expected_calls
        if unmeasured:
            # Surfaced rather than silent: a viewer showing four scores where a
            # sibling run showed five must say which metric went missing.
            metadata["unmeasured"] = unmeasured

        return EvalSample(
            id=sample.sample_id,
            epoch=1,
            input=sample.prompt_text or sample.sample_id,
            target=sample.reference_text,
            messages=InspectTranscript(sample).messages(),
            scores=scores,
            metadata=metadata,
        )
