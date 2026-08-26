"""``voice_eval`` — the Inspect task that ties our dataset, model and scorers together.

Everything a campaign used to need from us is now upstream: parallelism,
retries, running several variants in one command, the log and its viewer. What
is left here is the description of *what* to evaluate.

    inspect eval lfm2_audio.inspect_bridge.task \\
      --model lfm2/LiquidAI/LFM2.5-Audio-1.5B \\
      -T questions=benchmark/toolcalling_en/cases.sample.jsonl \\
      -T scorers=tool_call,dnsmos,utmos \\
      --max-samples 1

``--max-samples 1`` is not a detail on a local checkpoint: the samples share one
GPU, so raising it buys nothing and exhausts VRAM. Endpoints are the opposite —
independent workers, where concurrency is the whole point.
"""

from __future__ import annotations

import logging

from inspect_ai import Task, task
from inspect_ai.solver import generate

from lfm2_audio.ds.scoring_config import ScoringConfig
from lfm2_audio.inspect_bridge.dataset import question_set_dataset
from lfm2_audio.inspect_bridge.scorers import lfm2_scorer

logger = logging.getLogger(__name__)

DEFAULT_SCORERS = "tool_call,dnsmos,utmos"


@task
def voice_eval(
    questions: str = "benchmark/toolcalling_en/cases.sample.jsonl",
    audio_root: str | None = None,
    scorers: str = DEFAULT_SCORERS,
    limit: int | None = None,
    asr_language: str = "en",
) -> Task:
    """Answer every case once, then grade it with our own scorers."""
    names = [name.strip() for name in scorers.split(",") if name.strip()]
    # Campaign-level ASR default; a sample carrying metadata["lang"] still wins.
    scoring = ScoringConfig(asr_language=asr_language)
    return Task(
        dataset=question_set_dataset(questions, audio_root=audio_root, limit=limit),
        # One turn per case: no history, so latencies and answers stay
        # comparable from case to case — the invariant the old generators held.
        solver=generate(),
        scorer=[lfm2_scorer(name, scoring=scoring) for name in names],
    )
