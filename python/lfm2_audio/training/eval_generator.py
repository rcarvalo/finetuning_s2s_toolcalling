"""In-training response generator — the piece that makes ScoringCallback real.

``build_trainer`` forwards a ``generator_factory`` to the scoring callback; when
it is ``None`` the callback silently skips every measurement, which is how the
June run trained blind. This module provides that factory for the liquid
backend: it wraps the *live training model* in the same serving stack the final
campaign uses, so the curve at step N and the final report come from identical
code.

Decoding is text-only (interleaved audio shreds tool-call spans) and the system
prompt declares the same tools the packed dataset embedded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lfm2_audio.ds.checkpoint import Layout, ResolvedCheckpoint
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.evaluation.model_generator import ModelResponseGenerator
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.tool_prompt import resolve_system
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)


class TrainingEvalGenerator:
    """Generates with the live model, flipping it to eval mode around each turn.

    The trainer leaves the model in train mode (dropout active, LoRA included);
    scoring through it unchanged would add sampling noise to every point of the
    curve. State is restored afterwards so the next step trains normally.
    """

    def __init__(self, inner: ModelResponseGenerator, model: Any) -> None:
        self._inner = inner
        self._model = model

    def generate(self, question: Question) -> EvalSample:
        was_training = bool(getattr(self._model, "training", False))
        self._model.eval()
        try:
            import torch

            with torch.inference_mode():
                return self._inner.generate(question)
        finally:
            if was_training:
                self._model.train()


def liquid_generator_factory(config: TrainingConfig) -> Any:
    """A ``generator_factory`` for ``build_trainer``, bound to the recipe.

    The processor (audio tokenizer, Mimi) is loaded once and reused across
    scoring rounds — it is frozen, only the model under evaluation changes.
    """
    cache: dict[str, Any] = {}

    def factory(model: Any) -> TrainingEvalGenerator:
        from liquid_audio import LFM2AudioProcessor

        from lfm2_audio.serving.backends.liquid import LiquidAudioBackend

        if "processor" not in cache:
            cache["processor"] = LFM2AudioProcessor.from_pretrained(config.model_id, device="cuda")

        bare = getattr(model, "module", model)  # accelerate may hand us the DDP wrapper
        schedule = config.evaluation
        backend = LiquidAudioBackend(
            ResolvedCheckpoint(path=Path(config.model_id), layout=Layout.LIQUID),
            model=bare,
            processor=cache["processor"],
            system=resolve_system(schedule.tool_definitions),
            generation=GenerationConfig(text_only=True, max_tokens=schedule.max_new_tokens),
        )
        logger.info("in-training scoring generator ready (text-only, tools=%s)", schedule.tool_definitions)
        return TrainingEvalGenerator(ModelResponseGenerator(backend, max_tokens=schedule.max_new_tokens), bare)

    return factory
