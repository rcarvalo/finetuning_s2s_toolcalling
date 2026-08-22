"""``ReasoningScorer`` — qualité de la réponse, notée par un juge LLM.

Ce que les métriques mécaniques ne voient pas : une réponse peut appeler le bon
outil, avec les bons arguments, un WER parfait — et raconter quelque chose qui
ne découle pas du résultat d'outil. C'est le mode d'échec dominant d'un
assistant qui parle après avoir cherché, et il faut un jugement sémantique pour
l'attraper.

Le juge est injecté (:class:`Judge`) et la rubrique est un objet versionné
(:class:`JudgeRubric`) : deux campagnes ne sont comparables que si elles ont
tourné avec la même version de rubrique, ce que le rapport enregistre.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from lfm2_audio.core.chat_format import TOOL_CALL_END, TOOL_CALL_START
from lfm2_audio.core.prompt import strip_special_tokens
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample
from lfm2_audio.scorer.text.judge import Judge
from lfm2_audio.scorer.text.rubric import REASONING_RUBRIC, JudgeRubric

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", flags=re.DOTALL)

# Le span d'appel EN ENTIER, pas seulement ses marqueurs : retirer les seuls
# `<|tool_call_*|>` laisserait `[web_search(query="…")]`, qui passerait pour une
# réponse en langage naturel et serait envoyé au juge.
_TOOL_CALL_SPAN = re.compile(
    re.escape(TOOL_CALL_START) + r".*?" + re.escape(TOOL_CALL_END),
    flags=re.DOTALL,
)

_PROMPT_TEMPLATE = """You are grading a voice assistant's spoken reply.

Question asked by the user:
{question}

Tool result the assistant had available:
{tool_result}

Assistant's reply:
{answer}

Grade each criterion from 1 to {scale}:
{criteria}

Respond with JSON only, no prose:
{{"scores": {{{keys}}}, "rationale": "one sentence"}}"""


class ReasoningScorer(BaseScorer):
    """Note la réponse sur une rubrique, via un juge LLM."""

    name = "reasoning"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "qualité de la réponse jugée par un LLM (rubrique versionnée)"

    def __init__(self, judge: Judge, *, rubric: JudgeRubric = REASONING_RUBRIC) -> None:
        self._judge = judge
        self._rubric = rubric

    @property
    def rubric(self) -> JudgeRubric:
        return self._rubric

    def supports(self, sample: EvalSample) -> bool:
        # Un tour qui ne fait qu'émettre un tool call n'a pas de réponse parlée
        # à juger : c'est ToolCallScorer qui le mesure.
        return bool(spoken_part(sample.predicted_text))

    def skip_reason(self, sample: EvalSample) -> str:
        return "aucune réponse en langage naturel à juger (tour de tool call seul)"

    def measure(self, sample: EvalSample) -> ScoreResult:
        raw = self._judge.judge(self._build_prompt(sample))
        verdict = _parse_verdict(raw)
        scores = {k: float(v) for k, v in verdict.get("scores", {}).items() if _is_number(v)}

        if not scores:
            return ScoreResult.failed(self.name, f"réponse du juge inexploitable : {raw[:160]}")

        details: dict[str, Any] = dict(scores)
        details["rationale"] = verdict.get("rationale", "")
        details["rubric_version"] = self._rubric.version
        return ScoreResult.ok(self.name, self._rubric.weighted_mean(scores), details=details)

    def _build_prompt(self, sample: EvalSample) -> str:
        tool_result = (
            json.dumps(sample.tool_results, ensure_ascii=False) if sample.tool_results else "(no tool was called)"
        )
        return _PROMPT_TEMPLATE.format(
            question=sample.prompt_text or "(spoken question, transcript unavailable)",
            tool_result=tool_result,
            answer=spoken_part(sample.predicted_text),
            scale=self._rubric.scale,
            criteria=self._rubric.as_prompt_block(),
            keys=", ".join(f'"{key}": 0' for key in self._rubric.keys),
        )


def spoken_part(text: str) -> str:
    """Ce que le modèle a réellement dit : span d'appel et marqueurs retirés.

    Un span d'appel non fermé (stop token strippé par vLLM) est retiré jusqu'à la
    fin, sans quoi son contenu partirait au juge comme s'il s'agissait de parole.
    """
    without_calls = _TOOL_CALL_SPAN.sub(" ", text)
    if TOOL_CALL_START in without_calls:
        without_calls = without_calls[: without_calls.index(TOOL_CALL_START)]
    return strip_special_tokens(without_calls)


def _parse_verdict(raw: str) -> dict[str, Any]:
    """Extrait le JSON de la réponse du juge, tolérant aux fences markdown."""
    match = _JSON_BLOCK.search(raw)
    if not match:
        return {}
    try:
        parsed: dict[str, Any] = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.debug("verdict du juge illisible : %s", raw[:200])
        return {}
    return parsed


def _is_number(value: Any) -> bool:  # noqa: ANN401 — valeur JSON arbitraire du juge
    return isinstance(value, (int, float)) and not isinstance(value, bool)
