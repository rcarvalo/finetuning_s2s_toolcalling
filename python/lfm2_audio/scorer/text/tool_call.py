"""``ToolCallScorer`` — exactitude des appels d'outils émis dans le flux texte.

Réutilise le scoring BFCL-style déjà en place
(:mod:`lfm2_audio.evaluation.toolcalling`) et l'expose sous le contrat commun,
de sorte que la même métrique serve à la campagne d'éval et au suivi
d'entraînement.

Quatre facettes, chacune reportée dans ``details`` :

- ``parse`` — le span ``<|tool_call_start|>…`` est syntaxiquement exploitable ;
- ``relevance`` — appeler ou s'abstenir, décision correcte (les négatifs
  comptent : un modèle qui appelle toujours a une exactitude de nom parfaite et
  reste inutilisable) ;
- ``name`` — le bon outil ;
- ``call`` — le bon outil *avec* les bons arguments.

La valeur agrégée est **le tour réussi**, pas ``call`` brut : ``call_correct``
est faux par construction sur un cas négatif, si bien qu'agréger cette facette
seule punirait chaque abstention correcte. Un tour est réussi si le modèle a
appelé le bon outil avec les bons arguments quand il le fallait, **ou** s'est
abstenu quand il le fallait.
"""

from __future__ import annotations

from typing import Any, ClassVar

from lfm2_audio.evaluation.toolcalling import ArgMatch, score_case
from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample


class ToolCallScorer(BaseScorer):
    """Exactitude d'appel d'outil, à la BFCL."""

    name = "tool_call"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "exactitude des tool calls (parse/relevance/name/call)"

    def __init__(self, *, arg_match: ArgMatch = "token_f1", threshold: float = 0.7) -> None:
        self._arg_match = arg_match
        self._threshold = threshold

    def supports(self, sample: EvalSample) -> bool:
        # Les négatifs (aucun appel attendu) sont DANS le périmètre : c'est là
        # que se mesure l'abstention. On n'exclut que l'absence de génération.
        return bool(sample.predicted_text)

    def skip_reason(self, sample: EvalSample) -> str:
        return "aucun texte généré à analyser"

    def measure(self, sample: EvalSample) -> ScoreResult:
        case = score_case(
            sample.sample_id,
            sample.predicted_text,
            sample.expected_calls,
            arg_match=self._arg_match,
            threshold=self._threshold,
        )
        # La pertinence n'est pas un champ de CaseResult : c'est l'accord entre
        # « un appel était attendu » et « un appel a été émis ».
        relevant = case.expected_call == case.predicted_call
        succeeded = case.call_correct if case.expected_call else not case.predicted_call

        details: dict[str, Any] = {
            "parse": case.parsed,
            "relevance": relevant,
            "name": case.name_correct,
            "call": case.call_correct,
            "expected_call": case.expected_call,
            "predicted_call": case.predicted_call,
        }
        return ScoreResult.ok(self.name, float(succeeded), details=details)
