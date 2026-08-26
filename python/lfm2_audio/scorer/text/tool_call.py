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

from typing import ClassVar

from lfm2_audio.evaluation.argument_match import ArgMatch
from lfm2_audio.evaluation.tool_call_diagnosis import ToolCallDiagnosis
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
        # The value is turn success, unchanged; `details` carries the anatomy of
        # the failure, so a report can say WHICH argument diverged instead of
        # only that the call was wrong.
        diagnosis = ToolCallDiagnosis.of(
            sample.sample_id,
            sample.predicted_text,
            sample.expected_calls,
            arg_match=self._arg_match,
            threshold=self._threshold,
        )
        return ScoreResult.ok(self.name, float(diagnosis.succeeded), details=diagnosis.as_details())
