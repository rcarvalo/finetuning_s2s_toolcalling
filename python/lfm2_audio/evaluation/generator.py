"""``ResponseGenerator`` — ce qui produit une réponse à évaluer.

Un ``Protocol`` : la pipeline orchestre des scorers autour d'une source de
réponses, sans savoir si elles viennent d'un modèle chargé, d'un endpoint
distant ou d'un fichier de prédictions rejouées. C'est ce qui la rend testable
sans GPU.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lfm2_audio.evaluation.question import Question
from lfm2_audio.scorer.sample import EvalSample


@runtime_checkable
class ResponseGenerator(Protocol):
    """Répond à une question et rend l'échantillon à noter."""

    def generate(self, question: Question) -> EvalSample: ...
