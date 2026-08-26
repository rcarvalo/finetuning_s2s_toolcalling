"""``UtmosScorer`` — MOS de naturalité prédit, sans référence.

UTMOS est entraîné sur des jugements humains de qualité de parole synthétique,
là où DNSMOS vient du débruitage et note surtout la propreté du signal. Sur nos
sorties — de la parole synthétique propre — les deux ne mesurent PAS la même
chose : contre-examen du 26/08/2026 sur 10 réponses du modèle, la corrélation
de rang entre notre DNSMOS et UTMOS est de +0.13, alors qu'UTMOS s'accorde à
+0.70 avec NISQA. Pour juger « est-ce que ça sonne bien », c'est UTMOS qu'il
faut regarder ; DNSMOS reste utile comme indicateur de propreté du signal.

Les poids viennent de ``tarepan/SpeechMOS`` via ``torch.hub`` (~100 Mo, mis en
cache au premier appel) : aucune architecture à vendoriser, et les valeurs
concordent avec l'implémentation de référence VERSA (3.9704 contre 3.97 sur
``vllm_check_1.wav``).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)

HUB_REPO = "tarepan/SpeechMOS"
HUB_MODEL = "utmos22_strong"


class UtmosScorer(BaseScorer):
    """MOS de naturalité prédit par UTMOS, sans référence."""

    name = "utmos"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "MOS de naturalité prédit (UTMOS), sans référence"

    def __init__(self, *, device: str | None = None) -> None:
        self._device = device
        self._model: Any = None

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio

    def skip_reason(self, sample: EvalSample) -> str:
        return "aucun audio généré à noter"

    def measure(self, sample: EvalSample) -> ScoreResult:
        audio = sample.predicted_audio
        if audio is None:
            return ScoreResult.skipped(self.name, "aucun audio généré à noter")

        # UTMOS reçoit la fréquence en argument et rééchantillonne lui-même :
        # lui passer l'audio natif évite un rééchantillonnage de plus.
        signal = torch.from_numpy(audio.samples).unsqueeze(0).to(self._torch_device())
        with torch.no_grad():
            predicted = self._utmos()(signal, audio.sample_rate)
        value = float(torch.as_tensor(predicted).reshape(-1)[0])
        return ScoreResult.ok(self.name, value, details={"duration_s": round(audio.duration_s, 2)})

    def _torch_device(self) -> str:
        return self._device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _utmos(self) -> Any:  # noqa: ANN401 — modèle tiers non typé
        """Modèle UTMOS, chargé au premier usage puis conservé."""
        if self._model is None:
            logger.info("chargement d'UTMOS depuis %s (~100 Mo au premier appel)", HUB_REPO)
            model = torch.hub.load(HUB_REPO, HUB_MODEL, trust_repo=True)  # type: ignore[no-untyped-call]
            self._model = model.to(self._torch_device()).eval()
        return self._model
