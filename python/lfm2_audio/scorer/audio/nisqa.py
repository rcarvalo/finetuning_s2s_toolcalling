"""``NisqaScorer`` — qualité perçue de l'audio généré (NISQA v2).

Seconde métrique MOS sans référence, complémentaire de DNSMOS : NISQA est
entraînée sur des dégradations de transmission (codecs, pertes de paquets) là où
DNSMOS vise le débruitage. Sur de la parole synthétique, elles se trompent
rarement de la même façon — les faire tourner ensemble donne un signal plus
robuste qu'un seul MOS.

Elle produit aussi quatre dimensions : bruit (``noi``), coloration (``col``),
discontinuité (``dis``), intelligibilité (``loud``). Le MOS global est agrégé.

Le checkpoint NISQA (``nisqa.tar``, dépôt gabrielmittag/NISQA) n'est pas
redistribuable : à défaut, le scorer se déclare indisponible.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import torch

from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)

MODEL_ENV_VAR = "NISQA_MODEL_PATH"
NISQA_SAMPLE_RATE = 48_000
DIMENSIONS = ("mos", "noi", "col", "dis", "loud")


class NisqaScorer(BaseScorer):
    """MOS NISQA v2 prédit, sans référence."""

    name = "nisqa"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "MOS NISQA v2 prédit (mos/noi/col/dis/loud), sans référence"

    def __init__(self, model_path: str | Path | None = None, *, device: str | None = None) -> None:
        env_path = os.environ.get(MODEL_ENV_VAR)
        self._model_path = Path(model_path) if model_path else (Path(env_path) if env_path else None)
        self._device = device
        self._model: Any = None

    def unavailable_reason(self) -> str | None:
        if self._model_path is None:
            return (
                f"checkpoint NISQA introuvable — poser son chemin dans ${MODEL_ENV_VAR} "
                "(nisqa.tar du dépôt gabrielmittag/NISQA)"
            )
        if not self._model_path.exists():
            return f"checkpoint NISQA introuvable : {self._model_path}"
        return None

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio

    def skip_reason(self, sample: EvalSample) -> str:
        return "aucun audio généré à noter"

    def measure(self, sample: EvalSample) -> ScoreResult:
        audio = sample.predicted_audio
        if audio is None:
            return ScoreResult.skipped(self.name, "aucun audio généré à noter")

        resampled = audio.resample(NISQA_SAMPLE_RATE)
        with torch.no_grad():
            signal = torch.from_numpy(resampled.samples).unsqueeze(0).to(self._torch_device())
            predictions = self._nisqa()(signal)

        values = [float(v) for v in torch.as_tensor(predictions).reshape(-1)[: len(DIMENSIONS)]]
        details: dict[str, Any] = dict(zip(DIMENSIONS, values, strict=False))
        return ScoreResult.ok(self.name, float(values[0]), details=details)

    def _torch_device(self) -> str:
        return self._device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _nisqa(self) -> Any:  # noqa: ANN401 — modèle tiers non typé
        """Modèle NISQA, chargé au premier usage puis conservé."""
        if self._model is None:
            if self._model_path is None:  # BaseScorer guards via unavailable_reason()
                raise FileNotFoundError(f"checkpoint NISQA non configuré (${MODEL_ENV_VAR})")
            logger.info("chargement du checkpoint NISQA : %s", self._model_path)
            checkpoint = torch.load(self._model_path, map_location=self._torch_device())
            self._model = checkpoint["model"] if isinstance(checkpoint, dict) else checkpoint
            self._model.eval()
        return self._model
