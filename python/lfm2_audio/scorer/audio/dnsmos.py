"""``DnsmosScorer`` — qualité perçue de l'audio généré (DNSMOS P.835).

Métrique **sans référence** : elle note le signal seul, sans enregistrement
propre à comparer. C'est ce qui la rend utilisable en cours d'entraînement, où
il n'existe aucune vérité terrain audio.

Trois sous-notes P.835 sur 5 — ``sig`` (qualité de la parole), ``bak`` (bruit de
fond), ``ovrl`` (impression générale) ; c'est ``ovrl`` qui est agrégé.

``onnxruntime`` is imported lazily in :meth:`DnsmosScorer._onnx`: the pure
calibration below must stay importable (and testable) without the ``eval``
extra. Microsoft's ONNX weights are not redistributable, so their absence is
reported with the fix instead of failing the campaign.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from lfm2_audio.scorer.base import BaseScorer
from lfm2_audio.scorer.result import ScoreResult
from lfm2_audio.scorer.sample import EvalSample

if TYPE_CHECKING:
    import onnxruntime

logger = logging.getLogger(__name__)

MODEL_ENV_VAR = "DNSMOS_MODEL_PATH"
DNSMOS_SAMPLE_RATE = 16_000
INPUT_LENGTH_S = 9.01
SUBSCORES = ("sig", "bak", "ovrl")


class DnsmosScorer(BaseScorer):
    """MOS P.835 prédit, sans référence."""

    name = "dnsmos"
    higher_is_better: ClassVar[bool] = True
    description: ClassVar[str] = "MOS P.835 prédit (sig/bak/ovrl), sans référence"

    def __init__(self, model_path: str | Path | None = None) -> None:
        env_path = os.environ.get(MODEL_ENV_VAR)
        self._model_path = Path(model_path) if model_path else (Path(env_path) if env_path else None)
        self._session: onnxruntime.InferenceSession | None = None

    def unavailable_reason(self) -> str | None:
        if self._model_path is None:
            return (
                f"modèle DNSMOS introuvable — poser son chemin dans ${MODEL_ENV_VAR} "
                "(sig_bak_ovr.onnx du dépôt microsoft/DNS-Challenge)"
            )
        if not self._model_path.exists():
            return f"modèle DNSMOS introuvable : {self._model_path}"
        return None

    def supports(self, sample: EvalSample) -> bool:
        return sample.has_predicted_audio

    def skip_reason(self, sample: EvalSample) -> str:
        return "aucun audio généré à noter"

    def measure(self, sample: EvalSample) -> ScoreResult:
        audio = sample.predicted_audio
        if audio is None:
            return ScoreResult.skipped(self.name, "aucun audio généré à noter")

        samples = audio.resample(DNSMOS_SAMPLE_RATE).samples
        window = int(INPUT_LENGTH_S * DNSMOS_SAMPLE_RATE)
        if samples.size < window:  # le modèle attend une fenêtre de taille fixe
            samples = np.pad(samples, (0, window - samples.size))

        # Fenêtres non chevauchantes moyennées : le modèle est calibré sur ~9 s,
        # une réponse plus longue se note par morceaux.
        scores = [self._infer(samples[start : start + window]) for start in range(0, samples.size - window + 1, window)]
        averaged = [float(np.mean([s[i] for s in scores])) for i in range(len(SUBSCORES))]

        details: dict[str, Any] = dict(zip(SUBSCORES, averaged, strict=True))
        details["windows"] = len(scores)
        return ScoreResult.ok(self.name, float(details["ovrl"]), details=details)

    def _infer(self, window: np.ndarray) -> tuple[float, float, float]:
        outputs = self._onnx().run(None, {"input_1": window.reshape(1, -1).astype("float32")})
        raw = outputs[0][0]
        return calibrate_p835(float(raw[0]), float(raw[1]), float(raw[2]))

    def _onnx(self) -> onnxruntime.InferenceSession:
        """Session ONNX, construite au premier usage puis conservée."""
        if self._session is None:
            import onnxruntime

            logger.info("loading DNSMOS model: %s", self._model_path)
            self._session = onnxruntime.InferenceSession(str(self._model_path))
        return self._session


def calibrate_p835(sig_raw: float, bak_raw: float, ovrl_raw: float) -> tuple[float, float, float]:
    """Map raw ONNX outputs onto the published P.835 MOS scale.

    Official non-personalized coefficients from ``dnsmos_local.py``
    (microsoft/DNS-Challenge, ``get_polyfit_val``). They are QUADRATIC: an
    earlier cubic approximation here saturated BAK below 2.0, so clean speech
    scored like noisy speech and OVRL was meaningless.
    """
    sig = -0.08397278 * sig_raw**2 + 1.22083953 * sig_raw + 0.0052439
    bak = -0.13166888 * bak_raw**2 + 1.60915514 * bak_raw - 0.39604546
    ovrl = -0.06766283 * ovrl_raw**2 + 1.11546468 * ovrl_raw + 0.04602535
    return sig, bak, ovrl
