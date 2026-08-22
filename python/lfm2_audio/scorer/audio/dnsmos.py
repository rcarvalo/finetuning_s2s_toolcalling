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

# Non-personalised polyfit from the reference implementation.
SIG_POLY = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
BAK_POLY = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
OVRL_POLY = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


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
            return ScoreResult.skipped(self.name, "no generated audio to score")

        samples = audio.resample(DNSMOS_SAMPLE_RATE).samples
        window = int(INPUT_LENGTH_S * DNSMOS_SAMPLE_RATE)

        # Repeat the clip rather than zero-pad it. Padding a 5 s answer to 9 s
        # feeds the model 45 % silence, which it rates as poor quality — and does
        # so for every clip, which is how a real signal turns into a flat line.
        # The reference implementation tiles; so do we.
        while samples.size < window:
            samples = np.concatenate([samples, samples])

        # One-second hops, overlapping, as in the reference: more segments to
        # average over, and a bad passage cannot hide between two windows.
        hop = DNSMOS_SAMPLE_RATE
        scores = [self._infer(samples[start : start + window]) for start in range(0, samples.size - window + 1, hop)]
        averaged = [float(np.mean([s[i] for s in scores])) for i in range(len(SUBSCORES))]

        details: dict[str, Any] = dict(zip(SUBSCORES, averaged, strict=True))
        details["segments"] = len(scores)
        details["duration_s"] = round(audio.duration_s, 2)
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
    """Map the model's raw outputs onto the published MOS scale.

    Coefficients are the non-personalised polynomials of ``get_polyfit_val`` in
    microsoft/DNS-Challenge — second degree, applied as ``a*x^2 + b*x + c``.

    Getting this wrong is not a small offset: a mis-specified polynomial with an
    interior maximum saturates, and every clip above a certain quality collapses
    onto the same value. That is what produced a baseline where 24 answers all
    scored between 1.60 and 1.62.
    """
    sig = SIG_POLY(sig_raw)
    bak = BAK_POLY(bak_raw)
    ovrl = OVRL_POLY(ovrl_raw)
    return float(sig), float(bak), float(ovrl)
