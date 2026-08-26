"""``SampleArchive`` — garder ce qu'une campagne a produit, pour le renoter plus tard.

Une campagne coûte un GPU ; une métrique corrigée ne devrait pas coûter un
second GPU. C'est exactement ce qui s'est produit le 24/08/2026 : la calibration
DNSMOS s'est révélée fausse, les chiffres de la baseline EN sont devenus caducs,
et l'audio généré n'existait plus nulle part — il a fallu déclarer la métrique
« à recalculer » sans pouvoir le faire.

L'archive écrit, par échantillon, un WAV (l'audio généré) et un JSON (textes,
tool calls attendus, métadonnées). ``EvaluationPipeline.generate`` peut donc
être rejouée hors ligne : ``lfm2-eval-rescore`` relit l'archive et applique les
scorers d'aujourd'hui aux réponses d'hier.

L'audio du PROMPT n'est pas archivé : il vient du jeu de questions, qui est
versionné, et le stocker doublerait la taille de l'archive pour rien.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.scorer.sample import EvalSample

logger = logging.getLogger(__name__)

METADATA_SUFFIX = ".json"
AUDIO_SUFFIX = ".wav"


class SampleArchive:
    """Répertoire d'échantillons d'évaluation, écrit puis relu tel quel."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def save(self, sample: EvalSample) -> Path:
        """Écrit un échantillon et retourne le chemin de son JSON."""
        self._root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "prompt_text": sample.prompt_text,
            "predicted_text": sample.predicted_text,
            "reference_text": sample.reference_text,
            "expected_calls": sample.expected_calls,
            "tool_results": sample.tool_results,
            "metadata": sample.metadata,
        }
        if sample.has_predicted_audio and sample.predicted_audio is not None:
            audio_path = self._root / f"{sample.sample_id}{AUDIO_SUFFIX}"
            sample.predicted_audio.save(audio_path)
            payload["predicted_audio"] = audio_path.name
        path = self._root / f"{sample.sample_id}{METADATA_SUFFIX}"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def load(self) -> Iterator[EvalSample]:
        """Relit les échantillons, triés par identifiant (ordre stable des rapports)."""
        for path in sorted(self._root.glob(f"*{METADATA_SUFFIX}")):
            payload = json.loads(path.read_text())
            audio_name = payload.get("predicted_audio")
            audio = Waveform.from_file(self._root / audio_name) if audio_name else None
            yield EvalSample(
                sample_id=payload["sample_id"],
                prompt_text=payload.get("prompt_text", ""),
                predicted_text=payload.get("predicted_text", ""),
                predicted_audio=audio,
                reference_text=payload.get("reference_text", ""),
                expected_calls=payload.get("expected_calls", []),
                tool_results=payload.get("tool_results", []),
                metadata=payload.get("metadata", {}),
            )

    def __len__(self) -> int:
        return len(list(self._root.glob(f"*{METADATA_SUFFIX}"))) if self._root.exists() else 0
