"""``EvalSample`` — l'unité que tout scorer sait lire."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lfm2_audio.core.prompt import strip_special_tokens
from lfm2_audio.ds.audio import Waveform


@dataclass(frozen=True, slots=True)
class EvalSample:
    """Une question posée au modèle et ce qu'il a produit.

    Un seul type d'entrée pour tous les scorers : chacun lit les champs qui le
    concernent et se déclare ``SKIPPED`` sur les autres. C'est ce qui permet de
    faire tourner le même jeu de scorers depuis la pipeline d'éval et depuis une
    boucle d'entraînement, sans adaptateur.
    """

    sample_id: str

    prompt_text: str = ""
    """Question posée, en texte (référence de transcription du prompt audio)."""

    prompt_audio: Waveform | None = None
    """Audio d'entrée effectivement envoyé au modèle."""

    predicted_text: str = ""
    """Texte BRUT généré — marqueurs ``<|tool_call_*|>`` compris."""

    predicted_audio: Waveform | None = None
    """Audio généré par le modèle, à noter en qualité."""

    reference_text: str = ""
    """Réponse attendue, quand il y en a une."""

    expected_calls: list[dict[str, Any]] = field(default_factory=list)
    """Tool calls attendus : ``[{"name": ..., "arguments": {...}}]``."""

    tool_results: list[dict[str, Any]] = field(default_factory=list)
    """Résultats d'outils réinjectés — base de l'ancrage factuel."""

    trajectory: list[dict[str, Any]] = field(default_factory=list)
    """Les étapes du tour (appel, résultat d'outil, réponse), sérialisées.

    Portée en dict plutôt qu'en :class:`~lfm2_audio.evaluation.trajectory.Trajectory`
    pour que ``EvalSample`` reste ce que tout scorer sait lire, sans dépendre du
    paquet d'évaluation. ``Trajectory.from_list`` la relit quand on en a besoin.
    """

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_predicted_audio(self) -> bool:
        return self.predicted_audio is not None and not self.predicted_audio.is_empty

    @property
    def expects_tool_call(self) -> bool:
        return bool(self.expected_calls)

    @property
    def spoken_reference(self) -> str:
        """Texte que l'audio généré est censé prononcer.

        La référence explicite prime ; à défaut, le texte généré lui-même — un
        WER calculé là-dessus mesure la fidélité du TTS à sa propre sortie
        texte, ce qui est exactement ce qu'on veut en S2S interleaved.
        """
        return self.reference_text or strip_special_tokens(self.predicted_text)
