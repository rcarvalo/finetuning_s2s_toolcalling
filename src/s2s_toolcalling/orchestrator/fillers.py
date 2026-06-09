"""Fillers vocaux : masquent le round-trip d'exécution des outils (cible < 1,5 s).

Le transport joue le filler (wav pré-rendu si disponible, sinon TTS/phrase)
dès qu'un tool call est détecté, pendant que l'orchestrateur exécute l'outil.
Les wavs se pré-rendent une fois avec la voix du modèle (mode TTS sequential)
ou une TTS FR, et se posent dans ``filler_dir`` nommés ``<tool>_<i>.wav``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PHRASES: dict[str, list[str]] = {
    "check_appointment": [
        "Je vérifie votre rendez-vous, un instant.",
        "Un instant, je regarde l'agenda.",
    ],
    "notify_employee": [
        "Je préviens votre contact, un petit instant.",
        "J'envoie un message tout de suite.",
    ],
    "guide_visitor": [
        "Je regarde le plan, un instant.",
    ],
    "get_guest_wifi": [
        "Je récupère les informations wifi.",
    ],
    "notify_receptionist": [
        "Je contacte l'accueil, un instant s'il vous plaît.",
    ],
    "query_database": [
        "Je consulte notre base, un petit instant.",
    ],
    "search_knowledge_base": [
        "Je recherche cette information, un instant.",
    ],
    "_default": [
        "Un instant, s'il vous plaît.",
        "Je regarde ça tout de suite.",
    ],
}


@dataclass(slots=True)
class Filler:
    phrase: str
    wav_path: str | None = None


@dataclass
class FillerBank:
    filler_dir: Path | None = None
    phrases: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_PHRASES))
    rng: random.Random = field(default_factory=random.Random)

    def get(self, tool_name: str) -> Filler:
        candidates = self.phrases.get(tool_name) or self.phrases["_default"]
        idx = self.rng.randrange(len(candidates))
        phrase = candidates[idx]

        wav_path: str | None = None
        if self.filler_dir is not None:
            for candidate in (f"{tool_name}_{idx}.wav", f"{tool_name}.wav", f"default_{idx}.wav", "default.wav"):
                p = self.filler_dir / candidate
                if p.exists():
                    wav_path = str(p)
                    break
        return Filler(phrase=phrase, wav_path=wav_path)
