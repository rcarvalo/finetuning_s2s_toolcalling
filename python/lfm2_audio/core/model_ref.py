"""``model_ref`` — donner à liquid-audio le TYPE qu'il attend.

``liquid_audio.get_model_dir`` surcharge son argument : une ``str`` est un
repo Hugging Face, téléchargé via ``snapshot_download`` ; un ``Path`` est un
dossier local, lu tel quel. Un chemin local passé en ``str`` part donc chercher
un repo nommé ``/root/.cache/…`` et lève ``HFValidationError``.

Ce piège a coûté deux fois : d'abord au backend liquid (cf. CHANGELOG), puis
au démarrage de la démo tool-calling, où la fusion d'adaptateur stringifiait le
snapshot déjà résolu. La règle vit donc ici, dans un module sans dépendance
lourde — importable et testable sans GPU ni liquid-audio.
"""

from __future__ import annotations

from pathlib import Path


def model_ref(base: str | Path) -> str | Path:
    """Chemin local en ``Path``, identifiant de repo en ``str``."""
    path = Path(base)
    return path if path.is_dir() else str(base)
