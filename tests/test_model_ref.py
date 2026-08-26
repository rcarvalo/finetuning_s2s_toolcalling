"""``core.model_ref`` — le TYPE attendu par liquid-audio.

``get_model_dir`` surcharge son argument : ``str`` = repo Hub (téléchargé via
``snapshot_download``), ``Path`` = dossier local lu tel quel. Un chemin local
passé en ``str`` part donc chercher un repo nommé ``/root/.cache/...``.

Ce piège a coûté deux fois : au backend liquid (cf. CHANGELOG), puis au
démarrage de la démo tool-calling, où la fusion d'adaptateur stringifiait le
snapshot déjà résolu — d'où une règle unique, testable sans GPU.
"""

from __future__ import annotations

from pathlib import Path

from lfm2_audio.core.model_ref import model_ref


def test_should_keep_a_hub_repo_id_as_a_string() -> None:
    ref = model_ref("LiquidAI/LFM2.5-Audio-1.5B")

    assert isinstance(ref, str)
    assert ref == "LiquidAI/LFM2.5-Audio-1.5B"


def test_should_hand_an_existing_directory_as_a_path(tmp_path: Path) -> None:
    ref = model_ref(tmp_path)

    assert isinstance(ref, Path)
    assert ref == tmp_path


def test_should_convert_a_local_directory_given_as_a_string(tmp_path: Path) -> None:
    # Le cas qui a tué la démo : un snapshot résolu, stringifié par l'appelant.
    assert isinstance(model_ref(str(tmp_path)), Path)


def test_should_treat_a_missing_path_as_a_repo_id(tmp_path: Path) -> None:
    # Un dossier absent n'est pas un chemin local : on laisse le Hub rendre son
    # erreur plutôt que d'inventer un dossier qui n'existe pas.
    assert isinstance(model_ref(str(tmp_path / "nope")), str)
