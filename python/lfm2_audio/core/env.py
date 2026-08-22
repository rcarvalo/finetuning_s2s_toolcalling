"""Prérequis d'environnement du chemin d'inférence (imports lourds, CUDA).

Deux pièges connus, tous deux rencontrés sur Colab :

1. ``vllm-omni`` ne déclare **pas** ``vllm`` en dépendance (versions appariées
   major.minor) : l'import échoue sans message utile.
2. Le wheel PyPI de vLLM 0.22 est buildé en **CUDA 13** ; sur un hôte CUDA 12,
   ``libcudart.so.13`` est introuvable et l'import de vllm meurt.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import subprocess
import sys
from importlib.util import find_spec

import vllm_omni.plugins as omni_plugins
from vllm_omni.plugins import load_omni_general_plugins

import lfm2_audio.vllm_plugin  # noqa: F401 — enregistre l'entry point
from lfm2_audio.core.errors import BackendUnavailableError

logger = logging.getLogger(__name__)

_VLLM_CU129_WHEEL = (
    "pip install 'vllm @ https://github.com/vllm-project/vllm/releases/download/"
    "v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' "
    "--extra-index-url https://download.pytorch.org/whl/cu129"
)

_CUDART13_GLOB = "/usr/local/lib/python*/dist-packages/nvidia/**/libcudart.so.13"


def require_vllm() -> None:
    """Vérifie que ``vllm`` et ``vllm_omni`` sont importables.

    Lève ``BackendUnavailableError`` avec la commande d'installation exacte
    plutôt que de laisser remonter un ``ImportError`` opaque.
    """
    if find_spec("vllm") is None:
        raise BackendUnavailableError(
            "vllm n'est pas installé — vllm-omni ne le déclare pas en dépendance "
            "(versions appariées major.minor). Sur CUDA 12 (Colab), le build PyPI "
            f"(CUDA 13) ne marche pas, utiliser le wheel +cu129 du release :\n    {_VLLM_CU129_WHEEL}"
        )
    if find_spec("vllm_omni") is None:
        raise BackendUnavailableError("vllm-omni n'est pas installé — `uv sync --extra serving` (GPU requis).")


def require_liquid_audio() -> None:
    """Vérifie que ``liquid_audio`` est importable (backend de référence)."""
    if find_spec("liquid_audio") is None:
        raise BackendUnavailableError("liquid-audio n'est pas installé — `uv sync --extra serving` (GPU requis).")


def preload_cuda13(*, install_if_missing: bool = True) -> bool:
    """Précharge ``libcudart.so.13`` & co. pour un vLLM cu13 sur un hôte cu12.

    ``RTLD_GLOBAL`` pour CE process (le loader a déjà figé ``LD_LIBRARY_PATH`` au
    démarrage) **et** export de ``LD_LIBRARY_PATH`` pour les stage workers, qui
    sont des sous-process et le lisent à leur démarrage.

    No-op hors Linux ou si la lib est déjà résolvable. Retourne ``True`` si des
    bibliothèques ont été préchargées.
    """
    if not sys.platform.startswith("linux"):
        return False

    libs = glob.glob(_CUDART13_GLOB, recursive=True)
    if not libs and install_if_missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "nvidia-cuda-runtime-cu13"],
            check=False,
        )
        libs = glob.glob(_CUDART13_GLOB, recursive=True)
    if not libs:
        logger.warning("libcudart.so.13 introuvable — l'import de vllm risque d'échouer")
        return False

    lib_dirs = sorted({os.path.dirname(path) for path in libs})
    os.environ["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, os.environ.get("LD_LIBRARY_PATH", "")])
    for directory in lib_dirs:
        for shared_object in sorted(glob.glob(directory + "/lib*.so*")):
            try:
                ctypes.CDLL(shared_object, mode=ctypes.RTLD_GLOBAL)
            except OSError:  # lib incompatible ou déjà chargée : sans conséquence
                logger.debug("préchargement ignoré : %s", shared_object)
    logger.info("CUDA 13 préchargé depuis %s", lib_dirs)
    return True


def load_vllm_omni_plugins() -> None:
    """Force le chargement des plugins vLLM-Omni AVANT ``Omni(...)``.

    La détection de pipeline d'``Omni()`` précède le chargement des plugins ;
    sans cet appel explicite, notre architecture n'est pas encore enregistrée.
    Le flag ``omni_plugins_loaded`` est remis à ``False`` car un premier essai
    ayant échoué (import manquant) laisse le module en état « déjà chargé ».
    """
    require_vllm()

    omni_plugins.omni_plugins_loaded = False

    load_omni_general_plugins()
