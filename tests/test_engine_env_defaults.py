"""Contournements vLLM 0.22 posés au niveau PROCESSUS.

Ils ne vivaient que dans ``Dockerfile.serve`` : toute autre voie d'entrée —
démo locale, scénarios, notebook — démarrait sans eux et l'engine mourait sur
« StageEngineCoreProc died during READY ». Les tests portent sur le module de
constantes, sans importer vLLM (absent d'un poste sans GPU).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "python/lfm2_audio/serving/backends/omni_engine.py"


def _load_defaults() -> tuple[dict[str, str], object]:
    """Charge les constantes sans exécuter les imports vLLM du module."""
    source = MODULE.read_text(encoding="utf-8")
    start = source.index("ENGINE_ENV_DEFAULTS")
    end = source.index("def apply_engine_env_defaults")
    namespace: dict[str, object] = {}
    exec(compile(source[start:end], str(MODULE), "exec"), namespace)
    return namespace["ENGINE_ENV_DEFAULTS"], namespace


def test_should_disable_aot_compile() -> None:
    # vLLM 0.22.1 l'active par défaut, mais le torch qu'il épingle (2.11) n'a
    # pas l'attribut requis : l'étage compilé meurt au démarrage.
    defaults, _ = _load_defaults()

    assert defaults["VLLM_USE_AOT_COMPILE"] == "0"


def test_should_disable_the_flashinfer_sampler() -> None:
    # Son JIT réclame nvcc, absent des images sans toolkit CUDA.
    defaults, _ = _load_defaults()

    assert defaults["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_should_cover_the_variables_baked_into_the_serve_image() -> None:
    # Le Dockerfile et le code ne doivent pas diverger : c'est cette divergence
    # qui a fait mourir la démo alors que l'image serverless, elle, démarrait.
    dockerfile = (MODULE.parents[4] / "infra/Dockerfile.serve").read_text(encoding="utf-8")
    defaults, _ = _load_defaults()

    for name in defaults:
        assert name in dockerfile, f"{name} absent de Dockerfile.serve"


@pytest.mark.parametrize("name", ["VLLM_USE_AOT_COMPILE", "VLLM_USE_FLASHINFER_SAMPLER"])
def test_should_not_override_an_explicit_choice(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Un réglage explicite de l'opérateur prime sur le défaut.
    monkeypatch.setenv(name, "1")
    defaults, _ = _load_defaults()

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    assert os.environ[name] == "1"
