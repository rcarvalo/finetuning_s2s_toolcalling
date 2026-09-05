"""Le préchargement CUDA 13 : chemin pour l'enfant, jamais NVBLAS dans le parent.

Deux vagues de sonorisation sont mortes en -11 le 05/09, juste après
« [NVBLAS] CPU Blas library need to be provided » : libnvblas, chargée en
RTLD_GLOBAL avec le reste du dossier cu13, interceptait les GEMM de Whisper.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "voxtral_tts_synth.py"


@pytest.fixture
def vox() -> Any:
    spec = importlib.util.spec_from_file_location("voxtral_tts_synth", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_cuda_dir(tmp_path: Path) -> str:
    lib = tmp_path / "nvidia" / "cu13" / "lib"
    lib.mkdir(parents=True)
    for name in ("libcudart.so.13", "libcublas.so.13", "libnvblas.so.13", "libcublasLt.so.13"):
        (lib / name).write_bytes(b"")
    return str(lib)


def test_should_never_list_nvblas_among_the_libraries_to_load(vox: Any, tmp_path: Path) -> None:
    names = [Path(p).name for p in vox.loadable_cuda_libraries(_fake_cuda_dir(tmp_path))]

    assert "libnvblas.so.13" not in names
    assert {"libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"} <= set(names)


def test_should_only_export_the_path_for_the_server_route(
    vox: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _fake_cuda_dir(tmp_path)
    monkeypatch.setattr(vox, "cuda_library_dir", lambda: directory)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    loaded: list[str] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    vox.preload_cuda13()

    assert loaded == []
    assert vox.os.environ["LD_LIBRARY_PATH"].startswith(directory + ":")


def test_should_load_everything_but_nvblas_for_the_in_process_route(
    vox: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _fake_cuda_dir(tmp_path)
    monkeypatch.setattr(vox, "cuda_library_dir", lambda: directory)
    loaded: list[str] = []
    monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=0: loaded.append(Path(path).name))

    vox.preload_cuda13(in_process=True)

    assert "libnvblas.so.13" not in loaded
    assert "libcudart.so.13" in loaded
