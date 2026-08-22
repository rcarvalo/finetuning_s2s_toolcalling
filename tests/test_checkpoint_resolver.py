"""Tests de ``CheckpointResolver`` et de la chaîne de sources.

Toutes les dépendances sont injectées (sources, stratégies) : aucun accès réseau,
aucun GPU, aucune écriture hors ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from lfm2_audio.core.errors import CheckpointError
from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout
from lfm2_audio.serving.checkpoint.preparers import READY_MARKER, CheckpointPreparer
from lfm2_audio.serving.checkpoint.resolver import CheckpointResolver
from lfm2_audio.serving.checkpoint.sources import (
    CheckpointSource,
    HuggingFaceSource,
    LocalPathSource,
    SourceChain,
)

LIQUID_CONFIG = {"lfm": {}, "encoder": {}, "depthformer": {}, "preprocessor": {}}
OMNI_CONFIG = {"architectures": ["Lfm2AudioOmniModel"], **LIQUID_CONFIG}


def _checkpoint(tmp_path, name: str, config: dict):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _adapter(tmp_path, name: str, base: str):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}), encoding="utf-8")
    return directory


class RecordingPreparer(CheckpointPreparer):
    """Stratégie factice : note l'appel et crée un répertoire marqué prêt."""

    def __init__(self, layout: Layout, *, wants_adapter: bool) -> None:
        self._layout = layout
        self._wants_adapter = wants_adapter
        self.calls: list[tuple] = []

    def handles(self, layout: Layout, *, has_adapter: bool) -> bool:
        return layout is self._layout and has_adapter is self._wants_adapter

    def prepare(self, source, target, request, adapter):
        self.calls.append((source, target, adapter))
        target.mkdir(parents=True, exist_ok=True)
        self.mark_ready(target)
        return target


# --------------------------------------------------------------------------- #
# Chaîne de sources
# --------------------------------------------------------------------------- #


def test_local_source_should_accept_an_existing_directory(tmp_path):
    directory = _checkpoint(tmp_path, "ckpt", OMNI_CONFIG)

    assert LocalPathSource().accepts(directory)
    assert LocalPathSource().materialize(directory) == directory.resolve()


def test_local_source_should_reject_a_file(tmp_path):
    plain_file = tmp_path / "weights.bin"
    plain_file.write_text("x", encoding="utf-8")

    with pytest.raises(CheckpointError, match="pas un répertoire"):
        LocalPathSource().materialize(plain_file)


@pytest.mark.parametrize("spec", ["org/name", "Rcarvalo/lfm25-tc-en"])
def test_huggingface_source_should_accept_repo_ids(spec):
    assert HuggingFaceSource().accepts(spec)


@pytest.mark.parametrize("spec", ["./relative", "/absolute/path", "~/home", "bare"])
def test_huggingface_source_should_reject_paths(spec):
    assert not HuggingFaceSource().accepts(spec)


def test_chain_should_prefer_a_local_directory_named_like_a_repo(tmp_path, monkeypatch):
    # Un répertoire `org/nom` sur le disque doit gagner sur le Hub.
    directory = _checkpoint(tmp_path / "org", "name", OMNI_CONFIG)
    monkeypatch.chdir(tmp_path)

    assert SourceChain().materialize("org/name") == directory.resolve()


def test_chain_should_reject_something_that_is_neither(tmp_path):
    with pytest.raises(CheckpointError, match="Hugging Face"):
        SourceChain().materialize(str(tmp_path / "missing"))


# --------------------------------------------------------------------------- #
# Résolution
# --------------------------------------------------------------------------- #


def test_should_pass_an_omni_checkpoint_through_untouched(tmp_path):
    directory = _checkpoint(tmp_path, "omni", OMNI_CONFIG)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache")

    resolved = resolver.resolve(CheckpointRequest(spec=directory, backend="vllm"))

    assert resolved.path == directory.resolve()
    assert resolved.layout is Layout.OMNI


def test_should_convert_a_liquid_checkpoint(tmp_path):
    directory = _checkpoint(tmp_path, "liquid", LIQUID_CONFIG)
    preparer = RecordingPreparer(Layout.LIQUID, wants_adapter=False)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(preparer,))

    resolved = resolver.resolve(CheckpointRequest(spec=directory, backend="vllm"))

    assert len(preparer.calls) == 1
    assert resolved.layout is Layout.OMNI
    assert resolved.path.is_relative_to(tmp_path / "cache")


def test_should_reuse_a_ready_cache_entry_without_preparing_again(tmp_path):
    directory = _checkpoint(tmp_path, "liquid", LIQUID_CONFIG)
    preparer = RecordingPreparer(Layout.LIQUID, wants_adapter=False)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(preparer,))
    request = CheckpointRequest(spec=directory, backend="vllm")

    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first.path == second.path
    assert len(preparer.calls) == 1  # la 2e résolution a lu le cache


def test_should_ignore_a_cache_entry_without_its_ready_marker(tmp_path):
    """Un run interrompu laisse un répertoire incomplet : il doit être refait."""
    directory = _checkpoint(tmp_path, "liquid", LIQUID_CONFIG)
    preparer = RecordingPreparer(Layout.LIQUID, wants_adapter=False)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(preparer,))
    request = CheckpointRequest(spec=directory, backend="vllm")

    target = resolver.resolve(request).path
    (target / READY_MARKER).unlink()
    resolver.resolve(request)

    assert len(preparer.calls) == 2


def test_ratio_should_change_the_cache_entry(tmp_path):
    """Deux ratios interleaved produisent deux checkpoints distincts."""
    directory = _checkpoint(tmp_path, "liquid", LIQUID_CONFIG)
    preparer = RecordingPreparer(Layout.LIQUID, wants_adapter=False)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(preparer,))

    first = resolver.resolve(CheckpointRequest(spec=directory, backend="vllm", interleaved_ratio=(6, 10)))
    second = resolver.resolve(CheckpointRequest(spec=directory, backend="vllm", interleaved_ratio=(6, 12)))

    assert first.path != second.path


def test_should_route_base_plus_adapter_to_the_merge_strategy(tmp_path):
    base = _checkpoint(tmp_path, "base", LIQUID_CONFIG)
    adapter = _adapter(tmp_path, "adapter", base=str(base))
    merger = RecordingPreparer(Layout.LIQUID, wants_adapter=True)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(merger,))

    resolver.resolve(CheckpointRequest(spec=base, backend="vllm", adapter=adapter))

    assert len(merger.calls) == 1
    assert merger.calls[0][2] == adapter.resolve()


def test_an_adapter_alone_should_resolve_its_declared_base(tmp_path):
    base = _checkpoint(tmp_path, "base", LIQUID_CONFIG)
    adapter = _adapter(tmp_path, "adapter", base=str(base))
    merger = RecordingPreparer(Layout.LIQUID, wants_adapter=True)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=(merger,))

    resolver.resolve(CheckpointRequest(spec=adapter, backend="vllm"))

    source, _, resolved_adapter = merger.calls[0]
    assert source == base.resolve()
    assert resolved_adapter == adapter.resolve()


def test_should_reject_an_adapter_spec_combined_with_another_adapter(tmp_path):
    base = _checkpoint(tmp_path, "base", LIQUID_CONFIG)
    adapter = _adapter(tmp_path, "adapter", base=str(base))
    other = _adapter(tmp_path, "other", base=str(base))
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache")

    with pytest.raises(CheckpointError, match="ne peut pas être fourni en plus"):
        resolver.resolve(CheckpointRequest(spec=adapter, backend="vllm", adapter=other))


def test_should_reject_a_text_only_backbone(tmp_path):
    directory = _checkpoint(tmp_path, "backbone", {"architectures": ["Lfm2ForCausalLM"]})
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache")

    with pytest.raises(CheckpointError, match="backbone texte seul"):
        resolver.resolve(CheckpointRequest(spec=directory, backend="vllm"))


def test_liquid_backend_should_get_the_native_layout_and_its_adapter(tmp_path):
    """liquid-audio fusionne le LoRA en mémoire : aucun export disque attendu."""
    base = _checkpoint(tmp_path, "base", LIQUID_CONFIG)
    adapter = _adapter(tmp_path, "adapter", base=str(base))
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache")

    resolved = resolver.resolve(CheckpointRequest(spec=base, backend="liquid", adapter=adapter))

    assert resolved.path == base.resolve()
    assert resolved.layout is Layout.LIQUID
    assert resolved.adapter == adapter.resolve()
    assert not (tmp_path / "cache").exists()


def test_should_report_when_no_strategy_applies(tmp_path):
    directory = _checkpoint(tmp_path, "liquid", LIQUID_CONFIG)
    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", preparers=())

    with pytest.raises(CheckpointError, match="aucune stratégie"):
        resolver.resolve(CheckpointRequest(spec=directory, backend="vllm"))


def test_should_use_an_injected_source_chain(tmp_path):
    """La résolution ne doit jamais toucher le réseau si une source la couvre."""
    directory = _checkpoint(tmp_path, "omni", OMNI_CONFIG)

    class FixedSource(CheckpointSource):
        def accepts(self, spec) -> bool:
            return True

        def materialize(self, spec):
            return directory

    resolver = CheckpointResolver(cache_dir=tmp_path / "cache", sources=SourceChain([FixedSource()]))

    resolved = resolver.resolve(CheckpointRequest(spec="anything/at-all", backend="vllm"))

    assert resolved.path == directory
