"""Tests de la fabrique de backends."""

from __future__ import annotations

import pytest

from lfm2_audio.core.errors import BackendUnavailableError
from lfm2_audio.serving.registry import BACKENDS, BackendRegistry, BackendSpec


def _spec(name: str, *, requires: tuple[str, ...]) -> BackendSpec:
    return BackendSpec(
        name=name,
        module="lfm2_audio.serving.registry",
        class_name="BackendRegistry",  # cible importable quelconque
        requires=requires,
    )


def test_should_expose_the_registered_names():
    registry = BackendRegistry((_spec("a", requires=()), _spec("b", requires=())))

    assert registry.names == ("a", "b")


def test_availability_should_follow_installed_modules():
    installed = _spec("installed", requires=("json",))
    missing = _spec("missing", requires=("definitely_not_a_module",))
    registry = BackendRegistry((installed, missing))

    assert registry.available() == ("installed",)


def test_should_reject_an_unknown_backend():
    registry = BackendRegistry((_spec("a", requires=()),))

    with pytest.raises(BackendUnavailableError, match="backend inconnu"):
        registry.get("nope")


def test_should_name_the_missing_dependency():
    registry = BackendRegistry((_spec("a", requires=("definitely_not_a_module",)),))

    with pytest.raises(BackendUnavailableError, match="definitely_not_a_module"):
        registry.get("a")


def test_auto_should_pick_the_first_available_in_order():
    registry = BackendRegistry(
        (
            _spec("preferred", requires=("definitely_not_a_module",)),
            _spec("fallback", requires=("json",)),
        )
    )

    assert registry.get("auto").name == "fallback"


def test_auto_should_report_when_nothing_is_installed():
    registry = BackendRegistry((_spec("a", requires=("definitely_not_a_module",)),))

    with pytest.raises(BackendUnavailableError, match="aucun backend"):
        registry.get("auto")


def test_describe_should_work_for_an_uninstalled_backend():
    """L'introspection ne doit pas exiger que le backend soit installé."""
    registry = BackendRegistry((_spec("a", requires=("definitely_not_a_module",)),))

    assert registry.describe("a").name == "a"


def test_should_load_the_class_lazily():
    registry = BackendRegistry((_spec("a", requires=()),))

    assert registry.load("a") is BackendRegistry


def test_register_should_add_a_backend():
    registry = BackendRegistry()

    registry.register(_spec("late", requires=()))

    assert "late" in registry.names


def test_default_registry_should_prefer_vllm_over_liquid():
    # L'ordre fixe la préférence du mode "auto" : basse latence d'abord.
    assert BACKENDS.names == ("vllm", "liquid")


def test_default_registry_should_declare_heavy_requirements():
    """Aucun backend ne doit être « disponible » sans ses dépendances GPU."""
    assert set(BACKENDS.describe("vllm").requires) >= {"vllm", "vllm_omni"}
    assert "liquid_audio" in BACKENDS.describe("liquid").requires
