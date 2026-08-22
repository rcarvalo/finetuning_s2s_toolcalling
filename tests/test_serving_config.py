"""Tests des configs pydantic du serving.

Les défauts testés ici ne sont pas des préférences : ce sont des contournements
mesurés sur vLLM-Omni 0.22. Un changement silencieux casserait le flux audio
sans faire échouer quoi que ce soit d'autre — d'où ces assertions explicites.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.inference_config import EngineConfig


@pytest.fixture
def deploy_yaml(tmp_path):
    path = tmp_path / "deploy.yaml"
    path.write_text("async_chunk: true\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# EngineConfig
# --------------------------------------------------------------------------- #


def test_prefix_caching_and_async_scheduling_must_stay_off_by_default():
    config = EngineConfig(deploy_config=None)

    assert config.enable_prefix_caching is False
    assert config.async_scheduling is False


def test_should_be_immutable():
    config = EngineConfig(deploy_config=None)

    with pytest.raises(ValidationError):
        config.dtype = "float16"


def test_should_reject_unknown_fields():
    with pytest.raises(ValidationError):
        EngineConfig(deploy_config=None, gpu_memory_utilisation=0.5)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_should_reject_out_of_range_gpu_utilisation(value):
    with pytest.raises(ValidationError):
        EngineConfig(deploy_config=None, gpu_memory_utilization=value)


def test_should_reject_a_missing_deploy_config(tmp_path):
    with pytest.raises(ValidationError, match="introuvable"):
        EngineConfig(deploy_config=tmp_path / "absent.yaml")


def test_deploy_config_should_drive_the_engine_kwargs(deploy_yaml):
    kwargs = EngineConfig(deploy_config=deploy_yaml).to_omni_kwargs("/ckpt")

    assert kwargs["deploy_config"] == str(deploy_yaml)
    # Les réglages par stage viennent du YAML : ne pas les écraser globalement.
    assert "enforce_eager" not in kwargs
    assert "gpu_memory_utilization" not in kwargs


def test_legacy_path_should_spell_out_every_workaround():
    kwargs = EngineConfig(deploy_config=None).to_omni_kwargs("/ckpt")

    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["async_scheduling"] is False
    assert kwargs["dtype"] == "bfloat16"


def test_engine_kwargs_should_always_carry_model_and_timeouts():
    kwargs = EngineConfig(deploy_config=None).to_omni_kwargs("/ckpt")

    assert kwargs["model"] == "/ckpt"
    assert kwargs["async_chunk"] is True
    assert kwargs["stage_init_timeout"] > 0
    assert kwargs["init_timeout"] > 0


def test_should_load_from_yaml(tmp_path):
    path = tmp_path / "engine.yaml"
    path.write_text("dtype: float16\ndeploy_config: null\n", encoding="utf-8")

    assert EngineConfig.from_yaml(path).dtype == "float16"


def test_should_load_from_an_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert EngineConfig.from_yaml(path).dtype == "bfloat16"


# --------------------------------------------------------------------------- #
# GenerationConfig
# --------------------------------------------------------------------------- #


def test_text_should_be_greedy_by_default():
    # Les tool calls doivent être déterministes.
    assert GenerationConfig().temperature == 0.0


def test_audio_sampling_defaults_should_match_the_reference():
    config = GenerationConfig()

    assert config.audio_temperature == 1.0
    assert config.audio_top_k == 4


def test_text_only_should_be_off_by_default():
    assert GenerationConfig().text_only is False


def test_with_max_tokens_should_return_a_new_config():
    original = GenerationConfig(max_tokens=400)

    updated = original.with_max_tokens(128)

    assert updated.max_tokens == 128
    assert original.max_tokens == 400


def test_with_max_tokens_should_be_a_noop_for_none():
    original = GenerationConfig(max_tokens=400)

    assert original.with_max_tokens(None) is original


@pytest.mark.parametrize("value", [0, -1])
def test_should_reject_non_positive_max_tokens(value):
    with pytest.raises(ValidationError):
        GenerationConfig(max_tokens=value)
