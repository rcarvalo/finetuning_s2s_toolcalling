"""Lfm2AudioConfig : matérialisation Lfm2Config, round-trip, enregistrement."""

import pytest

pytest.importorskip("transformers")

from lfm2_audio.vllm_plugin.configuration import Lfm2AudioConfig, register_config

LFM_SECTION = {
    "architectures": ["Lfm2ForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "vocab_size": 65536,
}


def _make_config() -> Lfm2AudioConfig:
    return Lfm2AudioConfig(
        lfm=dict(LFM_SECTION),
        encoder={"d_model": 512},
        depthformer={"layers": 6, "dim": 1024, "tie": True},
        preprocessor={"sample_rate": 16000},
        interleaved_n_text=6,
        interleaved_n_audio=12,
    )


def test_should_materialize_lfm_section_as_lfm2_config():
    from transformers import Lfm2Config

    config = _make_config()
    assert isinstance(config.lfm, Lfm2Config)
    assert config.lfm.hidden_size == 2048
    assert config.lfm.vocab_size == 65536


def test_should_expose_backbone_via_get_text_config():
    config = _make_config()
    assert config.get_text_config() is config.lfm
    assert config.get_text_config(decoder=True) is config.lfm


def test_should_round_trip_through_to_dict():
    config = _make_config()
    data = config.to_dict()
    assert data["model_type"] == "lfm2_audio"
    assert isinstance(data["lfm"], dict)
    rebuilt = Lfm2AudioConfig(**{k: v for k, v in data.items() if k != "model_type"})
    assert rebuilt.lfm.hidden_size == config.lfm.hidden_size
    assert rebuilt.interleaved_n_audio == 12


def test_should_default_missing_sections_to_empty():
    config = Lfm2AudioConfig()
    assert config.encoder == {}
    assert config.interleaved_n_text == 6
    assert config.audio_frame_token_id != config.audio_eoa_token_id


def test_should_register_idempotently_with_autoconfig():
    register_config()
    register_config()  # second appel : ne lève pas
    from transformers import AutoConfig

    # AutoConfig résout bien le model_type vers notre classe
    cfg = AutoConfig.for_model("lfm2_audio")
    assert isinstance(cfg, Lfm2AudioConfig)
