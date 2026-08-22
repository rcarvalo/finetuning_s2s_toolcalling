import pytest

from lfm2_audio.vllm_plugin.constants import verify_placeholder_ids
from lfm2_audio.vllm_plugin.convert_checkpoint import ConversionError, build_omni_config

LIQUID_CONFIG = {
    "lfm": {"hidden_size": 2048, "vocab_size": 65536},
    "encoder": {"feat_in": 128},
    "depthformer": {"layers": 4, "dim": 1024, "tie": True},
    "preprocessor": {"sample_rate": 16000},
    "codebooks": 8,
    "interleaved_n_text": 6,
    "interleaved_n_audio": 10,
}


def test_build_omni_config():
    cfg = build_omni_config(LIQUID_CONFIG)
    assert cfg["architectures"] == ["Lfm2AudioOmniModel"]
    assert cfg["model_type"] == "lfm2_audio"
    assert cfg["interleaved_n_text"] == 6 and cfg["interleaved_n_audio"] == 10  # ratio calibré préservé
    assert cfg["audio_frame_token_id"] != cfg["audio_eoa_token_id"]
    assert cfg["lfm"]["hidden_size"] == 2048  # sections liquid intactes


def test_build_omni_config_defaults_ratio():
    config = {k: v for k, v in LIQUID_CONFIG.items() if not k.startswith("interleaved")}
    cfg = build_omni_config(config)
    assert cfg["interleaved_n_text"] == 6 and cfg["interleaved_n_audio"] == 12


def test_build_omni_config_rejects_partial_export():
    with pytest.raises(ConversionError, match="missing sections"):
        build_omni_config({"lfm": {}})  # export backbone-only, pas full


def test_build_omni_config_rejects_equal_placeholders():
    with pytest.raises(ConversionError):
        build_omni_config(LIQUID_CONFIG, audio_frame_token_id=128, audio_eoa_token_id=128)


class _FakeTokenizer:
    def convert_ids_to_tokens(self, tid):
        return {128: "<|audio_start|>", 129: "<|reserved_1|>", 1000: "bonjour"}.get(tid)


def test_verify_placeholder_ids_ok():
    assert verify_placeholder_ids(_FakeTokenizer(), 128, 129) == []


def test_verify_placeholder_ids_rejects_regular_token():
    problems = verify_placeholder_ids(_FakeTokenizer(), 1000, 129)
    assert any("regular token" in p for p in problems)


def test_verify_placeholder_ids_rejects_unknown_and_equal():
    problems = verify_placeholder_ids(_FakeTokenizer(), 555, 555)
    assert any("not in the tokenizer" in p for p in problems)
    assert any("must differ" in p for p in problems)
