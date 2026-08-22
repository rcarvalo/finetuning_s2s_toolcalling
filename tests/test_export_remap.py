import pytest

from lfm2_audio.core.errors import ExportError
from lfm2_audio.training.checkpoint_layout import (
    build_backbone_config,
    merged_full_mapping,
    remap_backbone_keys,
    strip_lora_keys,
    update_interleaved_ratio,
)

SAMPLE_KEYS = [
    "lfm.embed_tokens.weight",
    "lfm.layers.0.operator.in_proj.weight",
    "lfm.layers.10.self_attn.q_proj.weight",
    "lfm.layers.15.feed_forward.w1.weight",
    "lfm.embedding_norm.weight",
    "conformer.encoder.layers.0.linear.weight",
    "audio_adapter.0.weight",
    "audio_embedding.embedding.weight",
    "depthformer.layers.0.attn.weight",
    "depth_linear.weight",
    "depth_embeddings.0.embedding.weight",
]


def test_remap_only_backbone_keys():
    mapping = remap_backbone_keys(SAMPLE_KEYS)
    assert mapping["lfm.embed_tokens.weight"] == "model.embed_tokens.weight"
    assert mapping["lfm.layers.10.self_attn.q_proj.weight"] == "model.layers.10.self_attn.q_proj.weight"
    # aucun module hors backbone ne fuit dans le checkpoint texte
    assert all(dst.startswith("model.") for dst in mapping.values())
    assert not any("conformer" in src or "depth" in src or "audio" in src for src in mapping)


def test_remap_rejects_unmerged_lora():
    with pytest.raises(ExportError, match="unmerged LoRA"):
        remap_backbone_keys(["lfm.layers.0.self_attn.q_proj.lora_A.default.weight"])


def test_remap_requires_backbone():
    with pytest.raises(ExportError, match="no backbone"):
        remap_backbone_keys(["conformer.encoder.layers.0.linear.weight"])


def test_strip_lora_keys():
    keys = ["lfm.a.weight", "lfm.a.lora_A.default.weight", "lfm.a.lora_B.default.weight"]
    assert strip_lora_keys(keys) == ["lfm.a.weight"]


def test_merged_full_mapping_unwraps_base_layer():
    # state_dict réel d'un merge peft : modules wrappés en .base_layer.weight + lora_*.
    keys = [
        "lfm.layers.0.short_conv.in_proj.base_layer.weight",
        "lfm.layers.0.short_conv.in_proj.lora_A.default.weight",
        "lfm.layers.0.short_conv.in_proj.lora_B.default.weight",
        "lfm.embed_tokens.weight",  # module non wrappé → nom nu conservé
    ]
    mapping = merged_full_mapping(keys)
    # nom d'export nu → nom source wrappé (pour réindexer le state_dict)
    assert mapping["lfm.layers.0.short_conv.in_proj.weight"] == "lfm.layers.0.short_conv.in_proj.base_layer.weight"
    assert mapping["lfm.embed_tokens.weight"] == "lfm.embed_tokens.weight"
    assert not any("base_layer" in dst or "lora_" in dst for dst in mapping)


def test_remap_backbone_unwraps_base_layer():
    mapping = remap_backbone_keys(["lfm.layers.0.self_attn.q_proj.base_layer.weight"])
    assert mapping["lfm.layers.0.self_attn.q_proj.base_layer.weight"] == "model.layers.0.self_attn.q_proj.weight"


def test_build_backbone_config():
    liquid = {"lfm": {"hidden_size": 2048, "num_hidden_layers": 16, "vocab_size": 65536}, "codebooks": 8}
    cfg = build_backbone_config(liquid)
    assert cfg["architectures"] == ["Lfm2ForCausalLM"]
    assert cfg["model_type"] == "lfm2"
    assert cfg["tie_word_embeddings"] is True
    assert cfg["hidden_size"] == 2048
    assert "codebooks" not in cfg  # rien du chemin audio dans le config texte


def test_build_backbone_config_requires_lfm_section():
    with pytest.raises(ExportError):
        build_backbone_config({"encoder": {}})


def test_update_interleaved_ratio():
    cfg = update_interleaved_ratio({"interleaved_n_text": 6, "interleaved_n_audio": 12}, 6, 10)
    assert (cfg["interleaved_n_text"], cfg["interleaved_n_audio"]) == (6, 10)


def test_update_interleaved_ratio_noop_keeps_original():
    cfg = update_interleaved_ratio({"interleaved_n_text": 6, "interleaved_n_audio": 12}, None, None)
    assert (cfg["interleaved_n_text"], cfg["interleaved_n_audio"]) == (6, 12)


def test_update_interleaved_ratio_validates():
    with pytest.raises(ExportError):
        update_interleaved_ratio({}, 0, 12)
