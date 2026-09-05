"""Every historical import path still resolves to the toolkit's implementation."""

from __future__ import annotations

import importlib

import pytest

SHIMS = {
    "lfm2_audio.ds.audio": ("Waveform", "INPUT_SAMPLE_RATE", "OUTPUT_SAMPLE_RATE"),
    "lfm2_audio.ds.dialogue": ("Dialogue", "Turn", "ToolCall", "DialogueMeta", "load_dialogues", "parse_dialogue"),
    "lfm2_audio.core.lazy_component": ("LazyComponent",),
    "lfm2_audio.scorer": ("BaseScorer", "EvalSample", "ScoreResult", "ScorerFactory", "SCORERS", "MetricSummary"),
    "lfm2_audio.scorer.audio.wer": ("WerScorer", "word_error_rate", "normalize_transcript"),
    "lfm2_audio.scorer.audio.dnsmos": ("DnsmosScorer", "calibrate_p835"),
    "lfm2_audio.scorer.text.lang_match": ("LangMatchScorer", "detect_language"),
    "lfm2_audio.scorer.text.rubric": ("ANSWER_RUBRIC_V3", "resolve_rubric"),
    "lfm2_audio.scorer.text.reasoning": ("ReasoningScorer", "spoken_part"),
    "lfm2_audio.scorer.text.tool_call": ("ToolCallScorer",),
    "lfm2_audio.evaluation.argument_match": ("diff_arguments", "token_f1", "ArgumentMismatch"),
    "lfm2_audio.evaluation.tool_call_diagnosis": ("ToolCallDiagnosis", "OUTCOMES", "diagnose"),
    "lfm2_audio.evaluation.versa_runner": ("VersaRunner", "MOS_CONFIG", "nisqa_config", "DEFAULT_VERSA_ROOT"),
    "lfm2_audio.evaluation.eval_log_audio": ("LoggedReply", "extract_replies", "latest_log"),
    "lfm2_audio.inspect_bridge.audio": ("waveform_to_data_uri", "data_uri_to_waveform"),
    "lfm2_audio.inspect_bridge.scorers": ("to_eval_sample", "wrap", "lfm2_scorer"),
}


@pytest.mark.parametrize(("module", "names"), sorted(SHIMS.items()))
def test_shim_should_expose_its_historical_names(module: str, names: tuple[str, ...]) -> None:
    loaded = importlib.import_module(module)

    for name in names:
        assert hasattr(loaded, name), f"{module} lost {name}"


def test_scoring_config_should_default_to_the_lfm2_components() -> None:
    from lfm2_audio.ds.scoring_config import ScoringConfig

    config = ScoringConfig.with_defaults()

    assert config.text_cleaner == "lfm2"
    assert config.tool_call_parser == "lfm2"
    assert set(config.enabled_names) == {"wer", "dnsmos", "utmos", "tool_call", "reasoning"}


def test_versa_default_root_should_be_the_sibling_checkout() -> None:
    from lfm2_audio.evaluation.versa_runner import DEFAULT_VERSA_ROOT, VersaRunner

    assert DEFAULT_VERSA_ROOT.name == "versa-eval"
    assert VersaRunner().root == DEFAULT_VERSA_ROOT
