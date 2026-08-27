"""Tests for the Inspect bridge: dataset, provider glue, scorer adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.question import Question

pytest.importorskip("inspect_ai")

from inspect_ai.model import ChatMessageUser, ContentAudio
from inspect_ai.model import ContentText as InspectText

from lfm2_audio.inspect_bridge.audio import data_uri_to_waveform, waveform_to_data_uri
from lfm2_audio.inspect_bridge.dataset import question_set_dataset, resolve_dataset_path, to_sample
from lfm2_audio.inspect_bridge.provider import _last_user_turn
from lfm2_audio.inspect_bridge.scores import to_inspect_score
from lfm2_audio.inspect_bridge.task import voice_eval
from lfm2_audio.scorer.result import ScoreResult

# --------------------------------------------------------------------------- #
# Audio : le viewer n'accepte qu'une data URI
# --------------------------------------------------------------------------- #


def test_should_round_trip_a_waveform_through_a_data_uri() -> None:
    original = Waveform.of(np.linspace(-0.5, 0.5, 2400, dtype=np.float32), 24_000)

    restored = data_uri_to_waveform(waveform_to_data_uri(original))

    assert restored.sample_rate == 24_000
    assert restored.samples.size == 2400
    assert np.allclose(restored.samples, original.samples, atol=1e-4)


def test_data_uri_should_declare_the_mime_type_the_viewer_checks() -> None:
    """`isRenderableAudioSource` refuses anything else, and shows an inert link."""
    uri = waveform_to_data_uri(Waveform.of(np.zeros(240, dtype=np.float32), 24_000))

    assert uri.startswith("data:audio/wav;base64,")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_should_send_a_spoken_question_as_audio(tmp_path: Path) -> None:
    """The model must hear the question; a transcript would let it read instead."""
    wav = tmp_path / "q.wav"
    Waveform.of(np.zeros(1600, dtype=np.float32), 16_000).save(wav)
    question = Question(question_id="c1", text="transcript", audio_path=wav)

    sample = to_sample(question)

    assert isinstance(sample.input, list)
    parts = sample.input[0].content
    assert [type(p) for p in parts] == [ContentAudio]
    assert sample.metadata is not None
    assert sample.metadata["prompt_text"] == "transcript"


def test_should_send_a_written_question_as_text() -> None:
    sample = to_sample(Question(question_id="c1", text="What is the weather?"))

    assert sample.input == "What is the weather?"


def test_should_carry_the_expected_calls_into_metadata() -> None:
    expected = [{"name": "web_search", "arguments": {"query": "x"}}]

    sample = to_sample(Question(question_id="c1", text="q", expected_calls=expected))

    assert sample.metadata is not None
    assert sample.metadata["expected_calls"] == expected


def test_should_load_a_question_set_relative_to_the_repo(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id": "c1", "turns": [{"role": "user", "text": "hello"}]}))

    dataset = question_set_dataset(str(path), limit=1)

    assert len(dataset) == 1


def test_should_say_where_it_looked_when_the_dataset_is_missing() -> None:
    with pytest.raises(FileNotFoundError, match="introuvable"):
        resolve_dataset_path("nowhere/at/all.jsonl")


# --------------------------------------------------------------------------- #
# Provider : quelle question part au modèle
# --------------------------------------------------------------------------- #


def test_should_prefer_the_audio_over_its_transcript() -> None:
    audio = waveform_to_data_uri(Waveform.of(np.zeros(1600, dtype=np.float32), 16_000))
    message = ChatMessageUser(content=[InspectText(text="transcript"), ContentAudio(audio=audio, format="wav")])

    text, waveform = _last_user_turn([message])

    assert text is None
    assert waveform is not None


def test_should_pass_a_written_question_as_text() -> None:
    text, waveform = _last_user_turn([ChatMessageUser(content="hello")])

    assert (text, waveform) == ("hello", None)


# --------------------------------------------------------------------------- #
# Scores : ne jamais confondre « pas mesuré » et « mal mesuré »
# --------------------------------------------------------------------------- #


def test_should_translate_a_measured_result() -> None:
    score = to_inspect_score(ScoreResult.ok("utmos", 4.12, details={"duration_s": 3.0}))

    assert score is not None
    assert score.value == 4.12
    assert score.metadata == {"duration_s": 3.0}


@pytest.mark.parametrize(
    "result",
    [
        ScoreResult.unavailable("nisqa", "poids absents"),
        ScoreResult.skipped("wer", "aucun audio"),
        ScoreResult.failed("dnsmos", "boom"),
    ],
)
def test_should_refuse_to_turn_an_unmeasured_result_into_a_score(result: ScoreResult) -> None:
    assert to_inspect_score(result) is None


def test_voice_eval_should_accept_scorers_as_a_list() -> None:
    """Inspect turns `-T scorers=a,b` into a list before the task sees it."""
    task = voice_eval(
        questions="benchmark/lang_mirror/questions.jsonl",
        scorers=["lang_match", "dnsmos"],
        limit=1,
    )

    assert len(task.scorer) == 2


def test_audio_root_should_resolve_against_the_repo_root(tmp_path, monkeypatch) -> None:
    """inspect eval chdirs to the task file's directory; a repo-relative
    audio_root must keep pointing at the checkout."""
    monkeypatch.chdir(tmp_path)
    dataset = question_set_dataset(
        "benchmark/fleurs_fr_asr/questions.jsonl",
        audio_root="data/benchmark_audio/fleurs_fr",
        limit=1,
    )
    content = dataset[0].input[0].content
    audio = next(part for part in content if part.type == "audio")
    assert audio.audio.startswith("data:audio/wav;base64,")


def test_provider_should_glue_a_comma_split_system_prompt() -> None:
    """Inspect's -M parsing turns 'a, b' into ['a', ' b']; the tokenizer then
    crashes on the nested list. The provider glues it back before use."""
    from inspect_ai.model import GenerateConfig

    from lfm2_audio.inspect_bridge.provider import Lfm2AudioAPI

    api = Lfm2AudioAPI(
        model_name="lfm2/whatever",
        config=GenerateConfig(),
        system=["Transcribe exactly as spoken", " in the language spoken."],
    )

    assert api._system == "Transcribe exactly as spoken, in the language spoken."


def _task_state(input_content, target_text: str):
    from inspect_ai.model import ChatMessageUser, ModelName, ModelOutput
    from inspect_ai.scorer import Target
    from inspect_ai.solver import TaskState

    message = ChatMessageUser(content=input_content)
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s1",
        epoch=0,
        input=[message],
        messages=[message],
        target=Target(target_text),
        output=ModelOutput.from_content("mockllm/model", "the reply"),
        metadata={"prompt_text": "spoken transcript"},
    )


def test_reference_text_should_be_the_target_text_not_its_repr() -> None:
    """str(Target) is the object repr; the first campaign's WER compared
    replies against `<inspect_ai...Target object at 0x...>`."""
    from lfm2_audio.inspect_bridge.scorers import to_eval_sample

    sample = to_eval_sample(_task_state("hello", "the reference"))

    assert sample.reference_text == "the reference"


def test_prompt_text_should_fall_back_to_metadata_on_audio_only_input() -> None:
    """TaskState.input_text raises on an audio-only prompt (ASR campaigns)."""
    import numpy as np
    from inspect_ai.model import ContentAudio

    from lfm2_audio.ds.audio import OUTPUT_SAMPLE_RATE, Waveform
    from lfm2_audio.inspect_bridge.audio import waveform_to_data_uri
    from lfm2_audio.inspect_bridge.scorers import to_eval_sample

    wav = Waveform.of(np.zeros(OUTPUT_SAMPLE_RATE, dtype=np.float32), OUTPUT_SAMPLE_RATE)
    audio_only = [ContentAudio(audio=waveform_to_data_uri(wav), format="wav")]

    sample = to_eval_sample(_task_state(audio_only, "bonjour"))

    assert sample.prompt_text == "spoken transcript"


def test_provider_should_resolve_a_named_system_prompt() -> None:
    """Campaigns name their prompt so they can be reproduced — and so the text
    never travels on a command line that Inspect would split on commas."""
    from inspect_ai.model import GenerateConfig

    from lfm2_audio.core.prompt import BILINGUAL_SYSTEM
    from lfm2_audio.inspect_bridge.provider import Lfm2AudioAPI

    api = Lfm2AudioAPI(model_name="lfm2/whatever", config=GenerateConfig(), system="bilingual")

    assert api._system == BILINGUAL_SYSTEM


def test_provider_should_pass_an_unknown_system_through_unchanged() -> None:
    from inspect_ai.model import GenerateConfig

    from lfm2_audio.inspect_bridge.provider import Lfm2AudioAPI

    api = Lfm2AudioAPI(model_name="lfm2/whatever", config=GenerateConfig(), system="Transcris l'audio.")

    assert api._system == "Transcris l'audio."
