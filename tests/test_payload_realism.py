"""``PayloadRealism`` — payloads d'outil bruités et cas « réponse absente ».

Le corpus v3 ne contenait que des payloads mono-entrée contenant toujours la
réponse. v3 a donc appris à reformuler ce qu'elle voyait — et à **inventer**
quand le payload ne répondait pas (« Sarah's email address is
sarah.johnson@example.com » sur une table sans colonne e-mail).

Ces tests pinnent les deux propriétés qui corrigent ça : le payload vrai
survit au bruit dans le cas nominal, et il disparaît *avec* une réponse
honnête dans le cas « absent ».
"""

from __future__ import annotations

from typing import Any

import pytest

from lfm2_audio.data_prep.payload_realism import MISS_ANSWERS, PayloadRealism


def _dialogue(index: int, tool: str = "web_search") -> dict[str, Any]:
    return {
        "id": f"tc_{index:06d}",
        "turns": [
            {"role": "user", "text": f"question {index}", "audio": f"u{index}.wav"},
            {"role": "assistant", "tool_calls": [{"name": tool, "arguments": {"query": "q"}}]},
            {"role": "tool", "content": {"results": f"fact number {index}"}},
            {"role": "assistant", "text": f"answer {index}", "audio": f"a{index}.wav"},
        ],
    }


@pytest.fixture
def corpus() -> list[dict[str, Any]]:
    return [_dialogue(i) for i in range(60)]


def _entries(dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    content = next(t["content"] for t in dialogue["turns"] if t["role"] == "tool")
    return content.get("results") or content["rows"]


def _spoken(dialogue: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in dialogue["turns"] if t["role"] == "assistant" and t.get("text"))


def test_should_turn_a_single_payload_into_several_entries(corpus: list[dict[str, Any]]) -> None:
    out, _ = PayloadRealism().apply(corpus)

    assert len(_entries(out[0])) >= 3


def test_should_keep_the_true_payload_when_the_answer_is_present(corpus: list[dict[str, Any]]) -> None:
    out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

    for original, transformed in zip(corpus, out, strict=True):
        fact = original["turns"][2]["content"]["results"]
        assert any(entry.get("results") == fact for entry in _entries(transformed))


def test_should_drop_the_true_payload_and_answer_honestly_when_absent(corpus: list[dict[str, Any]]) -> None:
    out, misses = PayloadRealism(miss_ratio=1.0).apply(corpus)

    assert misses == len(corpus)
    for original, transformed in zip(corpus, out, strict=True):
        fact = original["turns"][2]["content"]["results"]
        assert not any(entry.get("results") == fact for entry in _entries(transformed))
        assert _spoken(transformed)["text"] in MISS_ANSWERS


def test_should_drop_the_stale_audio_of_a_rewritten_answer(corpus: list[dict[str, Any]]) -> None:
    out, _ = PayloadRealism(miss_ratio=1.0).apply(corpus)

    assert "audio" not in _spoken(out[0])


def test_should_keep_the_audio_of_an_unchanged_answer(corpus: list[dict[str, Any]]) -> None:
    out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

    assert _spoken(out[0])["audio"] == "a0.wav"


def test_should_shape_web_results_as_documents(corpus: list[dict[str, Any]]) -> None:
    out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

    assert {"title", "url"} <= set(_entries(out[0])[0])


def test_should_shape_database_results_as_rows() -> None:
    # Une base ne renvoie pas des pages web : mélanger les deux formes
    # apprendrait au modèle qu'un résultat SQL a une URL.
    corpus = [_dialogue(i, tool="db_query") for i in range(60)]

    out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

    content = next(t["content"] for t in out[0]["turns"] if t["role"] == "tool")
    assert "rows" in content
    assert "url" not in content["rows"][0]


def test_should_flag_which_dialogues_lost_their_answer(corpus: list[dict[str, Any]]) -> None:
    out, misses = PayloadRealism(miss_ratio=0.5).apply(corpus)

    assert sum(d["answer_absent"] for d in out) == misses


def test_should_be_reproducible_for_a_given_seed(corpus: list[dict[str, Any]]) -> None:
    first, _ = PayloadRealism(seed=3).apply(corpus)
    second, _ = PayloadRealism(seed=3).apply(corpus)

    assert first == second


def test_should_leave_dialogues_without_a_tool_turn_untouched() -> None:
    plain = {"id": "x", "turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]}

    out, _ = PayloadRealism().apply([_dialogue(i) for i in range(60)] + [plain])

    assert out[-1] == plain


def test_should_refuse_a_pool_too_small_to_build_distractors() -> None:
    with pytest.raises(ValueError, match="distracteurs"):
        PayloadRealism().apply([_dialogue(0)])
