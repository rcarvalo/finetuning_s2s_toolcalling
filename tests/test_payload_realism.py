"""``PayloadRealism`` — payloads d'outil bruités et cas « réponse absente ».

Le corpus v3 ne contenait que des payloads mono-entrée contenant toujours la
réponse. v3 a donc appris à reformuler ce qu'elle voyait — et à **inventer**
quand le payload ne répondait pas.

v5 ajoute deux propriétés que ces tests pinnent : la réponse vit désormais
DANS une prose (le réel ne livre pas de champ), et l'aveu d'échec **nomme**
ce qui a été trouvé (les gabarits interchangeables de v4 ont produit du refus
par réflexe, mesuré à 3,15 d'honnêteté sous une porte de 4).
"""

from __future__ import annotations

from typing import Any

import pytest

from lfm2_audio.data_prep.payload_realism import PayloadRealism

FACT = "the {tag} weighs {tag} kilos"


def _dialogue(index: int, tool: str = "web_search") -> dict[str, Any]:
    tag = f"widget{index:03d}"
    return {
        "id": f"tc_{index:06d}",
        "turns": [
            {"role": "user", "text": f"how heavy is the {tag}", "audio": f"u{index}.wav"},
            {"role": "assistant", "tool_calls": [{"name": tool, "arguments": {"query": tag}}]},
            {"role": "tool", "content": {"results": FACT.format(tag=tag)}},
            {"role": "assistant", "text": f"answer {index}", "audio": f"a{index}.wav"},
        ],
    }


@pytest.fixture
def corpus() -> list[dict[str, Any]]:
    return [_dialogue(i) for i in range(60)]


def _content(dialogue: dict[str, Any]) -> dict[str, Any]:
    return next(t["content"] for t in dialogue["turns"] if t["role"] == "tool")


def _entries(dialogue: dict[str, Any]) -> list[dict[str, Any]]:
    content = _content(dialogue)
    return content.get("results") or content["rows"]


def _all_text(dialogue: dict[str, Any]) -> str:
    return " ".join(str(entry) for entry in _entries(dialogue))


def _spoken(dialogue: dict[str, Any]) -> dict[str, Any]:
    return next(t for t in dialogue["turns"] if t["role"] == "assistant" and t.get("text"))


class TestNoise:
    def test_should_turn_a_single_payload_into_several_entries(self, corpus: list[dict[str, Any]]) -> None:
        out, _ = PayloadRealism().apply(corpus)

        assert len(_entries(out[0])) >= 3

    def test_should_keep_the_true_fact_when_the_answer_is_present(self, corpus: list[dict[str, Any]]) -> None:
        out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

        for index, transformed in enumerate(out):
            assert FACT.format(tag=f"widget{index:03d}") in _all_text(transformed)

    def test_should_carry_the_fact_inside_prose_not_as_a_field(self, corpus: list[dict[str, Any]]) -> None:
        # Le réel ne renvoie pas `{"results": "<la réponse>"}` mais un snippet
        # verbeux : c'est ce que le modèle doit apprendre à lire.
        out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

        entry = next(e for e in _entries(out[0]) if FACT.format(tag="widget000") in e["snippet"])
        assert set(entry) == {"title", "url", "snippet"}
        assert len(entry["snippet"]) > len(FACT.format(tag="widget000"))

    def test_should_shape_database_results_as_rows(self) -> None:
        # Une base ne renvoie pas des pages web : mélanger les deux formes
        # apprendrait au modèle qu'un résultat SQL a une URL.
        corpus = [_dialogue(i, tool="db_query") for i in range(60)]

        out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

        content = _content(out[0])
        assert "rows" in content
        assert "url" not in content["rows"][0]


class TestAbsence:
    def test_should_drop_the_true_fact_when_absent(self, corpus: list[dict[str, Any]]) -> None:
        out, misses = PayloadRealism(miss_ratio=1.0).apply(corpus)

        assert misses == len(corpus)
        for index, transformed in enumerate(out):
            assert FACT.format(tag=f"widget{index:03d}") not in _all_text(transformed)

    def test_should_name_what_was_asked_in_the_refusal(self, corpus: list[dict[str, Any]]) -> None:
        # Un refus qui ne cite rien peut s'écrire sans lire le payload : c'est
        # exactement ce que v4 a appris, d'où le sur-refus.
        out, _ = PayloadRealism(miss_ratio=1.0).apply(corpus)

        assert "widget000" in _spoken(out[0])["text"]

    def test_should_drop_the_stale_audio_of_a_rewritten_answer(self, corpus: list[dict[str, Any]]) -> None:
        out, _ = PayloadRealism(miss_ratio=1.0).apply(corpus)

        assert "audio" not in _spoken(out[0])

    def test_should_keep_the_audio_of_an_unchanged_answer(self, corpus: list[dict[str, Any]]) -> None:
        out, _ = PayloadRealism(miss_ratio=0.0).apply(corpus)

        assert _spoken(out[0])["audio"] == "a0.wav"

    def test_should_default_to_a_rate_that_does_not_teach_reflex_refusal(self) -> None:
        # v4 tournait à 0,15 et sur-refusait ; 0,08 ne laissait que 146
        # exemples contre 297, au risque de rouvrir l'hallucination.
        assert PayloadRealism().miss_ratio == 0.12


class TestContract:
    def test_should_flag_the_loss_in_meta_not_at_the_dialogue_root(self, corpus: list[dict[str, Any]]) -> None:
        # `Dialogue` est en extra="forbid" : un drapeau posé à la racine fait
        # rejeter le dataset ENTIER au packing. `DialogueMeta` est extra="allow".
        out, misses = PayloadRealism(miss_ratio=0.5).apply(corpus)

        assert sum(d["meta"]["answer_absent"] for d in out) == misses
        assert all("answer_absent" not in d for d in out)

    def test_should_be_reproducible_for_a_given_seed(self, corpus: list[dict[str, Any]]) -> None:
        first, _ = PayloadRealism(seed=3).apply(corpus)
        second, _ = PayloadRealism(seed=3).apply(corpus)

        assert first == second

    def test_should_leave_dialogues_without_a_tool_turn_untouched(self) -> None:
        plain = {"id": "x", "turns": [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hello"}]}

        out, _ = PayloadRealism().apply([_dialogue(i) for i in range(60)] + [plain])

        assert out[-1] == plain

    def test_should_refuse_a_pool_too_small_to_build_distractors(self) -> None:
        with pytest.raises(ValueError, match="distracteurs"):
            PayloadRealism().apply([_dialogue(0)])

    def test_should_reject_a_corpus_the_dialogue_schema_would_refuse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # La validation vit DANS apply() : sans elle, un dialogue non conforme
        # n'échoue qu'au packing, sur une VM, une heure plus tard — et fait
        # rejeter le dataset entier.
        def refuse(_: dict[str, Any]) -> None:
            raise ValueError("schéma refusé")

        monkeypatch.setattr("lfm2_audio.ds.dialogue.parse_dialogue", refuse)

        with pytest.raises(ValueError, match="schéma refusé"):
            PayloadRealism().apply([_dialogue(i) for i in range(60)])
