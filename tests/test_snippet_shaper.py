"""``SnippetShaper`` — la réponse dans une prose, comme un vrai moteur.

v4 livrait la réponse comme un champ. DuckDuckGo renvoie 400 caractères où le
fait utile est noyé parmi d'autres. Un modèle entraîné à lire un champ n'a
jamais eu à trouver une phrase — la raison la plus probable du refus mesuré
sur un payload qui répondait pourtant quatre fois.
"""

from __future__ import annotations

import random

from lfm2_audio.data_prep.snippet_shaper import DATE_PREFIXES, SnippetShaper

ANSWER = {"results": "Spain won the 2026 final."}
NOISE = [{"results": "The venue was in New Jersey."}, {"results": "Torres scored in extra time."}]


class TestRender:
    def test_should_keep_a_sentence_as_a_sentence(self) -> None:
        assert SnippetShaper().render(ANSWER) == "Spain won the 2026 final."

    def test_should_turn_a_keyed_scalar_into_prose(self) -> None:
        assert SnippetShaper().render({"largest_desert": "Antarctic"}) == "largest desert: Antarctic."

    def test_should_render_numbers_and_booleans(self) -> None:
        rendered = SnippetShaper().render({"count": 3, "open": True})

        assert "count: 3." in rendered
        assert "open: yes." in rendered

    def test_should_render_lists_and_nested_dicts(self) -> None:
        rendered = SnippetShaper().render({"cities": ["Paris", "Lyon"], "host": {"name": "Ada"}})

        assert "Paris, Lyon" in rendered
        assert "name Ada" in rendered

    def test_should_skip_empty_values(self) -> None:
        assert SnippetShaper().render({"a": "", "b": "kept."}) == "kept."


class TestSnippet:
    def test_should_contain_the_answer(self) -> None:
        text = SnippetShaper().snippet(ANSWER, NOISE, random.Random(0))

        assert "Spain won the 2026 final." in text

    def test_should_be_longer_than_the_answer_alone(self) -> None:
        text = SnippetShaper().snippet(ANSWER, NOISE, random.Random(0))

        assert len(text) > len("Spain won the 2026 final.") + 10

    def test_should_open_with_a_date_like_a_real_result(self) -> None:
        text = SnippetShaper().snippet(ANSWER, NOISE, random.Random(0))

        assert any(text.startswith(prefix) for prefix in DATE_PREFIXES)

    def test_should_not_always_place_the_answer_first(self) -> None:
        # Une position fixe s'apprend comme une position, pas comme une lecture.
        rng = random.Random(0)
        starts = {SnippetShaper().snippet(ANSWER, NOISE, rng).index("Spain") for _ in range(40)}

        assert len(starts) > 1

    def test_should_work_without_any_filler(self) -> None:
        text = SnippetShaper().snippet(ANSWER, [], random.Random(0))

        assert "Spain won the 2026 final." in text


class TestNoiseSnippet:
    def test_should_not_contain_the_answer(self) -> None:
        text = SnippetShaper().noise_snippet(NOISE, random.Random(0))

        assert "Spain" not in text

    def test_should_still_look_like_a_result_when_there_is_no_filler(self) -> None:
        text = SnippetShaper().noise_snippet([], random.Random(0))

        assert any(text.startswith(prefix) for prefix in DATE_PREFIXES)
        assert len(text) > 20
