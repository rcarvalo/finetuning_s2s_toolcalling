"""``salient_terms`` — ce dont une question parle, jamais sa forme.

v4 a routé « When is the next presidential election in France? » vers la base
de clients : la forme interrogative primait le domaine, parce que les questions
`db_query` du corpus sont massivement en who/which/when. Les mots
interrogatifs doivent donc être invisibles ici.
"""

from __future__ import annotations

from lfm2_audio.data_prep.question_terms import leading_term, salient_terms, topic_phrase


class TestSalientTerms:
    def test_should_drop_interrogative_forms(self) -> None:
        terms = salient_terms("When is the next presidential election in France?")

        assert "when" not in terms
        assert "presidential" in terms

    def test_should_keep_the_subject_of_the_question(self) -> None:
        assert set(salient_terms("who won the World Cup in football")) == {"football", "world", "cup", "won"}

    def test_should_rank_longer_words_first(self) -> None:
        terms = salient_terms("who won the World Cup in football")

        assert terms[0] == "football"

    def test_should_not_repeat_a_term(self) -> None:
        assert salient_terms("cup cup CUP") == ["cup"]

    def test_should_honour_the_limit(self) -> None:
        assert len(salient_terms("alpha bravo charlie delta echo foxtrot golf", limit=3)) == 3

    def test_should_return_nothing_for_a_question_made_only_of_stopwords(self) -> None:
        assert salient_terms("what is it") == []


class TestLeadingTerm:
    def test_should_return_the_most_distinctive_term(self) -> None:
        assert leading_term("how heavy is the widget042") == "widget042"

    def test_should_return_none_when_there_is_nothing_to_name(self) -> None:
        assert leading_term("what is it") is None


class TestTopicPhrase:
    def test_should_strip_the_leading_interrogative(self) -> None:
        assert topic_phrase("what is the average order value") == "average order value"

    def test_should_keep_a_query_that_is_already_a_topic(self) -> None:
        assert topic_phrase("cryptocurrency definition") == "cryptocurrency definition"

    def test_should_not_end_on_a_function_word(self) -> None:
        # « results about the customers in » : la préposition orpheline
        # s'entend, et le corpus est lu à voix haute.
        assert topic_phrase("the customers in") == "customers"

    def test_should_cap_the_length(self) -> None:
        phrase = topic_phrase("alpha bravo charlie delta echo foxtrot golf", max_words=3)

        assert phrase == "alpha bravo charlie"

    def test_should_return_none_when_nothing_survives(self) -> None:
        assert topic_phrase("what is it") is None
