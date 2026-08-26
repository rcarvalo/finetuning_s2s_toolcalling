"""L'axe « forme interrogative » du générateur de corpus.

v4 a routé « When is the next presidential election in France? » vers la base
de clients, deux fois de suite et sans rapport avec l'audio : les cas
``db_query`` du corpus étaient massivement en who/which/when, si bien que la
forme de la question primait le domaine.

La correction est un axe tiré INDÉPENDAMMENT de l'outil. Ces tests pinnent les
trois propriétés qui la portent : la consigne atteint le prompt, elle interdit
au générateur de changer de cible, et la forme retenue survit jusqu'au
dialogue — sans quoi on ne pourrait pas vérifier la décorrélation après coup.
"""

from __future__ import annotations

from lfm2_audio.data_prep import synth_dialogues as sd

TOOLS: list[dict] = [{"name": "web_search", "parameters": {}}]


def _prompt(form: str = "", target: sd.ToolTarget = "web_search") -> str:
    return sd.build_generation_prompt(
        target=target, style="polite question", depth="explicit arguments", n=5, tool_definitions=TOOLS, form=form
    )


class TestQuestionForms:
    def test_should_offer_every_interrogative_form_a_tool_could_take(self) -> None:
        joined = " ".join(sd.QUESTION_FORMS)

        for word in ("who", "which", "when", "where", "how"):
            assert f"{word}-question" in joined

    def test_should_offer_a_non_question_form(self) -> None:
        # Sans elle, l'axe apprendrait que tout appel d'outil est une question.
        assert any("not a question" in form for form in sd.QUESTION_FORMS)


class TestPrompt:
    def test_should_carry_the_form_into_the_prompt(self) -> None:
        assert "a when-question about a date or a schedule" in _prompt("a when-question about a date or a schedule")

    def test_should_forbid_the_form_from_changing_the_tool(self) -> None:
        # C'est la phrase qui empêche le générateur de « corriger » la cible
        # vers celle que la forme suggère — et donc de recréer le raccourci.
        assert "must not change which tool applies" in _prompt("a when-question about a date or a schedule")

    def test_should_stay_unchanged_when_no_form_is_asked(self) -> None:
        assert "must not change which tool applies" not in _prompt()

    def test_should_apply_the_same_form_to_either_tool(self) -> None:
        form = "a when-question about a date or a schedule"

        assert form in _prompt(form, target="web_search")
        assert form in _prompt(form, target="db_query")


class TestTraceability:
    def test_should_keep_the_form_on_a_parsed_case(self) -> None:
        cases = sd.parse_generation_response(
            '[{"utterance": "when did Spain win", "tool": "web_search", "arguments": {"query": "spain"}}]',
            target="web_search",
            style="polite question",
            depth="explicit arguments",
            form="a when-question about a date or a schedule",
        )

        assert cases[0].form == "a when-question about a date or a schedule"

    def test_should_write_the_form_into_the_dialogue_meta(self) -> None:
        # Sans trace, impossible de mesurer après coup si forme et outil sont
        # bien décorrélés — c'est la seule garde contre le retour du défaut.
        case = sd.SynthCase(
            utterance="when did Spain win",
            target="web_search",
            arguments={"query": "spain"},
            form="a when-question about a date or a schedule",
        )

        dialogue = sd.case_to_dialogue(case, 0, tools=["web_search"])

        assert dialogue["meta"]["form"] == "a when-question about a date or a schedule"

    def test_should_leave_the_form_empty_when_none_was_used(self) -> None:
        dialogue = sd.case_to_dialogue(sd.SynthCase(utterance="hi", target="none", answer="hello"), 0, tools=[])

        assert dialogue["meta"]["form"] == ""
