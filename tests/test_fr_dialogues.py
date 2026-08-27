"""Tests of brick C generation (French dialogues and code-switch)."""

from __future__ import annotations

import pytest

from lfm2_audio.data_prep.fr_dialogues import (
    DialogueError,
    DialogueTurn,
    FrDialogue,
    build_code_switch_prompt,
    build_fr_prompt,
    code_switch_rate,
    parse_dialogues,
)

FR_USER = "Bonjour, j'ai rendez-vous avec madame Perrin à quinze heures."
FR_REPLY = "Bien sûr, je préviens madame Perrin tout de suite. Vous pouvez patienter ici."
EN_USER = "Sorry, could we switch to English? I have a meeting with the sales team."
EN_REPLY = "Of course. I will let the sales team know that you have arrived."


def _dialogue(*turns: tuple[str, str], kind: str = "fr") -> FrDialogue:
    from lfm2_audio.data_prep.fr_dialogues import detect_turn_language

    return FrDialogue(
        dialogue_id="c_0001",
        turns=tuple(DialogueTurn(role=role, text=text, lang=detect_turn_language(text)) for role, text in turns),
        kind=kind,
    )


def test_expected_language_is_the_last_user_turn_not_the_first() -> None:
    """In a code-switch dialogue, answering in the language the user opened
    with is exactly the failure being trained away."""
    dialogue = _dialogue(("user", FR_USER), ("assistant", FR_REPLY), ("user", EN_USER), ("assistant", EN_REPLY))

    assert dialogue.expected_lang == "en"


def test_should_accept_a_well_formed_dialogue() -> None:
    _dialogue(("user", FR_USER), ("assistant", FR_REPLY)).validate()


def test_should_reject_an_assistant_answering_in_the_wrong_language() -> None:
    dialogue = _dialogue(("user", FR_USER), ("assistant", FR_REPLY), ("user", EN_USER), ("assistant", FR_REPLY))

    with pytest.raises(DialogueError, match="entraîne à corriger"):
        dialogue.validate()


def test_should_reject_a_turn_too_long_to_be_spoken() -> None:
    dialogue = _dialogue(("user", FR_USER), ("assistant", "Alors, " + "voilà une phrase bien trop longue. " * 12))

    with pytest.raises(DialogueError, match="ce n'est plus de l'oral"):
        dialogue.validate()


def test_should_reject_a_dialogue_not_starting_with_the_user() -> None:
    dialogue = _dialogue(("assistant", FR_REPLY), ("user", FR_USER))

    with pytest.raises(DialogueError, match="tour utilisateur"):
        dialogue.validate()


def test_parse_should_drop_invalid_dialogues_and_keep_the_rest() -> None:
    payload = [
        {"topic": "accueil", "turns": [{"role": "user", "text": FR_USER}, {"role": "assistant", "text": FR_REPLY}]},
        {"topic": "cassé", "turns": [{"role": "user", "text": FR_USER}]},  # pas de réponse
    ]

    dialogues = parse_dialogues(payload, prefix="c_fr", kind="fr")

    assert len(dialogues) == 1
    assert dialogues[0].dialogue_id == "c_fr_0000"
    assert dialogues[0].topic == "accueil"


def test_declared_language_should_not_be_trusted_over_the_text() -> None:
    """The generator is the thing being checked: a mislabelled turn would teach
    the model to answer in the wrong language."""
    payload = [
        {
            "turns": [
                {"role": "user", "text": EN_USER, "lang": "fr"},  # étiquette fausse
                {"role": "assistant", "text": EN_REPLY, "lang": "fr"},
            ]
        }
    ]

    dialogues = parse_dialogues(payload, prefix="c_cs", kind="code_switch")

    assert dialogues[0].turns[0].lang == "en"


def test_code_switch_rate_should_count_real_switches() -> None:
    switched = _dialogue(("user", FR_USER), ("assistant", FR_REPLY), ("user", EN_USER), ("assistant", EN_REPLY))
    monolingual = _dialogue(("user", FR_USER), ("assistant", FR_REPLY))

    assert code_switch_rate([switched, monolingual]) == 0.5
    assert code_switch_rate([]) == 0.0


def test_case_export_carries_the_expected_language() -> None:
    case = _dialogue(("user", FR_USER), ("assistant", FR_REPLY), ("user", EN_USER), ("assistant", EN_REPLY)).as_case()

    assert case["meta"]["expected_lang"] == "en"
    assert case["turns"][0]["role"] == "user"


def test_prompts_should_state_the_spoken_register_and_the_language_rule() -> None:
    fr = build_fr_prompt(10, "accueil d'un visiteur")
    switch = build_code_switch_prompt(10, "accueil d'un visiteur")

    assert "ORAL" in fr and "10" in fr
    assert "DERNIER tour" in switch
