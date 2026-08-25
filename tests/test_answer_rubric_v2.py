"""Rubrique de jugement v2 (`ANSWER_RUBRIC_V2`).

Deux défauts de la v1, révélés en jugeant v3, sont corrigés ici et pinnés.

1. Aucun critère ne mesurait le mode d'échec dominant à l'usage : réciter le
   résultat d'outil quand il ne contient pas la réponse. D'où `honesty`.
2. `grounding` pesait 1.5 et noyait la pertinence : le cas « who won the last
   Ballon d'Or ? » sortait à 0,775 avec une pertinence de 1/5.
"""

from __future__ import annotations

from lfm2_audio.scorer.text.rubric import ANSWER_RUBRIC_V2, REASONING_RUBRIC


def test_should_be_versioned_apart_from_v1() -> None:
    # Deux campagnes jugées sur des rubriques différentes ne sont pas
    # comparables : la version est ce qui rend la bascule visible.
    assert ANSWER_RUBRIC_V2.version != REASONING_RUBRIC.version


def test_should_add_the_honesty_criterion() -> None:
    assert "honesty" in ANSWER_RUBRIC_V2.keys
    assert "honesty" not in REASONING_RUBRIC.keys


def test_should_not_let_grounding_outweigh_relevance() -> None:
    weights = {c.key: c.weight for c in ANSWER_RUBRIC_V2.criteria}

    assert weights["grounding"] <= weights["relevance"]


def test_should_score_the_ballon_dor_failure_low() -> None:
    # Ancré mais hors-sujet et malhonnête : l'agrégat doit s'effondrer, là où
    # la v1 le notait 0,775.
    scores = {"relevance": 1.0, "grounding": 5.0, "honesty": 1.0, "coherence": 5.0, "conciseness": 4.0}

    assert ANSWER_RUBRIC_V2.weighted_mean(scores) < REASONING_RUBRIC.weighted_mean(scores)
    assert ANSWER_RUBRIC_V2.weighted_mean(scores) < 0.65


def test_should_still_reward_an_honest_miss() -> None:
    # « Je n'ai pas trouvé ça dans les résultats » est la BONNE réponse quand
    # le payload ne répond pas : elle ne doit pas être notée comme un échec.
    scores = {"relevance": 4.0, "grounding": 5.0, "honesty": 5.0, "coherence": 5.0, "conciseness": 5.0}

    assert ANSWER_RUBRIC_V2.weighted_mean(scores) > 0.9


def test_should_expose_every_criterion_in_the_prompt() -> None:
    block = ANSWER_RUBRIC_V2.as_prompt_block()

    for key in ANSWER_RUBRIC_V2.keys:
        assert key in block
