"""``NearDistractors`` — des voisins du bon sujet, pas des payloads au hasard.

En v4, un payload répondant « qui a gagné la Coupe du monde » côtoyait une
adresse e-mail : le sujet seul suffisait à isoler la bonne entrée. Le modèle a
appris ce raccourci, qui ne transfère pas — de vrais résultats parlent tous du
même sujet et ne diffèrent que par la présence de la réponse.
"""

from __future__ import annotations

import random
from typing import Any

from lfm2_audio.data_prep.near_distractors import NearDistractors

FOOTBALL = [
    ({"fact": "spain won the final"}, "who won the World Cup final"),
    ({"fact": "the final was in New Jersey"}, "where was the World Cup final played"),
    ({"fact": "torres scored"}, "who scored in the World Cup final"),
]
OFFICE = [
    ({"fact": "sarah works in sales"}, "which department does Sarah work in"),
    ({"fact": "the meeting is at three"}, "what time is the department meeting"),
    ({"fact": "invoices are due friday"}, "when are invoices due"),
]


def _pool() -> NearDistractors:
    return NearDistractors(FOOTBALL + OFFICE)


class TestNearDistractors:
    def test_should_report_its_size(self) -> None:
        assert len(_pool()) == 6

    def test_should_prefer_payloads_sharing_the_question_topic(self) -> None:
        picked = _pool().pick("who won the World Cup final", 2, random.Random(0))

        football = [payload for payload, _ in FOOTBALL]
        assert all(payload in football for payload in picked)

    def test_should_never_return_the_payload_it_was_told_to_exclude(self) -> None:
        target = FOOTBALL[0][0]

        picked = _pool().pick("who won the World Cup final", 5, random.Random(0), exclude=target)

        assert target not in picked

    def test_should_fall_back_to_the_whole_pool_when_nothing_overlaps(self) -> None:
        picked = _pool().pick("quantum chromodynamics", 3, random.Random(0))

        assert len(picked) == 3

    def test_should_return_what_it_can_when_asked_for_more_than_it_holds(self) -> None:
        picked = _pool().pick("who won the World Cup final", 99, random.Random(0))

        assert len(picked) == 6

    def test_should_give_back_the_question_a_payload_answered(self) -> None:
        # C'est ce qui permet au refus de NOMMER ce qui a été trouvé.
        assert _pool().question_of(FOOTBALL[1][0]) == "where was the World Cup final played"

    def test_should_return_an_empty_question_for_a_payload_it_does_not_hold(self) -> None:
        assert _pool().question_of({"fact": "unknown"}) == ""

    def test_should_return_nothing_from_an_empty_pool(self) -> None:
        empty: list[tuple[dict[str, Any], str]] = []

        assert NearDistractors(empty).pick("anything", 3, random.Random(0)) == []
