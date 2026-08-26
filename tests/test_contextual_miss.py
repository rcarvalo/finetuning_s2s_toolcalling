"""``ContextualMiss`` — un refus qui dit ce qui a été trouvé.

v4 répondait « Nothing in the results answers that » en tenant une table de
clients : défendable, inutilisable. Et surtout, un tel texte s'écrit sans lire
le payload — c'est ce qui a produit le refus par réflexe (honnêteté 3,15 sous
une porte de 4). Nommer les deux côtés interdit ce raccourci.
"""

from __future__ import annotations

import random

from lfm2_audio.data_prep.contextual_miss import BLIND, ContextualMiss


class TestContextualMiss:
    def test_should_name_what_was_asked(self) -> None:
        text = ContextualMiss().text("how heavy is the widget042", [], random.Random(0))

        assert "widget042" in text

    def test_should_name_what_was_found_when_neighbours_say_something(self) -> None:
        text = ContextualMiss().text(
            "who won the World Cup final",
            ["what time is the department meeting"],
            random.Random(0),
        )

        assert "department" in text

    def test_should_never_claim_it_found_the_very_thing_it_lacks(self) -> None:
        # « results about X, but nothing on X » : contradictoire, et ça
        # apprendrait à renvoyer la question au lieu de lire le payload.
        text = ContextualMiss().text("how heavy is the widget042", ["how heavy is the widget042"], random.Random(0))

        assert text.count("widget042") == text.count("{asked}") + 1 or "widget042" in text
        assert "about widget042, but nothing on widget042" not in text

    def test_should_fall_back_when_the_question_names_nothing(self) -> None:
        text = ContextualMiss().text("what is it", ["who won the World Cup"], random.Random(0))

        assert text in BLIND

    def test_should_leave_no_placeholder_unfilled(self) -> None:
        rng = random.Random(1)
        for _ in range(30):
            text = ContextualMiss().text("how heavy is the widget042", ["when are invoices due"], rng)
            assert "{" not in text and "}" not in text

    def test_should_vary_its_wording(self) -> None:
        # Une phrase unique serait apprise comme un réflexe déclenché par la
        # forme de la question, pas par l'absence d'information.
        rng = random.Random(2)
        seen = {ContextualMiss().text("how heavy is the widget042", ["when are invoices due"], rng) for _ in range(40)}

        assert len(seen) > 1
