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
    """Les entrées sont des REQUÊTES d'outil, pas des énoncés bruts.

    Lues sur l'énoncé, les phrases sortaient « results about natural, not
    effective » — un adjectif pris pour un sujet, constaté sur une vraie passe
    du corpus. La requête est le sujet déjà distillé.
    """

    def test_should_name_what_was_asked(self) -> None:
        text = ContextualMiss().text("cryptocurrency definition", [], random.Random(0))

        assert "cryptocurrency definition" in text

    def test_should_name_what_was_found_when_neighbours_say_something(self) -> None:
        text = ContextualMiss().text("cryptocurrency definition", ["department meeting time"], random.Random(0))

        assert "department meeting time" in text

    def test_should_strip_the_leading_interrogative_of_a_query(self) -> None:
        text = ContextualMiss().text("what is the average order value", [], random.Random(0))

        assert "average order value" in text
        assert "what is the average" not in text

    def test_should_never_claim_it_found_the_very_thing_it_lacks(self) -> None:
        # « results about X, but nothing on X » : contradictoire, et ça
        # apprendrait à renvoyer la question au lieu de lire le payload.
        text = ContextualMiss().text("cryptocurrency definition", ["cryptocurrency basics"], random.Random(0))

        assert "about cryptocurrency" not in text.split("nothing")[0] or "basics" not in text

    def test_should_ignore_function_words_when_comparing_topics(self) -> None:
        # « the » partagé ne fait pas deux sujets identiques : sans ça, le
        # repli s'enclenchait à tort et le refus ne nommait plus rien.
        text = ContextualMiss().text("the world cup winner", ["the department meeting"], random.Random(0))

        assert "department meeting" in text

    def test_should_fall_back_when_the_query_names_nothing(self) -> None:
        text = ContextualMiss().text("what is it", ["world cup winner"], random.Random(0))

        assert text in BLIND

    def test_should_leave_no_placeholder_unfilled(self) -> None:
        rng = random.Random(1)
        for _ in range(30):
            text = ContextualMiss().text("cryptocurrency definition", ["invoice due dates"], rng)
            assert "{" not in text and "}" not in text

    def test_should_vary_its_wording(self) -> None:
        # Une phrase unique serait apprise comme un réflexe déclenché par la
        # forme de la question, pas par l'absence d'information.
        rng = random.Random(2)
        seen = {ContextualMiss().text("cryptocurrency definition", ["invoice due dates"], rng) for _ in range(40)}

        assert len(seen) > 1
