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


class TestStrictGuard:
    """v5.1 : aucun mot plein en commun entre « trouvé » et « demandé ».

    v5 laissait passer tout recouvrement sauf l'inclusion ; 210 refus sur 213
    nommaient des sujets quasi identiques et le modèle a appris à se contredire
    (docs/v5_report.md). La rareté de la forme riche est le comportement voulu.
    """

    def test_should_refuse_a_neighbour_sharing_any_content_word(self) -> None:
        text = ContextualMiss().text(
            "current delivery status of order number 78901",
            ["current status of order number o-45678"],
            random.Random(0),
        )

        assert "o-45678" not in text
        assert "current delivery status of order number 78901" in text

    def test_should_still_name_a_genuinely_different_neighbour(self) -> None:
        text = ContextualMiss().text("current price of silver per ounce", ["planets in solar system"], random.Random(0))

        assert "planets in solar system" in text

    def test_should_never_produce_a_self_contradiction(self) -> None:
        rng = random.Random(3)
        asks = ["current price of gold", "latest news about humanoid robots", "phone number for customer innovate"]
        neighbours = ["current gold price today", "humanoid robots latest news", "phone number for john smith"]
        for ask, near in zip(asks, neighbours, strict=True):
            for _ in range(20):
                text = ContextualMiss().text(ask, [near], rng)
                assert text.count(ask) <= 2 and near not in text
