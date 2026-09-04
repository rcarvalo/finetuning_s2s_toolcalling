"""Write a refusal that says what came back, not just that something didn't.

v4's five refusals were interchangeable templates naming nothing: asked for the
best customer, it answered "Nothing in the results answers that" while holding
a customer table with no revenue column. Defensible and useless — the user
cannot tell what the assistant has, so cannot reformulate.

Worse for training: a refusal that never refers to the payload can be produced
without reading it, so the model learns to fire on the *shape* of a question.
That is the over-refusal measured in v4 (honesty 3.15, below the gate of 4).

Naming both sides forces the opposite: the text carries a term from the
question and a term from what was found, so neither can be written without
looking at both.
"""

from __future__ import annotations

import random

from lfm2_audio.data_prep.question_terms import salient_terms, topic_phrase

BOTH_SIDES = (
    "I'm seeing results about {found}, but nothing on {asked}. Want me to search for {asked} directly?",
    "What came back covers {found} — no {asked}. Should I look again with different terms?",
    "The results are about {found}, not {asked}. Want me to try a narrower search for {asked}?",
    "I found {found} in there, but nothing that mentions {asked}. Shall I search again?",
)
"""Refusals naming what was found *and* what was asked."""

ASKED_ONLY = (
    "I got results back, but none of them mention {asked}. Want me to search for {asked} specifically?",
    "Nothing in what came back covers {asked}. Should I try different terms?",
    "The results don't mention {asked} anywhere. Want me to look again?",
)
"""Fallback when nothing usable names the neighbours — still names the ask."""

BLIND = (
    "I couldn't find that in the results. Want me to search differently?",
    "The results don't cover that — should I try another search?",
)
"""Last resort. Kept few and deliberately vague: these are exactly the v4
templates, and a corpus made only of them is what taught refusal-by-reflex."""


class ContextualMiss:
    """Builds the answer given when the payload does not hold the answer."""

    def text(self, question: str, neighbour_questions: list[str], rng: random.Random) -> str:
        """A refusal grounded in this query and these neighbouring results.

        ``question`` and ``neighbour_questions`` should be the tool-call
        queries, not the raw utterances — see :func:`topic_phrase`.
        """
        asked = topic_phrase(question)
        if asked is None:
            return rng.choice(BLIND)

        found = self._found_topic(neighbour_questions, asked)
        if found is None:
            return rng.choice(ASKED_ONLY).format(asked=asked)
        return rng.choice(BOTH_SIDES).format(found=found, asked=asked)

    @staticmethod
    def _found_topic(neighbour_questions: list[str], asked: str) -> str | None:
        """A phrase describing the neighbours, sharing NO content word with the ask.

        Strict on purpose, and this is the v5.1 correction. The distractors are
        drawn on topic, so their subjects overlap the ask by construction; v5
        let any overlap short of containment through, and 210 of its 213
        two-sided refusals named near-identical topics ("current status of
        order o-45678" against "current delivery status of order 78901"). The
        model collapsed the two slots and learned to contradict itself:
        "I found current price of gold, but nothing that mentions current
        price of gold" (docs/v5_report.md).

        With near distractors there is usually no genuinely different subject
        to name, so this returns None most of the time and the refusal names
        only the ask. That rarity is the correct behaviour, not a defect.
        """
        asked_words = set(salient_terms(asked))
        for question in neighbour_questions:
            topic = topic_phrase(question)
            if not topic:
                continue
            words = set(salient_terms(topic))
            if words and not (words & asked_words):
                return topic
        return None
