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
        """A phrase describing the neighbours, never the one that was asked.

        Sharing a word is expected and welcome: the distractors are drawn on
        topic, so "results about the World Cup venue, but nothing on the World
        Cup winner" is both realistic and the most useful thing to say. Only a
        topic that says nothing *new* is refused — one whose content words are
        contained in the ask, or contain it — since "about X, but nothing on X"
        contradicts itself and would teach the model to echo the question back
        instead of reading the payload.
        """
        asked_words = set(salient_terms(asked))
        for question in neighbour_questions:
            topic = topic_phrase(question)
            if not topic:
                continue
            words = set(salient_terms(topic))
            if words and not (words <= asked_words or asked_words <= words):
                return topic
        return None
