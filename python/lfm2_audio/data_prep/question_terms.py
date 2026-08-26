"""Pull the terms a question is *about*, with no model in the loop.

Three v5 pieces need the same thing: near distractors need to know what a
question is about to find neighbours, contextual refusals need a term to name,
and snippets need filler drawn from the right topic. A shared extractor keeps
those three consistent — a distractor judged near on one notion and a refusal
naming another would teach contradictory lessons.

Deliberately lexical. An LLM pass here would cost a GPU run per corpus rebuild
and buy accuracy the downstream uses do not need: a distractor only has to be
plausibly on-topic, and a refusal only has to name a word the user said.
"""

from __future__ import annotations

import re

# Wh-words and auxiliaries carry the *form* of a question, never its subject —
# and the form is exactly what v4 over-fitted (a "when" question routed to the
# customer database). They must never count as what a question is about.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "out",
        "against",
        "during",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "who",
        "whom",
        "whose",
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "there",
        "here",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "as",
        "too",
        "very",
        "just",
        "now",
        "also",
        "only",
        "own",
        "same",
        "please",
        "tell",
        "show",
        "give",
        "find",
        "get",
        "know",
        "say",
        "said",
        "like",
        "want",
        "need",
    ]
)

_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9'’-]*")


def salient_terms(text: str, *, limit: int = 6) -> list[str]:
    """Content words of ``text``, most distinctive first, without duplicates.

    Longer words rank first: on questions of this length it separates the
    subject ("presidential", "election") from filler far more reliably than
    frequency, which needs a corpus this runs without.
    """
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text):
        word = match.group(0)
        lowered = word.lower()
        if lowered in STOPWORDS or len(lowered) < 3:
            continue
        seen.setdefault(lowered, None)
    ranked = sorted(seen, key=lambda w: (-len(w), w))
    return ranked[:limit]


def leading_term(text: str) -> str | None:
    """The single term a refusal should name, or None if there is nothing."""
    terms = salient_terms(text, limit=1)
    return terms[0] if terms else None
