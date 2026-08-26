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
        "many",
        "much",
    ]
)

# A word may open on a digit: requiring a letter turned "4G and 5G" into
# "g and g" in a real corpus pass, and that text gets spoken aloud.
_WORD = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'’-]*")


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


def topic_phrase(text: str, *, max_words: int = 8) -> str | None:
    """The phrase ``text`` is about, for a refusal to name out loud.

    Feed this the tool-call query, not the user's utterance. Ranking single
    words by length lands on whatever is longest — "the results are about
    natural, not effective" came out of a real corpus pass, because adjectives
    outran the subject. A query is already the distilled topic, so trimming its
    leading and trailing function words is enough and stays grammatical.
    """
    words = _WORD.findall(text)
    start = 0
    while start < len(words) and words[start].lower() in STOPWORDS:
        start += 1
    kept = words[start : start + max_words]
    while kept and kept[-1].lower() in STOPWORDS:
        kept.pop()
    return " ".join(word.lower() for word in kept) if kept else None
