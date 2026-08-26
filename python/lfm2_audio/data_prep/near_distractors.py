"""Draw distractors that are *about the same thing* as the question.

v4 drew them from arbitrary other dialogues, so a payload answering "who won
the World Cup" sat next to an email address. Topic alone told the answering
entry apart, and the model learned that shortcut. Real search results share the
query's topic and differ only in whether they carry the answer — which is the
discrimination the model must actually make, and the one it fails in the wild:
asked who won the 2026 World Cup, v4 refused while the payload said "Spain
have been crowned FIFA World Cup 2026 winners" four times over.

Selection is lexical overlap on :mod:`question_terms`. Neighbours only need to
be plausibly on-topic; ranking them precisely would buy nothing here.
"""

from __future__ import annotations

import random
from typing import Any

from lfm2_audio.data_prep.question_terms import salient_terms


class NearDistractors:
    """A pool of payloads, searchable by what their question was about."""

    def __init__(self, entries: list[tuple[dict[str, Any], str]]) -> None:
        """``entries`` pairs each payload with the question text it answered."""
        self._payloads = [payload for payload, _ in entries]
        self._questions = [question for _, question in entries]
        self._terms = [frozenset(salient_terms(question)) for _, question in entries]

    def __len__(self) -> int:
        return len(self._payloads)

    def question_of(self, payload: dict[str, Any]) -> str:
        """The question a pooled payload answered, for naming what was found."""
        for candidate, question in zip(self._payloads, self._questions, strict=True):
            if candidate is payload:
                return question
        return ""

    def pick(
        self,
        question: str,
        count: int,
        rng: random.Random,
        *,
        exclude: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """``count`` payloads, preferring those sharing terms with ``question``.

        Sampling among the near candidates rather than taking the top ``count``
        keeps the corpus varied: a deterministic top-N would pair the same
        payloads together in every epoch.
        """
        wanted = frozenset(salient_terms(question))
        scored = [
            (len(wanted & terms), index)
            for index, terms in enumerate(self._terms)
            if self._payloads[index] is not exclude
        ]
        if not scored:
            return []
        scored.sort(key=lambda pair: -pair[0])

        # Draw from a window wider than needed so the near payloads vary
        # between dialogues; a deterministic top-N would pair the same ones
        # together in every epoch.
        near = [index for score, index in scored if score > 0]
        chosen = rng.sample(near[: max(count * 3, count)], min(count, len(near)))
        if len(chosen) < count:
            # Top up from off-topic payloads rather than return a short list: a
            # search engine always fills the page, and a result count that
            # shrank with topic coverage would itself become a cue.
            far = [index for score, index in scored if score == 0]
            chosen += rng.sample(far, min(count - len(chosen), len(far)))
        return [self._payloads[index] for index in chosen]
