"""Shape a payload into a search snippet the way a real engine returns one.

v4 handed the model tidy structured entries — ``{"title": ..., "url": ...,
"largest_desert": "Antarctic Polar Desert"}`` — where the answer *is* a field.
DuckDuckGo returns nothing of the sort. It returns prose:

    "Jul 20, 2026 · Spain have been crowned FIFA World Cup 2026 winners
    following their 1-0 victory over Argentina after extra-time in the final
    at New York New Jersey Stadium. Here is a full breakdown of the 2026 FIFA
    World Cup. Who won the tournament? ..."

Four hundred characters, several facts, the useful one buried among them. A
model trained to read a field has never had to find a sentence, which is the
likeliest reason v4 refused on a payload that answered the question outright.

The padding is drawn from neighbouring payloads rather than invented: a real
page about the World Cup does mention the group stage and the venue, so
same-topic facts are the honest noise. Fixed boilerplate would teach the model
to skip a constant prefix instead of to read.
"""

from __future__ import annotations

import random
from typing import Any

DATE_PREFIXES = ("Jan 14, 2026 ·", "Mar 3, 2026 ·", "Jun 22, 2026 ·", "Jul 20, 2026 ·", "2 days ago -", "5 days ago -")
"""Real snippets almost always open with one. Kept as noise the model must
learn to step over — never as a cue, since it appears on every entry alike."""


class SnippetShaper:
    """Renders payloads as prose snippets, the answer buried among neighbours."""

    def __init__(self, *, min_fillers: int = 1, max_fillers: int = 3) -> None:
        self._min_fillers = min_fillers
        self._max_fillers = max_fillers

    def render(self, payload: dict[str, Any]) -> str:
        """Flatten a payload to a sentence, whatever shape it has.

        The corpus uses 200+ payload shapes; rendering values rather than
        pattern-matching keys is what keeps this independent of them.
        """
        parts: list[str] = []
        for key, value in payload.items():
            text = self._stringify(value)
            if not text:
                continue
            # A bare sentence stays a sentence; a keyed scalar reads as prose
            # so the model meets facts the way a page states them.
            parts.append(text if text[-1] in ".!?" else f"{key.replace('_', ' ')}: {text}.")
        return " ".join(parts)

    def snippet(self, payload: dict[str, Any], fillers: list[dict[str, Any]], rng: random.Random) -> str:
        """``payload`` rendered inside same-topic noise, at a varying position."""
        answer = self.render(payload)
        count = rng.randint(self._min_fillers, self._max_fillers)
        noise = [rendered for filler in fillers[:count] if (rendered := self.render(filler))]
        pieces = [*noise]
        pieces.insert(rng.randrange(len(pieces) + 1), answer)
        return f"{rng.choice(DATE_PREFIXES)} " + " ".join(pieces)

    def noise_snippet(self, fillers: list[dict[str, Any]], rng: random.Random) -> str:
        """A snippet with no answer in it, for the entries that are distractors."""
        rendered = [text for filler in fillers if (text := self.render(filler))]
        if not rendered:
            return f"{rng.choice(DATE_PREFIXES)} No further details are available."
        return f"{rng.choice(DATE_PREFIXES)} " + " ".join(rendered)

    @staticmethod
    def _stringify(value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ", ".join(filter(None, (SnippetShaper._stringify(item) for item in value)))
        if isinstance(value, dict):
            return "; ".join(
                f"{key.replace('_', ' ')} {SnippetShaper._stringify(item)}" for key, item in value.items() if item
            )
        return ""
