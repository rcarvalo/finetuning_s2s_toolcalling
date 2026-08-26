"""``PayloadRealism`` — rend les payloads d'outil aussi bruités que le réel.

Les payloads de la Phase B contiennent **toujours** la réponse, seule et
propre : ``{"results": "The Antarctic Polar Desert is the largest desert."}``.
Le réel renvoie plusieurs entrées hétérogènes dont la réponse est parfois
absente. Le modèle a donc appris « le payload contient la réponse, reformule-la ».

Le corpus utilise plus de 200 formes de payload : on ne cherche donc pas « le
fait » dans ces formes. Le payload entier **est** l'entrée qui répond, et les
distracteurs sont les payloads des *autres* dialogues.

Trois transformations, chacune visant un échec mesuré :

1. **Bruit du bon domaine** — distracteurs choisis parmi les questions
   proches (:class:`NearDistractors`), et non au hasard : en v4 le sujet seul
   suffisait à isoler la bonne entrée, raccourci qui ne transfère pas.
2. **Prose** — les résultats web deviennent des snippets verbeux
   (:class:`SnippetShaper`) : le réel noie la réponse dans 400 caractères,
   là où v4 la livrait comme un champ.
3. **Absence** (``miss_ratio``) — la réponse attendue devient un aveu d'échec
   *situé* (:class:`ContextualMiss`), à ré-enregistrer.

``miss_ratio`` retombe de 0,15 à 0,08 : à 15 %, v4 a appris à refuser trop
souvent (honnêteté 3,15, sous la porte de 4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from lfm2_audio.data_prep.contextual_miss import ContextualMiss
from lfm2_audio.data_prep.near_distractors import NearDistractors
from lfm2_audio.data_prep.snippet_shaper import SnippetShaper

WEB_SEARCH = "web_search"


@dataclass(frozen=True, slots=True)
class PayloadRealism:
    """Réécrit les tours ``tool`` d'un corpus de dialogues."""

    seed: int = 7
    miss_ratio: float = 0.08
    min_results: int = 3
    max_results: int = 5
    shaper: SnippetShaper = field(default_factory=SnippetShaper)
    misses: ContextualMiss = field(default_factory=ContextualMiss)

    def apply(self, dialogues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Retourne (dialogues transformés, nombre de cas « réponse absente »).

        Les dialogues sans tour ``tool`` exploitable traversent inchangés.
        """
        rng = random.Random(self.seed)
        pool = NearDistractors(
            [
                (payload, self._question_of(dialogue))
                for dialogue in dialogues
                if (payload := self._payload_of(dialogue)) is not None
            ]
        )
        if len(pool) <= self.max_results:
            raise ValueError(f"pool de distracteurs trop petit : {len(pool)}")

        out: list[dict[str, Any]] = []
        misses = 0
        for dialogue in dialogues:
            payload = self._payload_of(dialogue)
            if payload is None:
                out.append(dialogue)
                continue
            drop = rng.random() < self.miss_ratio
            misses += drop
            out.append(self._rewrite(dialogue, payload, pool, rng, drop=drop))
        self._validate(out)
        return out, misses

    @staticmethod
    def _validate(dialogues: list[dict[str, Any]]) -> None:
        """Échouer ici plutôt qu'au packing, sur une VM, une heure plus tard.

        Un seul dialogue non conforme fait rejeter le dataset ENTIER par
        ``pack_sft`` — c'est ce qu'un drapeau posé à la racine avait provoqué.
        """
        from lfm2_audio.ds.dialogue import parse_dialogue

        for dialogue in dialogues:
            parse_dialogue(dialogue)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _payload_of(dialogue: dict[str, Any]) -> dict[str, Any] | None:
        """Le contenu du tour ``tool``, quelle que soit sa forme."""
        for turn in dialogue.get("turns", []):
            if turn.get("role") == "tool":
                content = turn.get("content")
                return content if isinstance(content, dict) and content else None
        return None

    @staticmethod
    def _question_of(dialogue: dict[str, Any]) -> str:
        """Le texte de la question posée — ce dont le dialogue parle.

        Repli sur les arguments du tool call quand le tour user n'a pas de
        texte (dialogue purement audio) : la requête émise dit le sujet.
        """
        for turn in dialogue.get("turns", []):
            if turn.get("role") == "user" and turn.get("text"):
                return str(turn["text"])
        for turn in dialogue.get("turns", []):
            for call in turn.get("tool_calls") or []:
                if values := [str(v) for v in (call.get("arguments") or {}).values()]:
                    return " ".join(values)
        return ""

    @staticmethod
    def _tool_name(dialogue: dict[str, Any]) -> str:
        for turn in dialogue.get("turns", []):
            calls = turn.get("tool_calls")
            if calls:
                return str(calls[0].get("name", WEB_SEARCH))
        return WEB_SEARCH

    def _web_results(
        self,
        payload: dict[str, Any],
        distractors: list[dict[str, Any]],
        rng: random.Random,
        *,
        drop: bool,
    ) -> list[dict[str, Any]]:
        """Des documents : titre, url, et une prose où la réponse est noyée."""
        answering = None if drop else rng.randrange(len(distractors) + 1)
        results: list[dict[str, Any]] = []
        for index in range(len(distractors) + (0 if drop else 1)):
            others = [d for position, d in enumerate(distractors) if position != index]
            rng.shuffle(others)
            snippet = (
                self.shaper.snippet(payload, others, rng)
                if index == answering
                else self.shaper.noise_snippet(others[:1], rng)
            )
            results.append(
                {
                    "title": f"Result {index + 1}",
                    "url": f"https://example.org/r/{rng.randrange(10_000, 99_999)}",
                    "snippet": snippet,
                }
            )
        return results

    def _rewrite(
        self,
        dialogue: dict[str, Any],
        payload: dict[str, Any],
        pool: NearDistractors,
        rng: random.Random,
        *,
        drop: bool,
    ) -> dict[str, Any]:
        question = self._question_of(dialogue)
        count = rng.randint(self.min_results, self.max_results)
        distractors = pool.pick(question, count, rng, exclude=payload)

        # Un moteur web renvoie des documents, une base des lignes : garder les
        # deux formes distinctes évite d'apprendre au modèle qu'un résultat SQL
        # ressemble à une page web.
        if self._tool_name(dialogue) == WEB_SEARCH:
            content: dict[str, Any] = {"results": self._web_results(payload, distractors, rng, drop=drop)}
        else:
            rows = list(distractors)
            if not drop:
                rows.insert(rng.randrange(len(rows) + 1), payload)
            content = {"rows": rows}

        turns = [dict(turn) for turn in dialogue["turns"]]
        for turn in turns:
            if turn.get("role") == "tool":
                turn["content"] = content
            elif drop and turn.get("role") == "assistant" and turn.get("text") and not turn.get("tool_calls"):
                turn["text"] = self.misses.text(question, [pool.question_of(d) for d in distractors], rng)
                # La réponse a changé : garder l'ancien WAV apprendrait au modèle
                # un texte qui ne correspond pas à ce qui est prononcé.
                turn.pop("audio", None)

        # Le drapeau va dans ``meta`` (extra="allow"), pas à la racine du
        # dialogue : ``Dialogue`` est en extra="forbid" et rejetterait le
        # dataset entier au packing.
        meta = dict(dialogue.get("meta") or {}) | {"answer_absent": drop}
        return dict(dialogue) | {"turns": turns, "meta": meta}
