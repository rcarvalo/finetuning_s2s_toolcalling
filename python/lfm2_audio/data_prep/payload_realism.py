"""``PayloadRealism`` — rend les payloads d'outil aussi bruités que le réel.

Les payloads de la Phase B contiennent **toujours** la réponse, seule et
propre : ``{"results": "The Antarctic Polar Desert is the largest desert."}``,
``{"email": "sarah.chen@example.com"}``. Le réel renvoie plusieurs entrées
hétérogènes dont la réponse est parfois absente. Le modèle a donc appris
« le payload contient la réponse, reformule-la » — d'où les deux échecs
mesurés sur v3 : réciter la première entrée venue, et **inventer** ce qui
manque (« Sarah's email address is sarah.johnson@example.com » alors que la
table ne contient aucune colonne e-mail).

Le corpus utilise plus de 200 formes de payload (chaînes, nombres, listes,
dicts). On ne cherche donc pas « le fait » dans ces formes : le payload
entier **est** l'entrée qui répond, et les distracteurs sont les payloads des
*autres* dialogues — plausibles, du bon domaine, et gratuits (aucun LLM).

Deux transformations :

1. **Bruit** : le payload devient une entrée parmi N, à une position variable.
   La réponse assistant ne change pas — donc aucun ré-enregistrement audio.
2. **Absence** (``miss_ratio``) : le payload vrai est retiré. La réponse
   attendue devient un aveu d'échec, à ré-enregistrer : c'est le comportement
   que rien n'enseignait.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

MISS_ANSWERS = (
    "I couldn't find that in the results. Want me to search differently?",
    "The results don't cover that — should I try another search?",
    "I don't see that in what came back. Do you want me to look again?",
    "That isn't in the results I got. I can retry with different terms.",
    "Nothing in the results answers that. Want me to try another angle?",
)
"""Aveux d'échec. Plusieurs formulations : une phrase unique serait apprise
comme un réflexe déclenché par la forme de la question, pas par l'absence
d'information dans le payload."""

WEB_SEARCH = "web_search"


@dataclass(frozen=True, slots=True)
class PayloadRealism:
    """Réécrit les tours ``tool`` d'un corpus de dialogues."""

    seed: int = 7
    miss_ratio: float = 0.15
    min_results: int = 3
    max_results: int = 5

    def apply(self, dialogues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Retourne (dialogues transformés, nombre de cas « réponse absente »).

        Les dialogues sans tour ``tool`` exploitable traversent inchangés.
        """
        rng = random.Random(self.seed)
        pool = [payload for dialogue in dialogues if (payload := self._payload_of(dialogue)) is not None]
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
    def _tool_name(dialogue: dict[str, Any]) -> str:
        for turn in dialogue.get("turns", []):
            calls = turn.get("tool_calls")
            if calls:
                return str(calls[0].get("name", WEB_SEARCH))
        return WEB_SEARCH

    def _entries(
        self, payload: dict[str, Any], pool: list[dict[str, Any]], rng: random.Random, *, drop: bool
    ) -> list[dict[str, Any]]:
        """Le payload vrai noyé parmi des distracteurs — ou seulement ceux-ci."""
        count = rng.randint(self.min_results, self.max_results)
        distractors = [other for other in rng.sample(pool, count + 1) if other is not payload]
        entries = distractors[: count if drop else count - 1]
        if not drop:
            entries.insert(rng.randrange(len(entries) + 1), payload)
        return entries

    def _rewrite(
        self,
        dialogue: dict[str, Any],
        payload: dict[str, Any],
        pool: list[dict[str, Any]],
        rng: random.Random,
        *,
        drop: bool,
    ) -> dict[str, Any]:
        entries = self._entries(payload, pool, rng, drop=drop)
        # Un moteur web renvoie des documents, une base des lignes : garder les
        # deux formes distinctes évite d'apprendre au modèle qu'un résultat SQL
        # ressemble à une page web.
        if self._tool_name(dialogue) == WEB_SEARCH:
            content: dict[str, Any] = {
                "results": [
                    {"title": f"Result {index + 1}", "url": f"https://example.org/r/{rng.randrange(10_000, 99_999)}"}
                    | entry
                    for index, entry in enumerate(entries)
                ]
            }
        else:
            content = {"rows": list(entries)}

        turns = [dict(turn) for turn in dialogue["turns"]]
        for turn in turns:
            if turn.get("role") == "tool":
                turn["content"] = content
            elif drop and turn.get("role") == "assistant" and turn.get("text") and not turn.get("tool_calls"):
                turn["text"] = rng.choice(MISS_ANSWERS)
                # La réponse a changé : garder l'ancien WAV apprendrait au modèle
                # un texte qui ne correspond pas à ce qui est prononcé.
                turn.pop("audio", None)

        # Le drapeau va dans ``meta`` (extra="allow"), pas à la racine du
        # dialogue : ``Dialogue`` est en extra="forbid" et rejetterait le
        # dataset entier au packing.
        meta = dict(dialogue.get("meta") or {}) | {"answer_absent": drop}
        return dict(dialogue) | {"turns": turns, "meta": meta}
