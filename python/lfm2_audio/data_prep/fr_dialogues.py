"""Generation of brick C: French conversational dialogues, including code-switch.

Separate from :mod:`synth_dialogues`, which generates tool-calling cases and
carries a tool registry, argument verification and a target taxonomy. Bending it
into conversational French would drag all of that along; what is worth reusing —
the contamination filter and the JSON extraction — is imported instead.

What brick C must supply, from the 0B baseline:

* **French conversation**, spoken register. The model already writes clean
  French in text-only mode, so the corpus is not teaching vocabulary; it is
  teaching French that survives being spoken while audio is interleaved. Short,
  said-out-loud turns are therefore worth more than well-written paragraphs.
* **Code-switch**, and this is the priority. A system prompt already lifts plain
  French mirroring from 40 % to 75 % for free, but it plateaus at 84 % on
  code-switch — the one axis training has to win on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MIN_TURNS = 2
MAX_UTTERANCE_CHARS = 320

_WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœ']+", flags=re.IGNORECASE)


class DialogueError(Exception):
    """A generated dialogue does not satisfy the brick C contract."""


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """One turn, with the language the reply is expected to be in."""

    role: str
    text: str
    lang: str


@dataclass(frozen=True, slots=True)
class FrDialogue:
    """A generated conversation, ready for TTS and for the corpus manifest."""

    dialogue_id: str
    turns: tuple[DialogueTurn, ...]
    kind: str = "fr"
    topic: str = ""

    @property
    def expected_lang(self) -> str:
        """The language the assistant must answer in — the LAST user turn's.

        Not the first: in a code-switch dialogue the user starts in one language
        and moves to another, and answering in the language they opened with is
        exactly the failure being trained away.
        """
        user_turns = [turn for turn in self.turns if turn.role == "user"]
        if not user_turns:
            raise DialogueError(f"{self.dialogue_id} : aucun tour utilisateur")
        return user_turns[-1].lang

    def validate(self) -> None:
        if len(self.turns) < MIN_TURNS:
            raise DialogueError(f"{self.dialogue_id} : {len(self.turns)} tour(s), minimum {MIN_TURNS}")
        if self.turns[0].role != "user":
            raise DialogueError(f"{self.dialogue_id} : doit commencer par un tour utilisateur")
        for turn in self.turns:
            if turn.role not in {"user", "assistant"}:
                raise DialogueError(f"{self.dialogue_id} : rôle inattendu {turn.role!r}")
            if not turn.text.strip():
                raise DialogueError(f"{self.dialogue_id} : tour vide")
            if len(turn.text) > MAX_UTTERANCE_CHARS:
                raise DialogueError(
                    f"{self.dialogue_id} : tour de {len(turn.text)} caractères, "
                    f"au-delà de {MAX_UTTERANCE_CHARS} ce n'est plus de l'oral"
                )
        assistant = [turn for turn in self.turns if turn.role == "assistant"]
        if not assistant:
            raise DialogueError(f"{self.dialogue_id} : aucune réponse d'assistant")
        if assistant[-1].lang != self.expected_lang:
            raise DialogueError(
                f"{self.dialogue_id} : l'assistant répond en {assistant[-1].lang}, "
                f"l'utilisateur a fini en {self.expected_lang} — c'est le défaut qu'on entraîne à corriger"
            )

    def as_case(self) -> dict[str, Any]:
        """The shared dialogue JSONL shape, so TTS and packing read one format."""
        return {
            "id": self.dialogue_id,
            "tools": [],
            "meta": {
                "lang": self.expected_lang,
                "expected_lang": self.expected_lang,
                "kind": self.kind,
                "topic": self.topic,
            },
            "turns": [{"role": turn.role, "text": turn.text, "lang": turn.lang} for turn in self.turns],
        }


FR_PROMPT = """Génère {count} conversations courtes en FRANÇAIS entre un visiteur
et l'assistant vocal d'accueil d'une entreprise.

Contraintes impératives :
- Registre ORAL. Ce sera lu à voix haute : phrases courtes, contractions naturelles, pas de style écrit.
- 2 à 4 tours au total, en commençant par le visiteur.
- Aucune liste à puces, aucun markdown, aucune énumération numérotée.
- Chaque tour fait moins de 300 caractères.
- Thème : {topic}
- Varie les situations, les niveaux de politesse et les registres.
- Évite de commencer chaque conversation de la même façon.

Réponds UNIQUEMENT par un tableau JSON, chaque élément ayant la forme :
{{"topic": "...", "turns": [{{"role": "user", "text": "..."}}, {{"role": "assistant", "text": "..."}}]}}"""

CODE_SWITCH_PROMPT = """Génère {count} conversations où l'utilisateur CHANGE DE
LANGUE en cours d'échange, avec l'assistant vocal d'accueil d'une entreprise.

C'est le cas le plus important : l'assistant doit répondre dans la langue du
DERNIER tour de l'utilisateur, pas dans celle qu'il utilisait au début.

Contraintes impératives :
- Registre ORAL, phrases courtes, ce sera lu à voix haute.
- 3 à 4 tours, en commençant par le visiteur.
- L'utilisateur commence dans une langue ({first}) et bascule vers l'autre
  ({second}) — soit au tour suivant, soit à l'intérieur d'une même phrase.
- L'assistant répond CHAQUE FOIS dans la langue du tour utilisateur qui précède.
- Chaque tour fait moins de 300 caractères. Aucun markdown.
- Thème : {topic}

Réponds UNIQUEMENT par un tableau JSON, chaque élément ayant la forme :
{{"topic": "...", "turns": [
  {{"role": "user", "text": "...", "lang": "fr"}},
  {{"role": "assistant", "text": "...", "lang": "fr"}}
]}}
en indiquant pour chaque tour la langue ("fr" ou "en") de son texte."""

TOPICS_FR = (
    "accueil d'un visiteur qui a rendez-vous",
    "visiteur sans rendez-vous qui cherche quelqu'un",
    "demande du code wifi invité",
    "orientation dans le bâtiment",
    "livraison ou colis à déposer",
    "question sur les horaires ou l'accès",
    "petite conversation en attendant",
    "problème pratique : parking, badge, ascenseur",
    "demande d'aide sur un imprévu",
    "prise de congé et remerciements",
)


def build_fr_prompt(count: int, topic: str) -> str:
    return FR_PROMPT.format(count=count, topic=topic)


def build_code_switch_prompt(count: int, topic: str, *, first: str = "français", second: str = "anglais") -> str:
    return CODE_SWITCH_PROMPT.format(count=count, topic=topic, first=first, second=second)


def detect_turn_language(text: str) -> str:
    """Language of one turn, reusing the scorer's FR/EN discriminator.

    Generated ``lang`` fields are taken as a hint, never as truth: the generator
    is the thing being checked, and a mislabelled turn would teach the model to
    answer in the wrong language — precisely the defect brick C exists to fix.
    """
    from lfm2_audio.scorer.text.lang_match import detect_language

    return detect_language(text) or "fr"


def parse_dialogues(payload: list[dict[str, Any]], *, prefix: str, kind: str, start: int = 0) -> list[FrDialogue]:
    """Turn a model's JSON array into validated dialogues, dropping bad ones."""
    dialogues: list[FrDialogue] = []
    for offset, item in enumerate(payload):
        raw_turns = item.get("turns") or []
        turns = tuple(
            DialogueTurn(
                role=str(turn.get("role", "")),
                text=str(turn.get("text", "")).strip(),
                # The declared language is cross-checked against the text itself.
                lang=detect_turn_language(str(turn.get("text", ""))),
            )
            for turn in raw_turns
        )
        dialogue = FrDialogue(
            dialogue_id=f"{prefix}_{start + offset:04d}",
            turns=turns,
            kind=kind,
            topic=str(item.get("topic", "")),
        )
        try:
            dialogue.validate()
        except DialogueError:
            continue
        dialogues.append(dialogue)
    return dialogues


def code_switch_rate(dialogues: list[FrDialogue]) -> float:
    """Share of dialogues where the user actually changes language.

    The generator is asked for code-switch; whether it delivered is measured,
    because a batch that merely looks bilingual would train nothing.
    """
    if not dialogues:
        return 0.0
    switched = 0
    for dialogue in dialogues:
        languages = [turn.lang for turn in dialogue.turns if turn.role == "user"]
        if len(set(languages)) > 1:
            switched += 1
    return switched / len(dialogues)
