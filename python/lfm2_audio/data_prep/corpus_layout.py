"""Layout of the FR corpus repository: one folder per brick, one manifest schema.

The corpus is stored as four separate bricks rather than one pooled dataset, and
that separation is a working tool, not filing tidiness. Sizing is decided at
gates: when a gate fails, the useful question is *which brick* is short — the
assistant's speech, the user's speech, the dialogue content, or the English
preservation — and a pooled corpus cannot answer it. Separate bricks also let a
single one be topped up and re-pushed without touching the rest.

  A_assistant_speech  what the model must learn to SAY: one voice, clean text
                      alignment, conversational register
  B_user_speech       what it must learn to HEAR: maximum speaker diversity
  C_dialogues         the conversational content, including code-switch
  D_english           the English share that keeps the frozen anchors intact

Every brick carries the same manifest schema, so the mixer reads one format.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.jsonl"
AUDIO_DIR = "audio"


@dataclass(frozen=True, slots=True)
class Brick:
    """One folder of the corpus repository."""

    key: str
    folder: str
    purpose: str
    requirement: str


BRICKS: tuple[Brick, ...] = (
    Brick(
        key="A",
        folder="A_assistant_speech",
        purpose="parole que le modèle doit produire",
        requirement="une seule identité vocale, texte strictement aligné, registre parlé",
    ),
    Brick(
        key="B",
        folder="B_user_speech",
        purpose="parole que le modèle doit comprendre",
        requirement="diversité maximale de locuteurs, d'accents et de conditions",
    ),
    Brick(
        key="C",
        folder="C_dialogues",
        purpose="contenu conversationnel, dont le code-switch",
        requirement="tours alignés, langue attendue explicite pour chaque tour",
    ),
    Brick(
        key="D",
        folder="D_english",
        purpose="part anglaise qui protège les ancres gelées",
        requirement="même schéma, lang=en, issue des corpus EN validés",
    ),
    Brick(
        key="E",
        folder="E_long_form",
        purpose="parole longue que le modèle doit tenir sans dériver (6 à 20 s)",
        requirement="même voix que A, phrases longues lues, texte vérifié par ré-écoute",
    ),
)

BRICKS_BY_KEY = {brick.key: brick for brick in BRICKS}
BRICKS_BY_FOLDER = {brick.folder: brick for brick in BRICKS}


class CorpusError(Exception):
    """Manifest or layout does not satisfy the corpus contract."""


@dataclass
class CorpusEntry:
    """One clip, in the single schema every brick shares.

    ``text`` is the transcript the audio must match — for brick A it is what the
    assistant says and what will be interleaved, so an entry whose audio drifts
    from it teaches drift. ``voxtral_wer`` records the independent re-listen
    that decided the clip was kept.
    """

    id: str
    audio: str
    text: str
    lang: str
    duration_s: float
    role: str = "assistant"
    speaker: str = ""
    source: str = ""
    voxtral_wer: float | None = None
    voxtral_cer: float | None = None
    utmos: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id or not self.audio or not self.text.strip():
            raise CorpusError(f"entrée incomplète : {self.id!r}")
        if self.lang not in {"fr", "en"}:
            raise CorpusError(f"langue inattendue pour {self.id!r} : {self.lang!r}")
        if self.role not in {"user", "assistant"}:
            raise CorpusError(f"rôle inattendu pour {self.id!r} : {self.role!r}")
        if self.duration_s <= 0:
            raise CorpusError(f"durée invalide pour {self.id!r} : {self.duration_s}")

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value not in (None, "", {})}


def write_manifest(entries: Iterable[CorpusEntry], path: Path) -> int:
    """Write a brick's manifest, validating every entry first.

    Validation happens at the boundary — once a bad row is in the corpus it is
    silently trained on.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            entry.validate()
            handle.write(json.dumps(entry.as_json(), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_manifest(path: Path) -> Iterator[CorpusEntry]:
    """Entries of a brick manifest."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        known = {f for f in CorpusEntry.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in payload.items() if k not in known}
        yield CorpusEntry(**{k: v for k, v in payload.items() if k in known}, extra=extra)


def brick_readme(brick: Brick, entries: int, hours: float) -> str:
    """The README that travels with a brick, so its contract is where it lives."""
    return (
        f"# {brick.folder}\n\n"
        f"**Rôle** : {brick.purpose}\n\n"
        f"**Exigence** : {brick.requirement}\n\n"
        f"{entries} clips · {hours:.1f} h\n\n"
        f"Schéma : `{MANIFEST_NAME}` (une entrée par ligne) + `{AUDIO_DIR}/`.\n"
        "Champs : `id`, `audio`, `text`, `lang`, `duration_s`, `role`, `speaker`,\n"
        "`source`, `voxtral_wer`, `utmos`.\n\n"
        "`voxtral_wer` est l'écart entre `text` et une ré-écoute Voxtral indépendante :\n"
        "c'est le filtre qui décide si un clip entre dans le corpus ; `voxtral_cer` est\n"
        "le même écart au caractère, qui rattrape un nom propre mal entendu.\n"
    )
