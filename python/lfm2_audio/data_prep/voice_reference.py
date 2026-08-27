"""Where the assistant's voice is cloned from.

Two sources were compared by ear on the same ten receptionist sentences, same
engine, and the verdict was unambiguous:

* **``dialogue``** — a clip from ``french-dialogue-tts-1000h``. Synthetic in
  origin, but genuinely *spoken* register ("Ah, tu sais, pour un vrai coq au
  vin…"). **Retained.**
* ``siwis`` — real studio human, but SIWIS is a speech *synthesis* database and
  every register in it is READ. The neutral clone was judged robotic, the
  emphatic one theatrical.

So register beats authenticity here, which is not what one would guess and is
exactly why it was measured. Cloning carries the speaking style over with the
timbre: cloning someone reading a book yields an assistant that reads.

None of this was visible to the metrics — UTMOS put the emphatic clone at 4.02
and the neutral at 3.99 while a listener separated them instantly. Naturalness
scores do not know what register the task calls for.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from lfm2_audio.data_prep.siwis_reference import (
    MAX_SECONDS,
    MIN_SECONDS,
    SiwisError,
    VoiceReference,
    resolve_reference,
)

logger = logging.getLogger(__name__)

DIALOGUE_REPO = "Rcarvalo/french-dialogue-tts-1000h"
MIN_TEXT_CHARS = 60
MAX_TEXT_CHARS = 220
"""A reference wants a few seconds of continuous speech: too short carries no
prosody, too long only slows the prompt build."""

DEFAULT_SOURCE = "dialogue"


def resolve_dialogue_reference(
    *,
    repo: str = DIALOGUE_REPO,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    scan: int = 200,
) -> VoiceReference:
    """A conversational clip and its text, from the dialogue corpus.

    Chosen by duration and text length rather than by position: the corpus
    mixes one-liners with long monologues, and the first entry is not
    necessarily usable.
    """
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    manifest = Path(hf_hub_download(repo, "metadata.jsonl", repo_type="dataset"))
    for line in manifest.read_text(encoding="utf-8").splitlines()[:scan]:
        if not line.strip():
            continue
        row = json.loads(line)
        text = " ".join(str(row.get("text", "")).split())
        if not MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS:
            continue
        wav = Path(hf_hub_download(repo, row["file_name"], repo_type="dataset"))
        duration = sf.info(str(wav)).duration
        if not min_seconds <= duration <= max_seconds:
            continue
        stem = Path(row["file_name"]).stem
        logger.info("référence dialogue : %s (%.1f s) — %s", stem, duration, text[:60])
        return VoiceReference(wav_path=wav, text=text, stem=stem)

    raise SiwisError(f"aucun clip utilisable dans les {scan} premières entrées de {repo}")


def resolve_voice_reference(source: str = DEFAULT_SOURCE) -> VoiceReference:
    """The clip to clone the assistant voice from, by source name."""
    if source == "siwis":
        return resolve_reference()
    if source == "dialogue":
        return resolve_dialogue_reference()
    raise SiwisError(f"source de voix inconnue : {source!r} (connues : siwis, dialogue)")
