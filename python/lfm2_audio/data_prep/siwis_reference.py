"""Find a usable SIWIS clip to clone the assistant voice from.

SIWIS is studio French from a professional speaker under CC-BY-4.0, which makes
it the voice identity of brick A. Cloning needs a clip **and its transcript**,
and getting that pair out of this distribution is not obvious:

* the ``labs/`` directory holds 300 phonetic labels whose ids do **not**
  intersect the 314 wavs at all — pairing on them silently yields nothing, which
  cost two bake-off runs their main candidate;
* the real transcripts live under ``text/``, and only 36 of them match a wav;
* one of those matches is a whole chapter, far too long to serve as a reference.

So the pair is resolved by intersecting wav and *text* ids and then keeping a
clip whose duration is in the cloning range, rather than by trusting a name.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SIWIS_REPO = "Aviv-anthonnyolime/SIWIS_French_Speech_Synthesis_Database"
MIN_SECONDS = 3.0
MAX_SECONDS = 15.0
"""Qwen advertises cloning from three seconds; past ~15 s the extra audio only
slows the prompt build without improving the voice."""


class SiwisError(Exception):
    """No usable reference clip could be resolved."""


@dataclass(frozen=True, slots=True)
class VoiceReference:
    """A clip and the text it says — both required to clone a voice."""

    wav_path: Path
    text: str
    stem: str

    @property
    def audio_bytes(self) -> bytes:
        return self.wav_path.read_bytes()


def pair_stems(files: Iterable[str]) -> list[tuple[str, str, str]]:
    """``(stem, wav_path, text_path)`` for every clip that has a transcript.

    Transcripts are taken from ``text/`` only: ``labs/`` holds phonetic labels
    for a different set of utterances, and matching against them returns an
    empty intersection that looks like "nothing available".
    """
    listed = list(files)
    wavs = {Path(name).stem: name for name in listed if name.endswith(".wav")}
    texts = {Path(name).stem: name for name in listed if name.endswith(".txt") and name.startswith("text/")}
    return [(stem, wavs[stem], texts[stem]) for stem in sorted(set(wavs) & set(texts))]


def resolve_reference(
    *,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    repo: str = SIWIS_REPO,
) -> VoiceReference:
    """Download and return the first SIWIS clip usable as a cloning reference."""
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    pairs = pair_stems(HfApi().list_repo_files(repo, repo_type="dataset"))
    if not pairs:
        raise SiwisError(f"aucune paire wav/texte dans {repo}")

    for stem, wav_name, text_name in pairs:
        wav = Path(hf_hub_download(repo, wav_name, repo_type="dataset"))
        duration = sf.info(str(wav)).duration
        if not min_seconds <= duration <= max_seconds:
            logger.info("%s écarté : %.1f s hors de [%.0f, %.0f]", stem, duration, min_seconds, max_seconds)
            continue
        text_path = Path(hf_hub_download(repo, text_name, repo_type="dataset"))
        text = " ".join(text_path.read_text(encoding="utf-8", errors="replace").split())
        logger.info("référence SIWIS : %s (%.1f s) — %s", stem, duration, text[:60])
        return VoiceReference(wav_path=wav, text=text, stem=stem)

    raise SiwisError(f"{len(pairs)} paires trouvées, aucune entre {min_seconds} et {max_seconds} s")
