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

REGISTER_PREFERENCE = ("neut", "expr", "emph", "chap")
"""SIWIS encodes the speaking style in the file name, and cloning carries the
style over with the timbre.

The first reference this project used was ``emph_book`` — an *emphatic book
reading*, Jules Verne declaimed — and the clone was judged "very robotic but
clean" by ear. It was: a theatrical reading voice put to work as a
receptionist. ``neut`` (neutral) comes first here for that reason, and the
metrics that scored the emphatic clone well never saw the problem, because
naturalness scores do not know what register the task calls for.
"""


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


def register_of(stem: str) -> str:
    """The speaking style SIWIS encodes in a file name (``neut``, ``emph``…)."""
    return stem.split("_", maxsplit=1)[0]


def candidates(
    files: Iterable[str], *, prefer: tuple[str, ...] = REGISTER_PREFERENCE
) -> list[tuple[str, str, str | None]]:
    """``(stem, wav, text_or_None)``, ordered by how well the register fits.

    Clips without a shipped transcript are kept rather than dropped: in this
    distribution **no neutral clip has one**, so filtering on transcripts alone
    silently forces the theatrical registers. The transcript for a neutral clip
    is produced by transcribing it — which is what the labelling engine is for.
    """
    listed = list(files)
    wavs = {Path(name).stem: name for name in listed if name.endswith(".wav")}
    texts = {Path(name).stem: name for name in listed if name.endswith(".txt") and name.startswith("text/")}

    def rank(stem: str) -> tuple[int, str]:
        register = register_of(stem)
        order = prefer.index(register) if register in prefer else len(prefer)
        return order, stem

    return [(stem, wavs[stem], texts.get(stem)) for stem in sorted(wavs, key=rank)]


def resolve_reference(
    *,
    min_seconds: float = MIN_SECONDS,
    max_seconds: float = MAX_SECONDS,
    repo: str = SIWIS_REPO,
    prefer: tuple[str, ...] = REGISTER_PREFERENCE,
    transcribe_missing: bool = True,
) -> VoiceReference:
    """The best-fitting SIWIS clip usable as a cloning reference.

    Register decides the order, duration filters, and a missing transcript is
    produced rather than treated as disqualifying.
    """
    import soundfile as sf
    from huggingface_hub import HfApi, hf_hub_download

    listed = candidates(HfApi().list_repo_files(repo, repo_type="dataset"), prefer=prefer)
    if not listed:
        raise SiwisError(f"aucun wav dans {repo}")

    for stem, wav_name, text_name in listed:
        if text_name is None and not transcribe_missing:
            continue
        wav = Path(hf_hub_download(repo, wav_name, repo_type="dataset"))
        duration = sf.info(str(wav)).duration
        if not min_seconds <= duration <= max_seconds:
            continue
        if text_name is not None:
            text = " ".join(
                Path(hf_hub_download(repo, text_name, repo_type="dataset"))
                .read_text(encoding="utf-8", errors="replace")
                .split()
            )
        else:
            text = _transcribe(wav)
            if not text:
                continue
        logger.info("référence SIWIS : %s [%s] (%.1f s) — %s", stem, register_of(stem), duration, text[:60])
        return VoiceReference(wav_path=wav, text=text, stem=stem)

    raise SiwisError(f"aucun clip entre {min_seconds} et {max_seconds} s parmi {len(listed)}")


def _transcribe(wav: Path) -> str:
    """Transcribe a reference clip when the corpus ships no text for it."""
    from lfm2_audio.ds.audio import Waveform
    from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber

    transcriber = FasterWhisperTranscriber(model_size="small", device="cpu", compute_type="int8")
    return " ".join(transcriber.transcribe(Waveform.from_file(str(wav)), language="fr").split())
