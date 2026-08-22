"""Handler serverless RunPod — un job = un tour speech-to-speech.

Générateur : chaque chunk audio est *yield* dès sa sortie du modèle, donc
``/stream/{id}`` sert le streaming et — grâce à ``return_aggregate_stream`` —
``/runsync`` reçoit la même séquence agrégée. Un seul handler pour les deux
modes ; le contrat d'événements est celui de ``lfm2_audio.remote.client``.

Le modèle est chargé UNE fois par worker (import module), pas par job : c'est
tout l'intérêt de FlashBoot. v1 stateless : chaque job repart d'un contexte
neuf (le multi-tour côté endpoint viendra avec les sessions).

Variables d'environnement :
    LFM2_CHECKPOINT  — repo HF ou chemin baké dans l'image (requis)
    LFM2_BACKEND     — "vllm" | "liquid" | "auto" (défaut : auto)
    LFM2_SYSTEM      — prompt système (défaut : DEFAULT_SYSTEM du paquet)
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator
from typing import Any

import runpod

from lfm2_audio.core.prompt import DEFAULT_SYSTEM
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote.codec import waveform_from_wav_b64, waveform_to_wav_b64
from lfm2_audio.serving.model import LFM2Audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lfm2_audio.handler")


@functools.cache
def get_model() -> LFM2Audio:
    """Charge le modèle au premier appel puis le réutilise (un par worker)."""
    checkpoint = os.environ["LFM2_CHECKPOINT"]
    backend = os.environ.get("LFM2_BACKEND", "auto")
    logger.info("chargement de %s (backend=%s)", checkpoint, backend)
    return LFM2Audio.from_pretrained(
        checkpoint,
        backend=backend,
        system=os.environ.get("LFM2_SYSTEM", DEFAULT_SYSTEM),
    )


def handler(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """audio/texte en entrée → chunks audio puis événement final."""
    job_input: dict[str, Any] = job.get("input") or {}
    text: str | None = job_input.get("text")
    audio_b64: str | None = job_input.get("audio_b64")
    if text is None and audio_b64 is None:
        yield {"kind": "error", "error": "input.text ou input.audio_b64 requis"}
        return

    model = get_model()
    model.reset()  # v1 stateless : pas d'historique entre jobs
    audio: Waveform | None = waveform_from_wav_b64(audio_b64) if audio_b64 else None

    for chunk in model.stream(text=text, audio=audio, max_tokens=job_input.get("max_tokens")):
        if not chunk.is_empty:
            yield {
                "kind": "audio",
                "audio_b64": waveform_to_wav_b64(chunk),
                "sample_rate": chunk.sample_rate,
            }

    reply = model.last_reply
    yield {
        "kind": "final",
        "text": reply.text if reply else "",
        "raw_text": reply.raw_text if reply else "",
        "metrics": reply.metrics.as_dict() if reply else {},
    }


if __name__ == "__main__":
    get_model()  # échec de chargement = échec du worker au boot, pas au 1er job
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
