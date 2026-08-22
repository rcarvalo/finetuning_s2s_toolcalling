"""Handler serverless RunPod — un job = un tour speech-to-speech.

Générateur : chaque chunk audio est *yield* dès sa sortie du modèle, donc
``/stream/{id}`` sert le streaming et — grâce à ``return_aggregate_stream`` —
``/runsync`` reçoit la même séquence agrégée. Un seul handler pour les deux
modes ; le contrat d'événements est celui de
:mod:`lfm2_audio.remote.protocol`, partagé avec le client.

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
from pydantic import ValidationError

from lfm2_audio.core.prompt import DEFAULT_SYSTEM
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.remote.protocol import (
    AudioChunkEvent,
    ErrorEvent,
    FinalEvent,
    TurnMetricsPayload,
    TurnRequest,
)
from lfm2_audio.remote.wav_base64 import waveform_from_wav_b64, waveform_to_wav_b64
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
    try:
        request = TurnRequest.model_validate(job.get("input") or {})
    except ValidationError as exc:
        yield ErrorEvent(kind="error", error=str(exc.errors()[0]["msg"])).model_dump()
        return

    model = get_model()
    model.reset()  # v1 stateless : pas d'historique entre jobs
    audio: Waveform | None = waveform_from_wav_b64(request.audio_b64) if request.audio_b64 else None

    for chunk in model.stream(text=request.text, audio=audio, max_tokens=request.max_tokens):
        if not chunk.is_empty:
            yield AudioChunkEvent(
                audio_b64=waveform_to_wav_b64(chunk),
                sample_rate=chunk.sample_rate,
            ).model_dump()

    reply = model.last_reply
    yield FinalEvent(
        text=reply.text if reply else "",
        raw_text=reply.raw_text if reply else "",
        metrics=TurnMetricsPayload(**reply.metrics.as_dict()) if reply else TurnMetricsPayload(),
    ).model_dump()


if __name__ == "__main__":
    get_model()  # échec de chargement = échec du worker au boot, pas au 1er job
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
