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
from pathlib import Path
from typing import Any

import runpod
from pydantic import ValidationError

from lfm2_audio.core.prompt import DEFAULT_SYSTEM
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.inference_config import EngineConfig
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


def _engine_config() -> EngineConfig | None:
    """Engine settings from LFM2_DEPLOY_CONFIG, when the image ships one.

    The per-stage deploy YAML is the difference between ~750 ms and ~300 ms of
    TTFA on the vLLM backend (stage-0 CUDA graphs, short initial codec chunk).
    The package's own default only resolves inside a repo checkout, so the
    worker must be pointed at the baked file explicitly. A missing file fails
    loudly at boot — a silently eager engine would look like "vLLM is slow".
    """
    deploy = os.environ.get("LFM2_DEPLOY_CONFIG")
    if not deploy:
        return None
    return EngineConfig(deploy_config=Path(deploy))


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
        engine=_engine_config(),
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


def _warm_first_turn(model: LFM2Audio) -> None:
    """Absorb the first-generation warmup at boot, not on the first job.

    Measured on an L4: the engine reports ready, then the FIRST generation
    still pays ~6.5s of lazy CUDA-graph/JIT warmup (TTFA 6.8s vs 0.28s steady
    state). On serverless that cost would land on the user's first question
    after every cold start, so we spend it here — before the worker starts
    taking jobs. Best-effort: a warmup failure must not kill a worker whose
    real jobs might still succeed.
    """
    try:
        for _ in model.stream(text="Hi.", max_tokens=16):
            pass
        logger.info("boot warmup done; first job will see steady-state TTFA")
    except Exception:
        logger.warning("boot warmup failed; first job pays warmup", exc_info=True)
    finally:
        model.reset()


if __name__ == "__main__":
    # Échec de chargement = échec du worker au boot, pas au 1er job.
    _warm_first_turn(get_model())
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
