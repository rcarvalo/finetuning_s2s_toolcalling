"""``LiquidAudioClient`` — invoquer l'endpoint serverless RunPod comme un modèle local.

>>> from lfm2_audio.remote import LiquidAudioClient          # doctest: +SKIP
>>> llm = LiquidAudioClient("abc123")                        # doctest: +SKIP
>>> text, audio = llm.invoke(audio="question.wav")           # doctest: +SKIP
>>> for chunk in llm.invoke_stream(audio="question.wav"):    # doctest: +SKIP
...     play(chunk)                                          # doctest: +SKIP

Même contrat que :class:`~lfm2_audio.serving.model.LFM2Audio` (``invoke`` ↔
``reply``, ``invoke_stream`` ↔ ``stream``) mais au-dessus de HTTPS : le worker
tient le GPU, le client n'a besoin que de ``httpx`` (extra ``client``).
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy.typing as npt

from lfm2_audio.core.errors import RemoteInferenceError
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply, TurnMetrics
from lfm2_audio.remote.wav_base64 import waveform_from_wav_b64, waveform_to_wav_b64

type AudioInput = Waveform | str | Path | tuple[npt.ArrayLike, int]

_TERMINAL_FAILURES = frozenset({"FAILED", "TIMED_OUT", "CANCELLED"})
# /runsync rend la main au bout de ~90 s même si le job tourne encore : le read
# timeout httpx doit couvrir cette attente, le délai global est géré à part.
_HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=95.0, write=30.0, pool=10.0)


class LiquidAudioClient:
    """Client HTTP d'un endpoint serverless exécutant ``infra/handler.py``."""

    def __init__(
        self,
        endpoint_id: str,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.runpod.ai/v2",
        timeout_s: float = 600.0,
        poll_interval_s: float = 0.3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            message = "api_key manquante : passer api_key= ou définir RUNPOD_API_KEY"
            raise RemoteInferenceError(message)
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self._base = f"{base_url.rstrip('/')}/{endpoint_id}"
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {key}"},
            timeout=_HTTP_TIMEOUT,
            transport=transport,
        )
        self._last_reply: Reply | None = None

    # ------------------------------------------------------------------ #
    # API publique — miroir de LFM2Audio
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        *,
        text: str | None = None,
        audio: AudioInput | None = None,
        max_tokens: int | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Reply:
        """Un tour complet : bloque jusqu'à la réponse (texte + audio concaténé).

        ``history`` — past ``(role, text)`` turns replayed on the worker so a
        stateless endpoint can hold a conversation (the session lives here,
        on the client).
        """
        payload = {"input": self._build_input(text=text, audio=audio, max_tokens=max_tokens, history=history)}
        job = self._post("/runsync", payload)
        job = self._wait_terminal(job)
        reply = self._parse_events(self._output_events(job))
        self._last_reply = reply
        return reply

    def invoke_stream(
        self,
        *,
        text: str | None = None,
        audio: AudioInput | None = None,
        max_tokens: int | None = None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> Iterator[Waveform]:
        """Un tour en streaming : yield chaque chunk audio dès sa génération."""
        payload = {"input": self._build_input(text=text, audio=audio, max_tokens=max_tokens, history=history)}
        job = self._post("/run", payload)
        job_id = str(job["id"])
        deadline = time.monotonic() + self.timeout_s
        chunks: list[Waveform] = []
        final: dict[str, Any] = {}
        while True:
            page = self._get(f"/stream/{job_id}")
            for item in page.get("stream", []):
                event = item.get("output", {})
                if event.get("kind") == "audio":
                    chunk = waveform_from_wav_b64(event["audio_b64"])
                    chunks.append(chunk)
                    yield chunk
                elif event.get("kind") == "final":
                    final = event
            status = str(page.get("status", ""))
            if status == "COMPLETED":
                break
            self._raise_if_failed(status, page)
            if time.monotonic() > deadline:
                message = f"job {job_id} : délai de {self.timeout_s:.0f} s dépassé"
                raise RemoteInferenceError(message)
            time.sleep(self.poll_interval_s)
        self._last_reply = self._assemble(final, Waveform.concat(chunks))

    def health(self) -> dict[str, Any]:
        """État de l'endpoint (workers prêts, jobs en file)."""
        return self._get("/health")

    @property
    def last_reply(self) -> Reply | None:
        """Dernière réponse complète (renseignée aussi après ``invoke_stream``)."""
        return self._last_reply

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LiquidAudioClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Construction de la requête et parsing de la réponse
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_input(
        *,
        text: str | None,
        audio: AudioInput | None,
        max_tokens: int | None,
        history: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        if text is None and audio is None:
            message = "il faut au moins text= ou audio="
            raise RemoteInferenceError(message)
        job_input: dict[str, Any] = {}
        if text is not None:
            job_input["text"] = text
        if audio is not None:
            job_input["audio_b64"] = waveform_to_wav_b64(_coerce_audio(audio))
        if max_tokens is not None:
            job_input["max_tokens"] = max_tokens
        if history:
            job_input["history"] = [{"role": role, "text": turn} for role, turn in history]
        return job_input

    def _parse_events(self, events: list[dict[str, Any]]) -> Reply:
        chunks = [waveform_from_wav_b64(e["audio_b64"]) for e in events if e.get("kind") == "audio"]
        final = next((e for e in events if e.get("kind") == "final"), {})
        return self._assemble(final, Waveform.concat(chunks))

    @staticmethod
    def _assemble(final: dict[str, Any], audio: Waveform | None) -> Reply:
        metrics = final.get("metrics", {})
        return Reply(
            text=str(final.get("text", "")),
            audio=audio,
            raw_text=str(final.get("raw_text", "")),
            metrics=TurnMetrics(
                ttfa_s=metrics.get("ttfa_s"),
                total_s=float(metrics.get("total_s", 0.0)),
                audio_frames=int(metrics.get("audio_frames", 0)),
            ),
        )

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _wait_terminal(self, job: dict[str, Any]) -> dict[str, Any]:
        """Suit un job jusqu'à son état final (``/runsync`` lâche prise à ~90 s)."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            status = str(job.get("status", ""))
            if status == "COMPLETED":
                return job
            self._raise_if_failed(status, job)
            if time.monotonic() > deadline:
                message = f"job {job.get('id')} : délai de {self.timeout_s:.0f} s dépassé"
                raise RemoteInferenceError(message)
            time.sleep(self.poll_interval_s)
            job = self._get(f"/status/{job['id']}")

    @staticmethod
    def _output_events(job: dict[str, Any]) -> list[dict[str, Any]]:
        output = job.get("output", [])
        if not isinstance(output, list):  # handler non-générateur ou erreur amont
            message = f"sortie inattendue du worker : {type(output).__name__}"
            raise RemoteInferenceError(message)
        return output

    @staticmethod
    def _raise_if_failed(status: str, job: dict[str, Any]) -> None:
        if status in _TERMINAL_FAILURES:
            message = f"job {job.get('id')} terminé en {status} : {job.get('error', 'sans détail')}"
            raise RemoteInferenceError(message)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._send("POST", path, payload)

    def _get(self, path: str) -> dict[str, Any]:
        return self._send("GET", path, None)

    def _send(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        try:
            response = self._http.request(method, f"{self._base}{path}", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            message = f"{method} {path} : {exc}"
            raise RemoteInferenceError(message) from exc
        data: dict[str, Any] = response.json()
        return data


def _coerce_audio(audio: AudioInput) -> Waveform:
    """Normalise l'entrée audio ; les chemins passent par soundfile (absent du Pi → WAV requis)."""
    if isinstance(audio, Waveform):
        return audio
    if isinstance(audio, (str, Path)):
        return Waveform.from_file(audio)
    samples, sample_rate = audio
    return Waveform.of(samples, int(sample_rate))
