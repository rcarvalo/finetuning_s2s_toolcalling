"""Contrat de fil entre le client et le worker serverless.

Une **seule** définition, importée par les deux bords : le client
(:mod:`lfm2_audio.remote.client`) et le handler RunPod (``infra/handler.py``).
Un champ qui change ici casse les tests des deux côtés — c'est exactement ce
qu'on veut, plutôt qu'un contrat implicite en ``dict`` qui dérive en silence.

Le JSON vient du réseau : il est **validé**, jamais lu à la pioche.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class HistoryTurnPayload(BaseModel):
    """A past turn, replayed to give the stateless worker its context.

    Text only: the one-audio-per-conversation invariant means past user audio
    cannot be replayed anyway — the model's own past replies carry the thread,
    exactly as they do in the local multi-turn path.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = ""


class TurnRequest(BaseModel):
    """Ce que le client place dans ``input`` — un tour de dialogue à générer."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    audio_b64: str | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    history: list[HistoryTurnPayload] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_text_or_audio(self) -> TurnRequest:
        if self.text is None and self.audio_b64 is None:
            message = "il faut au moins text= ou audio="
            raise ValueError(message)
        return self


class TurnMetricsPayload(BaseModel):
    """Latences mesurées par le worker (miroir de :class:`~lfm2_audio.ds.reply.TurnMetrics`)."""

    model_config = ConfigDict(extra="ignore")

    ttfa_s: float | None = None
    total_s: float = 0.0
    audio_frames: int = 0


class AudioChunkEvent(BaseModel):
    """Un morceau d'audio, émis dès sa sortie du modèle."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["audio"] = "audio"
    audio_b64: str
    sample_rate: int = Field(gt=0)


class FinalEvent(BaseModel):
    """Dernier événement du tour : texte et métriques."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["final"] = "final"
    text: str = ""
    raw_text: str = ""
    metrics: TurnMetricsPayload = TurnMetricsPayload()


class ErrorEvent(BaseModel):
    """Le worker n'a pas pu traiter la requête."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["error"]
    error: str


TurnEvent = Annotated[AudioChunkEvent | FinalEvent | ErrorEvent, Field(discriminator="kind")]
"""Union discriminée par ``kind`` — le parsing choisit le bon modèle tout seul."""

TURN_EVENT_ADAPTER: TypeAdapter[AudioChunkEvent | FinalEvent | ErrorEvent] = TypeAdapter(TurnEvent)


class JobEnvelope(BaseModel):
    """Réponse de ``/run``, ``/runsync`` et ``/status`` — l'enveloppe RunPod."""

    model_config = ConfigDict(extra="ignore")

    id: str = ""
    status: str = ""
    output: list[dict[str, Any]] | dict[str, Any] | None = None
    error: str | None = None

    @property
    def events(self) -> list[AudioChunkEvent | FinalEvent | ErrorEvent]:
        """Les événements du tour, validés. Une sortie non-liste est une anomalie amont."""
        if self.output is None:
            return []
        if not isinstance(self.output, list):
            message = f"sortie inattendue du worker : {type(self.output).__name__}"
            raise ValueError(message)
        return [TURN_EVENT_ADAPTER.validate_python(item) for item in self.output]


class StreamItem(BaseModel):
    """Une ligne de ``/stream`` : l'événement émis par le handler."""

    model_config = ConfigDict(extra="ignore")

    output: TurnEvent


class StreamPage(BaseModel):
    """Une page de ``/stream`` — le polling en consomme une à chaque tour de boucle."""

    model_config = ConfigDict(extra="ignore")

    status: str = ""
    stream: list[StreamItem] = Field(default_factory=list)
