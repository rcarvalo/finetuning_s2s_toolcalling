"""``LatencySample`` — une mesure de latence : un prompt, un tour."""

from __future__ import annotations

from dataclasses import dataclass

from lfm2_audio.ds.reply import Reply


@dataclass(frozen=True, slots=True)
class LatencySample:
    """Une mesure : un prompt, un tour."""

    prompt: str
    ttfa_s: float | None
    total_s: float
    audio_s: float
    audio_frames: int
    text: str

    @property
    def real_time_factor(self) -> float | None:
        return self.total_s / self.audio_s if self.audio_s else None

    @classmethod
    def from_reply(cls, prompt: str, reply: Reply) -> LatencySample:
        return cls(
            prompt=prompt,
            ttfa_s=reply.metrics.ttfa_s,
            total_s=reply.metrics.total_s,
            audio_s=reply.audio.duration_s if reply.audio else 0.0,
            audio_frames=reply.metrics.audio_frames,
            text=reply.text,
        )
