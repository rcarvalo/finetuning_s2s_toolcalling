"""``BenchSession`` — a loaded model plus the test set being listened through.

Holds the one expensive thing (the model) so the UI can stay stateless, and
names the version being judged. That name is what ties a verdict to a model:
without it, ratings collected across two checkpoints are indistinguishable and
the whole exercise is worthless.
"""

from __future__ import annotations

import logging
from pathlib import Path

from lfm2_audio.bench.rating import Rating
from lfm2_audio.bench.store import RatingStore
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.reply import Reply
from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)


class BenchSession:
    """Generates answers for a test set and records what the listener thought."""

    def __init__(
        self,
        model: LFM2Audio,
        questions: QuestionSet,
        *,
        version: str,
        store: RatingStore | None = None,
        audio_dir: str | Path = "reports/bench_audio",
    ) -> None:
        self._model = model
        self._questions = questions
        self._version = version
        self._store = store or RatingStore()
        self._audio_dir = Path(audio_dir)

    @property
    def version(self) -> str:
        return self._version

    @property
    def questions(self) -> QuestionSet:
        return self._questions

    @property
    def store(self) -> RatingStore:
        return self._store

    def case_ids(self) -> list[str]:
        return [q.question_id for q in self._questions]

    def question(self, case_id: str) -> Question:
        for question in self._questions:
            if question.question_id == case_id:
                return question
        message = f"unknown case: {case_id!r}"
        raise KeyError(message)

    def pending(self) -> list[str]:
        """Cases not yet rated for this version, so a session can be resumed."""
        done = self._store.rated_cases(self._version)
        return [case for case in self.case_ids() if case not in done]

    def generate(self, case_id: str, *, max_tokens: int = 400) -> tuple[Reply, Path | None]:
        """Answer one case and keep the audio on disk.

        The audio is persisted because a verdict without the clip it refers to
        cannot be revisited, re-checked, or compared against a later version.
        """
        question = self.question(case_id)
        self._model.reset()

        audio_in = Waveform.from_file(question.audio_path) if question.audio_path else None
        reply = self._model.reply(
            text=None if audio_in is not None else question.text,
            audio=audio_in,
            max_tokens=max_tokens,
        )

        saved: Path | None = None
        if reply.audio is not None and not reply.audio.is_empty:
            destination = self._audio_dir / _slug(self._version) / f"{case_id}.wav"
            saved = reply.audio.save(destination)
        return reply, saved

    def talk(
        self,
        *,
        text: str | None = None,
        audio: Waveform | None = None,
        max_tokens: int = 400,
    ) -> Reply:
        """One conversational turn, history preserved.

        Unlike :meth:`generate`, this does **not** reset between turns: the talk
        tab is for holding a conversation, where the context is the point.
        """
        return self._model.reply(text=text, audio=audio, max_tokens=max_tokens)

    def reset_conversation(self) -> None:
        """Drop the conversational history."""
        self._model.reset()

    def record(
        self,
        case_id: str,
        *,
        intelligibility: int,
        naturalness: int,
        overall: int,
        derailed: bool = False,
        notes: str = "",
    ) -> Rating:
        """Persist one verdict and return it."""
        rating = Rating.create(
            case_id,
            self._version,
            intelligibility=intelligibility,
            naturalness=naturalness,
            overall=overall,
            derailed=derailed,
            notes=notes,
        )
        self._store.append(rating)
        logger.info("rated %s (%s): overall=%d", case_id, self._version, overall)
        return rating

    def progress(self) -> str:
        total = len(self._questions)
        remaining = len(self.pending())
        return f"{total - remaining}/{total} rated for {self._version}"


def _slug(version: str) -> str:
    """Filesystem-safe form of a version label."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in version)
