"""Our question sets, as Inspect datasets.

A spoken question travels as ``ContentAudio`` rather than as its transcript: the
model must hear it, and a run where it read the text instead would measure
something else entirely while looking identical in the report.

The expected tool calls ride in the sample metadata, which is where the
tool-call scorer reads them from — and where the viewer shows them next to what
the model actually emitted.
"""

from __future__ import annotations

import logging
from pathlib import Path

from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageUser, Content, ContentAudio

from lfm2_audio.evaluation.question import Question
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.inspect_bridge.audio import wav_file_to_data_uri

logger = logging.getLogger(__name__)


def to_sample(question: Question) -> Sample:
    """One evaluation case as an Inspect sample."""
    if question.audio_path is not None:
        content: list[Content] = [ContentAudio(audio=wav_file_to_data_uri(str(question.audio_path)), format="wav")]
        message = ChatMessageUser(content=content)
        sample_input: str | list[ChatMessage] = [message]
    else:
        sample_input = question.text

    return Sample(
        id=question.question_id,
        input=sample_input,
        target=question.reference_answer,
        metadata={
            **question.metadata,
            "expected_calls": question.expected_calls,
            # Kept even when the question is spoken: the viewer needs something
            # readable in the list, and a scorer may want the reference text.
            "prompt_text": question.text,
        },
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
"""``<repo>/python/lfm2_audio/inspect_bridge/dataset.py`` → the checkout root."""


def resolve_dataset_path(path: str) -> Path:
    """Find a dataset whether the caller passed a repo-relative path or not.

    ``inspect eval`` changes the working directory to the task file's own
    directory, so ``benchmark/…`` — which is what every other CLI in this repo
    takes — would not resolve. Falling back to the checkout root keeps one
    spelling working everywhere instead of forcing absolute paths.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    from_root = REPO_ROOT / path
    if from_root.exists():
        return from_root
    message = f"jeu de questions introuvable : {path} (ni depuis {Path.cwd()}, ni depuis {REPO_ROOT})"
    raise FileNotFoundError(message)


def question_set_dataset(
    path: str,
    *,
    audio_root: str | None = None,
    limit: int | None = None,
) -> Dataset:
    """Load a question-set JSONL as an Inspect dataset."""
    resolved = resolve_dataset_path(path)
    questions = QuestionSet.from_jsonl(resolved, audio_root=audio_root)
    if limit:
        questions = questions.take(limit)
    samples = [to_sample(question) for question in questions]
    spoken = sum(1 for question in questions if question.audio_path is not None)
    logger.info("%d cas chargés depuis %s (%d parlés)", len(samples), resolved, spoken)
    return MemoryDataset(samples=samples, name=str(resolved))
