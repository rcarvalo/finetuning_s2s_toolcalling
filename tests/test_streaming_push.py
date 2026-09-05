"""The streaming pusher must send only new files, and never kill the run.

The loss mode it guards against is real: an exhausted RunPod balance deletes
the pod and its disk — so the push must work from the FIRST interval (verify
before spending), and a Hub error must cost one interval, not the job.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lfm2_audio.data_prep.streaming_push import StreamingPusher


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio/a.wav").write_bytes(b"a")
    (tmp_path / "manifest.jsonl").write_text("{}\n")
    return tmp_path


@pytest.fixture
def pusher(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> StreamingPusher:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    pusher = StreamingPusher(corpus, "user/repo", "A_assistant_speech")
    pusher._api = MagicMock()
    return pusher


def test_should_push_each_new_file_exactly_once(pusher: StreamingPusher, corpus: Path) -> None:
    assert pusher.push(message="lot 1") == 2  # a.wav + manifest

    (corpus / "audio/b.wav").write_bytes(b"b")
    sent = pusher.push(message="lot 2")

    assert sent == 2  # b.wav + manifest re-sent, a.wav NOT re-sent
    operations = pusher._api.create_commit.call_args.kwargs["operations"]
    paths = {op.path_in_repo for op in operations}
    assert paths == {"A_assistant_speech/audio/b.wav", "A_assistant_speech/manifest.jsonl"}


def test_should_resend_the_manifest_every_push(pusher: StreamingPusher) -> None:
    pusher.push(message="lot 1")
    assert pusher.push(message="lot 2") == 1  # rien de neuf sauf le manifeste


def test_should_survive_a_hub_error_and_retry_next_interval(pusher: StreamingPusher, corpus: Path) -> None:
    pusher._api.create_commit.side_effect = [RuntimeError("hub down"), None]

    assert pusher.push(message="lot 1") == 0  # avalé, pas levé
    assert pusher.push(message="lot 2") == 2  # a.wav retenté + manifeste


def test_should_do_nothing_without_verification(corpus: Path) -> None:
    pusher = StreamingPusher(corpus, "user/repo", "A")
    assert pusher.push(message="x") == 0


def test_should_refuse_verification_without_token(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    pusher = StreamingPusher(corpus, "user/repo", "A")
    assert pusher.verify() is False


def test_should_resend_the_rejection_log_every_push(pusher: StreamingPusher, corpus: Path) -> None:
    (corpus / "dropped.jsonl").write_text("{}\n")
    pusher.push(message="lot 1")

    assert pusher.push(message="lot 2") == 2  # manifeste + journal des rejets, rien d'autre


def test_preload_should_count_only_accepted_clips_as_done(pusher: StreamingPusher) -> None:
    pusher._api.list_repo_files.return_value = [
        "A_assistant_speech/audio/a.wav",
        "A_assistant_speech/rejected/b.wav",
        "A_assistant_speech/dropped.jsonl",
        "A_assistant_speech/manifest.jsonl",
    ]

    assert pusher.preload_existing() == {"a.wav"}
