"""Tests of the VERSA bridge — subprocess mocked, no VERSA install needed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from lfm2_audio.evaluation.versa_runner import (
    MOS_CONFIG,
    VersaError,
    VersaRunner,
    nisqa_config,
    wer_config,
)


@pytest.fixture
def versa_root(tmp_path: Path) -> Path:
    """A fake install: the two files ``available`` checks for."""
    python = tmp_path / ".venv" / "bin" / "python"
    scorer = tmp_path / "versa" / "versa" / "bin" / "scorer.py"
    for file in (python, scorer):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    return tmp_path


class FakeRun:
    """Records the command and simulates scorer.py writing its output file."""

    def __init__(self, scores: list[dict[str, Any]], returncode: int = 0) -> None:
        self._scores = scores
        self._returncode = returncode
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self._returncode == 0:
            output = Path(command[command.index("--output_file") + 1])
            output.write_text(
                "".join(json.dumps(record) + "\n" for record in self._scores),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, self._returncode, stdout="", stderr="boom\n")


def test_should_report_unavailable_without_an_install(tmp_path: Path) -> None:
    runner = VersaRunner(tmp_path / "nowhere")

    assert runner.available is False
    with pytest.raises(VersaError, match="introuvable"):
        runner.score({"a": tmp_path / "a.wav"}, MOS_CONFIG)


def test_should_return_scores_keyed_by_utterance(versa_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun([{"key": "u1", "utmos": 4.3}, {"key": "u2", "utmos": 2.1}])
    monkeypatch.setattr(subprocess, "run", fake)
    runner = VersaRunner(versa_root)

    scores = runner.score({"u1": Path("/x/u1.wav"), "u2": Path("/x/u2.wav")}, MOS_CONFIG)

    assert scores == {"u1": {"utmos": 4.3}, "u2": {"utmos": 2.1}}


def test_should_call_the_isolated_venv_with_scp_and_config(versa_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun([{"key": "u1", "utmos": 4.0}])
    monkeypatch.setattr(subprocess, "run", fake)

    VersaRunner(versa_root).score({"u1": Path("/x/u1.wav")}, MOS_CONFIG)

    command = fake.commands[0]
    assert command[0] == str(versa_root / ".venv" / "bin" / "python")
    assert command[1] == str(versa_root / "versa" / "versa" / "bin" / "scorer.py")
    assert "--gt" not in command
    assert "--text" not in command


def test_should_pass_references_when_given(versa_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRun([{"key": "u1", "spk_similarity": 0.91}])
    monkeypatch.setattr(subprocess, "run", fake)

    VersaRunner(versa_root).score(
        {"u1": Path("/x/u1.wav")},
        MOS_CONFIG,
        gt={"u1": Path("/x/ref1.wav")},
        text={"u1": "bonjour tout le monde"},
    )

    command = fake.commands[0]
    assert "--gt" in command
    assert "--text" in command


def test_should_raise_with_stderr_when_the_scorer_fails(versa_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", FakeRun([], returncode=1))

    with pytest.raises(VersaError, match="boom"):
        VersaRunner(versa_root).score({"u1": Path("/x/u1.wav")}, MOS_CONFIG)


def test_should_raise_when_no_output_was_written(versa_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def silent(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", silent)

    with pytest.raises(VersaError, match="sans écrire"):
        VersaRunner(versa_root).score({"u1": Path("/x/u1.wav")}, MOS_CONFIG)


def test_empty_input_should_short_circuit(versa_root: Path) -> None:
    assert VersaRunner(versa_root).score({}, MOS_CONFIG) == {}


def test_config_builders_should_produce_valid_yaml() -> None:
    for config in (MOS_CONFIG, nisqa_config(Path("/opt/versa")), wer_config()):
        parsed = yaml.safe_load(config)
        assert isinstance(parsed, list) and parsed[0]["name"]


def test_should_keep_scores_when_the_scorer_dies_after_writing_them(
    versa_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On macOS scorer.py regularly crashes during teardown, after logging
    'Scoring completed' and writing every score. Discarding a complete
    measurement over that exit code throws away real work."""
    fake = FakeRun([{"key": "u1", "utmos": 4.1}])
    fake._returncode = 0  # write the file...

    def crash_after_writing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        fake(command, **kwargs)
        return subprocess.CompletedProcess(command, -6, stdout="", stderr="recursive_mutex lock failed\n")

    monkeypatch.setattr(subprocess, "run", crash_after_writing)

    scores = VersaRunner(versa_root).score({"u1": Path("/x/u1.wav")}, MOS_CONFIG)

    assert scores == {"u1": {"utmos": 4.1}}
