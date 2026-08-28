"""A remote job must be able to say where it is.

Regression for a 45-minute silent run: the stack was installing with ``pip -q``
and nothing distinguished progress from a hang — RunPod's CPU/GPU metrics read
near zero for a working pod too.
"""

from __future__ import annotations

import sys

import pytest

from lfm2_audio.core.progress import Progress, stream_command


class FakeClock:
    """Advances by one second per read, so elapsed times are deterministic."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def test_should_announce_a_phase_before_it_runs(capsys: pytest.CaptureFixture[str]) -> None:
    progress = Progress("job", clock=FakeClock())
    progress.step("installation")

    out = capsys.readouterr().out
    assert "▶ installation" in out


def test_should_close_the_previous_phase_with_its_duration(capsys: pytest.CaptureFixture[str]) -> None:
    progress = Progress("job", clock=FakeClock())
    progress.step("installation")
    progress.step("téléchargement")

    out = capsys.readouterr().out
    assert "✓ installation" in out
    assert "▶ téléchargement" in out


def test_should_report_the_failing_phase_when_the_block_raises(capsys: pytest.CaptureFixture[str]) -> None:
    """A crash must name where it happened, not just end the log."""

    def crash_inside_a_phase() -> None:
        with Progress("job", clock=FakeClock()) as progress:
            progress.step("moteur")
            raise RuntimeError("orchestrator init failed")

    with pytest.raises(RuntimeError):
        crash_inside_a_phase()

    out = capsys.readouterr().out
    assert "✗ moteur" in out
    assert "orchestrator init failed" in out


def test_should_forward_periodic_lines_from_a_command(capsys: pytest.CaptureFixture[str]) -> None:
    progress = Progress("job", clock=FakeClock())
    code = stream_command([sys.executable, "-c", "[print(i) for i in range(60)]"], progress, every=25)

    out = capsys.readouterr().out
    assert code == 0
    assert "  0" in out
    assert "  50" in out
