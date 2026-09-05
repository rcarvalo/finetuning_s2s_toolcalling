"""Un shard = un sous-processus dont on relit le statut ET la dépense."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[1] / "infra" / "jobs" / "_llm_run.py"


@pytest.fixture
def llm_run() -> Any:
    spec = importlib.util.spec_from_file_location("_llm_run", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses résout les annotations différées via sys.modules
    spec.loader.exec_module(module)
    return module


class TestLlmArgs:
    def test_should_default_to_gemini_without_a_cap(self, llm_run: Any) -> None:
        assert llm_run.LlmArgs().flags(None) == ["--provider", "gemini", "--effort", "low"]

    def test_should_forward_model_batch_and_cap(self, llm_run: Any) -> None:
        args = llm_run.LlmArgs("anthropic", "claude-sonnet-5", "medium", True)

        expected = ["--provider", "anthropic", "--effort", "medium", "--model", "claude-sonnet-5"]
        assert args.flags(1.5) == [*expected, "--batch", "--max-usd", "1.5000"]


class TestRunCapturing:
    def test_should_return_the_status_and_the_spend_line(
        self, llm_run: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        script = "print('travail'); print('===SPEND=== model=m calls=1 in=1 out=1 usd=0.25'); raise SystemExit(3)"

        result = llm_run.run_capturing([sys.executable, "-c", script], tmp_path)

        assert (result.status, result.usd) == (3, 0.25)
        assert "travail" in capsys.readouterr().out

    def test_should_report_no_spend_for_a_run_that_does_not_count(self, llm_run: Any, tmp_path: Path) -> None:
        result = llm_run.run_capturing([sys.executable, "-c", "print('ok')"], tmp_path)

        assert (result.status, result.usd) == (0, None)
