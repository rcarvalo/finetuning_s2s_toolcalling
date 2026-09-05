"""VERSA bridge — the runner now lives in the evaluation toolkit; this keeps the checkout default.

VERSA lives in ``versa-eval/`` next to this repository (its dependency set
conflicts with ours), so the historical default root and the metric-set YAML
strings are kept here for the CLIs and jobs that import them.
"""

from __future__ import annotations

from pathlib import Path

from avet.errors import VersaError
from avet.versa.versa_metric_config import VersaMetricConfig
from avet.versa.versa_runner import VersaRunner as _VersaRunner

DEFAULT_VERSA_ROOT = Path(__file__).resolve().parents[4] / "versa-eval"
"""Sibling of the repo checkout, where the 26/08 cross-validation installed VERSA."""

MOS_CONFIG = VersaMetricConfig.of("mos", root=DEFAULT_VERSA_ROOT).yaml
SPEAKER_CONFIG = VersaMetricConfig.of("speaker", root=DEFAULT_VERSA_ROOT).yaml


def nisqa_config(versa_root: Path) -> str:
    """NISQA config; the weights were fetched by ``tools/setup_nisqa.sh``."""
    return VersaMetricConfig.of("nisqa", root=versa_root).yaml


def wer_config(model_tag: str = "medium") -> str:
    """Whisper WER config (language auto-detected per utterance)."""
    return VersaMetricConfig.of("wer", root=DEFAULT_VERSA_ROOT, whisper_tag=model_tag).yaml


class VersaRunner(_VersaRunner):
    """The toolkit's runner, pointed at the sibling checkout by default."""

    def __init__(self, versa_root: Path = DEFAULT_VERSA_ROOT, *, timeout_s: float = 3600.0) -> None:
        super().__init__(versa_root, timeout_s=timeout_s)


__all__ = [
    "DEFAULT_VERSA_ROOT",
    "MOS_CONFIG",
    "SPEAKER_CONFIG",
    "VersaError",
    "VersaRunner",
    "nisqa_config",
    "wer_config",
]
