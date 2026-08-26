"""Bridge to the VERSA toolkit — the metric authority at every gate.

Our own scorers steer training (fast, validated against VERSA by rank
correlation), but gate verdicts are read off VERSA numbers: multi-ear WER,
NISQA/UTMOS naturalness, speaker similarity for the single-voice gate. VERSA
lives in its own venv (``versa-eval/``) because its dependency set conflicts
with ours; this module shells out to that venv and never imports it.

The exchange format is VERSA's native one: a Kaldi-style ``.scp`` mapping
utterance keys to wav paths, a YAML metric config, and a JSONL output file
with one ``{"key": ..., metric: value, ...}`` object per utterance.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lfm2_audio.core.errors import Lfm2AudioError

logger = logging.getLogger(__name__)

DEFAULT_VERSA_ROOT = Path(__file__).resolve().parents[4] / "versa-eval"
"""Sibling of the repo checkout, where the 26/08 cross-validation installed VERSA."""

MOS_CONFIG = """\
- name: pseudo_mos
  predictor_types: ["dnsmos", "utmos"]
  predictor_args:
    dnsmos:
      fs: 16000
    utmos:
      fs: 16000
"""

SPEAKER_CONFIG = """\
- name: speaker
  model_tag: default
"""


def nisqa_config(versa_root: Path) -> str:
    """NISQA config; the weights were fetched by ``tools/setup_nisqa.sh``."""
    return f"""\
- name: nisqa
  nisqa_model_path: {versa_root / "versa" / "versa_cache" / "nisqa" / "nisqa.tar"}
  use_gpu: false
"""


def wer_config(model_tag: str = "medium") -> str:
    """Whisper WER config. VERSA's whisper_wer auto-detects the language per
    utterance, which is exactly right for a bilingual campaign — no per-run
    language knob to misconfigure."""
    return f"""\
- name: whisper_wer
  model_tag: {model_tag}
  beam_size: 1
  text_cleaner: whisper_basic
"""


class VersaError(Lfm2AudioError):
    """VERSA could not produce scores (missing install, subprocess failure)."""


class VersaRunner:
    """Runs VERSA metrics on wav files through the isolated venv."""

    def __init__(
        self,
        versa_root: Path = DEFAULT_VERSA_ROOT,
        *,
        timeout_s: float = 3600.0,
    ) -> None:
        self._root = versa_root
        self._timeout_s = timeout_s

    @property
    def root(self) -> Path:
        return self._root

    @property
    def python(self) -> Path:
        return self._root / ".venv" / "bin" / "python"

    @property
    def scorer_script(self) -> Path:
        return self._root / "versa" / "versa" / "bin" / "scorer.py"

    @property
    def available(self) -> bool:
        return self.python.exists() and self.scorer_script.exists()

    def score(
        self,
        wavs: Mapping[str, Path],
        config_yaml: str,
        *,
        gt: Mapping[str, Path] | None = None,
        text: Mapping[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Score ``wavs`` (key → path) and return ``key → {metric: value}``.

        ``gt`` provides reference wavs (speaker similarity, SpeechBERTScore);
        ``text`` provides reference transcripts (WER against a known text).
        """
        if not self.available:
            raise VersaError(
                f"VERSA introuvable sous {self._root} — venv ou scorer.py absent. Voir versa-eval/ (install du 26/08)."
            )
        if not wavs:
            return {}
        with tempfile.TemporaryDirectory(prefix="versa-") as workdir:
            work = Path(workdir)
            command = self._build_command(work, wavs, config_yaml, gt=gt, text=text)
            self._run(command)
            return self._read_scores(work / "scores.jsonl")

    def _build_command(
        self,
        work: Path,
        wavs: Mapping[str, Path],
        config_yaml: str,
        *,
        gt: Mapping[str, Path] | None,
        text: Mapping[str, str] | None,
    ) -> list[str]:
        (work / "config.yaml").write_text(config_yaml, encoding="utf-8")
        _write_scp(work / "pred.scp", wavs)
        command = [
            str(self.python),
            str(self.scorer_script),
            "--score_config",
            str(work / "config.yaml"),
            "--pred",
            str(work / "pred.scp"),
            "--output_file",
            str(work / "scores.jsonl"),
            "--io",
            "soundfile",
            # Without this, VERSA drops a versa_cache/ wherever cwd happens to
            # be — including our repo root during an audit run.
            "--cache_folder",
            str(self._root / "versa" / "versa_cache"),
        ]
        if gt:
            _write_scp(work / "gt.scp", gt)
            command += ["--gt", str(work / "gt.scp")]
        if text:
            lines = "".join(f"{key} {value}\n" for key, value in text.items())
            (work / "text").write_text(lines, encoding="utf-8")
            command += ["--text", str(work / "text")]
        return command

    def _run(self, command: list[str]) -> None:
        logger.info("VERSA : %s", " ".join(command))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            tail = completed.stderr.strip().splitlines()[-8:]
            raise VersaError("scorer.py a échoué :\n" + "\n".join(tail))

    def _read_scores(self, output: Path) -> dict[str, dict[str, Any]]:
        if not output.exists():
            raise VersaError("scorer.py s'est terminé sans écrire de scores")
        scores: dict[str, dict[str, Any]] = {}
        for line in output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            key = str(record.pop("key"))
            scores[key] = record
        return scores


def _write_scp(path: Path, entries: Mapping[str, Path]) -> None:
    path.write_text(
        "".join(f"{key} {wav}\n" for key, wav in entries.items()),
        encoding="utf-8",
    )
