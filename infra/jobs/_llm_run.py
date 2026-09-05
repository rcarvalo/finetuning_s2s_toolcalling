"""Run one generator shard as a subprocess and read back what it cost."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

JOBS_DIR = Path(__file__).resolve().parent
if str(JOBS_DIR) not in sys.path:
    sys.path.insert(0, str(JOBS_DIR))
from _budget import parse_spend  # noqa: E402


@dataclass(frozen=True)
class LlmArgs:
    """The provider flags forwarded verbatim to every generator run."""

    provider: str = "gemini"
    model: str | None = None
    effort: str = "low"
    batch: bool = False

    def flags(self, max_usd: float | None) -> list[str]:
        flags = ["--provider", self.provider, "--effort", self.effort]
        if self.model:
            flags += ["--model", self.model]
        if self.batch:
            flags.append("--batch")
        if max_usd is not None:
            flags += ["--max-usd", f"{max_usd:.4f}"]
        return flags


@dataclass(frozen=True)
class ShardRun:
    status: int
    usd: float | None


def run_capturing(cmd: list[str], cwd: Path) -> ShardRun:
    """Echo the run line by line (the log is the only progress signal) and keep its spend line."""
    print("===", " ".join(cmd), flush=True)
    usd: float | None = None
    with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            spent = parse_spend(line)
            if spent is not None:
                usd = spent
    return ShardRun(proc.returncode, usd)
