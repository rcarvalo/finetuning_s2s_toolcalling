"""Start a repo job on a Colab VM, detached, and return immediately.

`colab exec` has a short timeout — long enough to launch a job, far too short to
run one — so the work is detached and observed through small probes afterwards.

Uploaded and run with:

    colab upload -s <session> infra/colab_launch.py /content/colab_launch.py
    colab exec -s <session> -f infra/colab_launch.py

The job to run comes from the environment written into the launcher call below;
edit LFM2_JOB/LFM2_ARGS at the top or pass them through the shell environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

BRANCH = os.environ.get("LFM2_BRANCH", "rd/pr_rca_eval_baseline")
JOB = os.environ.get("LFM2_JOB", "")
ARGS = os.environ.get("LFM2_ARGS", "")
ENTRYPOINT_URL = (
    f"https://raw.githubusercontent.com/rcarvalo/finetuning_s2s_toolcalling/{BRANCH}/infra/colab_entrypoint.sh"
)


def main() -> None:
    if not JOB:
        raise SystemExit("LFM2_JOB manquant")
    entrypoint = Path("/content/colab_entrypoint.sh")
    # Fetched from the branch rather than uploaded: the launcher and the job
    # then always come from the same commit.
    with urllib.request.urlopen(ENTRYPOINT_URL) as response:
        entrypoint.write_bytes(response.read())

    log = Path("/content/job.log")
    subprocess.run(["pkill", "-f", "colab_entrypoint.sh"], capture_output=True, check=False)
    proc = subprocess.Popen(
        ["bash", str(entrypoint)],
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "LFM2_BRANCH": BRANCH, "LFM2_JOB": JOB, "LFM2_ARGS": ARGS},
    )
    print(f"job {JOB} lancé (pid {proc.pid}), log /content/job.log", file=sys.stderr)
    print(f"job {JOB} lancé, pid {proc.pid}")


if __name__ == "__main__":
    main()
