"""Résout le checkpoint au BUILD de l'image serverless (download → merge LoRA
éventuel → conversion Omni) dans ``LFM2_SERVE_CACHE``.

Exécuté par ``Dockerfile.serve``. Au boot du worker, le résolveur retrouve le
cache et ne refait rien : le cold start ne paie ni téléchargement ni conversion.

Env : LFM2_CHECKPOINT (requis), LFM2_BACKEND, LFM2_ADAPTER (optionnel),
HF_TOKEN (repos privés).
"""

from __future__ import annotations

import os

from lfm2_audio.ds.checkpoint import CheckpointRequest
from lfm2_audio.serving.checkpoint.resolver import CheckpointResolver


def main() -> None:
    request = CheckpointRequest(
        spec=os.environ["LFM2_CHECKPOINT"],
        backend=os.environ.get("LFM2_BACKEND", "vllm"),
        adapter=os.environ.get("LFM2_ADAPTER") or None,
    )
    resolved = CheckpointResolver().resolve(request)
    print(f"checkpoint résolu : {resolved.path} (layout {resolved.layout})")


if __name__ == "__main__":
    main()
