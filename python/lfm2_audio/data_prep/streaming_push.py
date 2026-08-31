"""Incremental HF push while a synthesis job is still running.

Exists because of a precise loss mode: a RunPod balance hitting zero deletes
the pod *and everything on it* — one pod once ran two hours for nothing. The
answer is to make the Hub the primary store while the job runs: push every few
batches, so an exhausted balance costs at most one interval of work, and the
run can deliberately be left to die on empty credit.

Only NEW files travel: ``upload_folder`` re-lists the whole tree per commit,
which grows quadratically painful past a few thousand clips. Here each push is
one commit holding the wavs added since the last push, plus the fresh manifest.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StreamingPusher:
    """Pushes a growing corpus folder to a HF dataset repo, new files only.

    Never raises out of :meth:`push`: a Hub hiccup must not kill a synthesis
    run that costs GPU money — the next interval retries what this one missed.
    """

    def __init__(self, folder: Path, repo_id: str, path_in_repo: str) -> None:
        self.folder = folder
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self._pushed: set[str] = set()
        # HfApi, importé paresseusement : le module doit rester importable sans
        # huggingface_hub (les tests unitaires le remplacent par un mock).
        self._api: Any = None

    @property
    def enabled(self) -> bool:
        return bool(os.environ.get("HF_TOKEN"))

    def verify(self) -> bool:
        """Prove the pipe before spending GPU time: create the repo, push a marker.

        A push pipeline that fails only at the FIRST real push has already
        burned the batches that preceded it.
        """
        if not self.enabled:
            logger.warning("HF_TOKEN absent — le run produira sur disque pod UNIQUEMENT (perdu si solde épuisé)")
            return False
        try:
            from huggingface_hub import HfApi

            self._api = HfApi()
            self._api.create_repo(self.repo_id, repo_type="dataset", exist_ok=True)
            who = self._api.whoami()["name"]
            logger.info("push HF vérifié : %s → %s/%s", who, self.repo_id, self.path_in_repo)
            return True
        except Exception as error:
            logger.warning("vérification du push HF échouée : %s", error)
            self._api = None
            return False

    def preload_existing(self) -> set[str]:
        """Mark what the repo already holds, so a fresh VM never re-sends it.

        Colab sessions die after an hour or two and their disks with them; the
        Hub is the only memory a relaunch can trust. Returns the basenames
        already present under this brick, for the caller to skip producing.
        """
        if self._api is None:
            return set()
        try:
            prefix = f"{self.path_in_repo}/"
            existing = [
                f[len(prefix) :]
                for f in self._api.list_repo_files(self.repo_id, repo_type="dataset")
                if f.startswith(prefix) and not f.endswith("manifest.jsonl")
            ]
            self._pushed.update(existing)
            logger.info("reprise Hub : %d fichiers déjà présents sous %s", len(existing), self.path_in_repo)
            return {Path(f).name for f in existing}
        except Exception as error:
            logger.warning("préchargement de l'existant échoué : %s", error)
            return set()

    def push(self, *, message: str) -> int:
        """One commit with every file not pushed yet. Returns how many went up."""
        if self._api is None:
            return 0
        try:
            from huggingface_hub import CommitOperationAdd

            fresh = [
                path
                for path in sorted(self.folder.rglob("*"))
                if path.is_file() and str(path.relative_to(self.folder)) not in self._pushed
            ]
            # The manifest changes every batch: always re-send it.
            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{self.path_in_repo}/{path.relative_to(self.folder)}",
                    path_or_fileobj=str(path),
                )
                for path in fresh
            ]
            if not operations:
                return 0
            self._api.create_commit(
                repo_id=self.repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=message,
            )
            for path in fresh:
                relative = str(path.relative_to(self.folder))
                if not relative.endswith("manifest.jsonl"):
                    self._pushed.add(relative)
            logger.info("push HF : %d fichiers (%s)", len(operations), message)
            return len(operations)
        except Exception as error:
            logger.warning("push HF échoué (retenté au prochain intervalle) : %s", error)
            return 0
