"""``HubPushCallback`` — pousse l'adaptateur LoRA sur le Hub à intervalle régulier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from lfm2_audio.training.callback import TrainingCallback
from lfm2_audio.training.lora import LoraSettings, save_lora
from lfm2_audio.training.step_context import StepContext

logger = logging.getLogger(__name__)


class HubPushCallback(TrainingCallback):
    """Sauvegarde l'adaptateur et l'envoie sur un repo privé.

    Pousser en cours de route, et pas seulement à la fin, permet de reprendre une
    évaluation depuis un pas intermédiaire quand la courbe se dégrade ensuite —
    et de survivre à une instance interrompue.
    """

    def __init__(
        self,
        repo_id: str,
        output_dir: str | Path,
        lora_settings: LoraSettings,
        *,
        interval: int = 500,
        accelerator: Any = None,
        private: bool = True,
    ) -> None:
        self._repo_id = repo_id
        self._output_dir = Path(output_dir)
        self._lora_settings = lora_settings
        self._interval = interval
        self._accelerator = accelerator
        self._private = private

    def on_step_end(self, context: StepContext) -> None:
        if context.is_main_process and context.every(self._interval):
            self._push(context, label=f"step-{context.step}")

    def on_train_end(self, context: StepContext) -> None:
        if context.is_main_process:
            self._push(context, label="final")

    def _push(self, context: StepContext, *, label: str) -> None:
        model = context.model
        if model is None:
            logger.warning("aucun modèle dans le contexte : push %s ignoré", label)
            return

        adapter_dir = self._output_dir / "adapter"
        save_lora(model, adapter_dir, self._lora_settings)

        api = HfApi()
        api.create_repo(self._repo_id, private=self._private, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_dir),
            repo_id=self._repo_id,
            commit_message=f"adapter @ {label} (step {context.step})",
        )
        logger.info("adaptateur poussé vers %s (%s)", self._repo_id, label)
