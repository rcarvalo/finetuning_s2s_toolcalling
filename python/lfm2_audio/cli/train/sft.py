"""Lancement du SFT LoRA (Trainer officiel liquid-audio + observateurs).

    accelerate launch -m lfm2_audio.cli.train.sft --config configs/training/<recette>.yaml

La logique vit dans :mod:`lfm2_audio.training` — ce module ne porte que la CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from lfm2_audio.ds.training_config import TrainingConfig
from lfm2_audio.training.lora import save_lora
from lfm2_audio.training.train_sft import build_trainer, lora_settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="recette YAML (cf. configs/training/)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="valide la recette et l'affiche sans rien entraîner",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = TrainingConfig.from_yaml(args.config)
    if args.print_config:
        print(json.dumps(config.as_dict(), indent=2, ensure_ascii=False))
        return 0

    from lfm2_audio.training.eval_generator import liquid_generator_factory

    # Without a factory the ScoringCallback silently skips every measurement:
    # the June run trained blind because of exactly this missing argument.
    trainer = build_trainer(config, generator_factory=liquid_generator_factory(config))
    trainer.train()

    # Le Trainer écrit déjà `<output_dir>/final` (modèle complet, LoRA non fusionné) :
    # on ajoute l'adaptateur seul, léger à versionner et à recharger.
    if config.lora.enabled and trainer.accelerator.is_main_process:
        path = save_lora(
            trainer.accelerator.unwrap_model(trainer.model),
            Path(config.output_dir) / "final_adapter",
            lora_settings(config),
        )
        logger.info("adaptateur LoRA écrit : %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
