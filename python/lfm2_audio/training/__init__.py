"""Entraînement : wrapper du ``Trainer`` officiel et observateurs configurables.

Pas de réexport ici : ``lora`` importe peft, ``instrumented_trainer`` importe
liquid-audio. Un ``__init__`` qui les réexporterait rendrait
``import lfm2_audio.training.callback`` — du Python pur — impossible sans GPU.
Importer les modules directement.
"""
