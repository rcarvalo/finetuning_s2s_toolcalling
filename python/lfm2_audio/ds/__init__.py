"""Structures de données du projet.

Deux familles, choisies selon l'origine de la donnée (règle « validation aux
frontières ») :

- **pydantic** pour ce qui vient de l'extérieur — configs YAML/env
  (:mod:`~lfm2_audio.ds.inference_config`), dialogues JSONL (:mod:`~lfm2_audio.ds.dialogue`) ;
- **dataclasses** pour les objets construits par le code — value objects audio
  (:mod:`~lfm2_audio.ds.audio`), tours de conversation, réponses.
"""

from lfm2_audio.ds.audio import INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, Waveform
from lfm2_audio.ds.checkpoint import CheckpointRequest, Layout, ResolvedCheckpoint
from lfm2_audio.ds.conversation import Conversation, ConversationTurn
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.ds.reply import Reply, TurnMetrics

__all__ = [
    "INPUT_SAMPLE_RATE",
    "OUTPUT_SAMPLE_RATE",
    "CheckpointRequest",
    "Conversation",
    "ConversationTurn",
    "EngineConfig",
    "GenerationConfig",
    "Layout",
    "Reply",
    "ResolvedCheckpoint",
    "TurnMetrics",
    "Waveform",
]
