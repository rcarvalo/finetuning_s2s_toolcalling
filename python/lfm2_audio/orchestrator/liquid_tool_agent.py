"""``LiquidToolAgent`` — le ReceptionAgent liquid sous l'interface de l'agent vLLM.

Les deux agents font le même travail mais ne s'appellent pas pareil :
``VllmToolAgent.respond(wave)`` porte sa propre conversation, tandis que
``ReceptionAgent.respond(chat, wav, rate)`` réclame un ``ChatState`` que
l'appelant gère. Cette classe tient le ``ChatState`` et expose ``respond(wave)``
— la démo tool-calling devient donc indifférente au backend.

Pourquoi ce chemin existe : vLLM-Omni importe ``diffusers``, dont la chaîne de
dépendances (peft ↔ torchao) s'est révélée non résoluble sur l'image Colab du
26/08. Le backend liquid n'a aucune de ces dépendances et porte le correctif
deux-passes (réponse en mode parole quand aucun outil n'est appelé). Sa
contrepartie est un TTFA plus élevé — d'où les fillers audio.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from lfm2_audio.ds.audio import Waveform
from lfm2_audio.orchestrator.events import AgentEvent
from lfm2_audio.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from lfm2_audio.orchestrator.agent import ReceptionAgent


class LiquidToolAgent:
    """Adapte :class:`ReceptionAgent` à ``respond(wave)``."""

    def __init__(self, agent: ReceptionAgent) -> None:
        self._agent = agent
        self._chat: Any = agent.new_session()

    @property
    def registry(self) -> ToolRegistry:
        return self._agent.registry

    def reset(self) -> None:
        """Nouvelle conversation — le contexte multi-tours repart à zéro."""
        self._chat = self._agent.new_session()

    def respond(self, wave: Waveform) -> Iterator[AgentEvent]:
        samples = torch.as_tensor(wave.samples, dtype=torch.float32).reshape(1, -1)
        yield from self._agent.respond(self._chat, samples, wave.sample_rate)
