"""``RoundResult`` — ce qu'une passe de génération a produit, committable à part.

Séparer la génération de son intégration au contexte est ce qui rend la passe
de décision **jetable**. Sans ça, une passe qui n'émet aucun appel d'outil a
déjà pollué le contexte quand on s'en aperçoit, et l'agent n'a plus d'autre
choix que de rendre son texte tel quel — c'est ainsi que v3 répondait aux tours
conversationnels dans le mode séquentiel où elle n'a jamais appris à répondre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from liquid_audio import ChatState

    from lfm2_audio.orchestrator.tool_parser import ParsedToolCall


@dataclass(slots=True)
class RoundResult:
    """Sortie d'une passe de génération et ses tokens bruts.

    ``commit`` réintègre les tokens dans le ``ChatState`` (cf. ``demo/chat.py``).
    Tant qu'il n'est pas appelé, la passe n'a laissé aucune trace.
    """

    pending_calls: list[ParsedToolCall] = field(default_factory=list)
    visible_text: str = ""
    audio_frames: int = 0
    interrupted: bool = False
    text_tokens: list[torch.Tensor] = field(default_factory=list)
    audio_tokens: list[torch.Tensor] = field(default_factory=list)
    modality_flags: list[int] = field(default_factory=list)

    @property
    def emitted_tool_call(self) -> bool:
        return bool(self.pending_calls)

    def commit(self, chat: ChatState) -> None:
        """Réintègre les tokens générés dans le contexte de conversation."""
        if not (self.text_tokens or self.audio_tokens):
            return
        device: Any = chat.device
        text = (
            torch.stack(self.text_tokens, 1)
            if self.text_tokens
            else torch.empty((1, 0), dtype=torch.long, device=device)
        )
        audio = (
            torch.stack(self.audio_tokens, 1)
            if self.audio_tokens
            else torch.empty((chat.codebooks, 0), dtype=torch.long, device=device)
        )
        chat.append(text=text, audio_out=audio, modality_flag=torch.tensor([self.modality_flags], device=device))
