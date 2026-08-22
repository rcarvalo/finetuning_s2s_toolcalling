"""``Tokenizer`` — surface minimale attendue d'un tokenizer HF.

Un ``Protocol`` plutôt qu'un ``Any`` : le code du projet n'utilise que trois
opérations sur le tokenizer, et les nommer ici documente le contrat tout en
gardant le typage vérifiable. N'importe quel ``PreTrainedTokenizer`` le satisfait
sans héritage — et un double de test aussi, ce qui rend le rendu de prompt
testable sans télécharger de modèle.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


class Encoding(Protocol):
    """Ce que renvoie un appel de tokenisation."""

    input_ids: list[int]


@runtime_checkable
class Tokenizer(Protocol):
    """Tokenizer texte du backbone LFM2.5."""

    def __call__(self, text: str, *, add_special_tokens: bool = ...) -> Encoding: ...

    def encode(self, text: str, *, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...
