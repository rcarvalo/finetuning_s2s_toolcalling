"""Deterministic per-turn randomness for speech synthesis.

Kept out of the TTS CLI, which imports torch and kokoro at load time: this
policy is pure and must stay testable on a machine with neither.

A synthesis run over thousands of utterances gets interrupted — Colab reclaims
the VM, the TTS server dies. Resuming skips the turns already on disk, and with
a single shared RNG that skip would shift every later draw: the same corpus
would end up voiced differently depending on where the previous run stopped.
Deriving each turn's draw from its own identity removes that coupling.
"""

from __future__ import annotations

import hashlib
import random


def turn_random(seed: int, dialogue_id: str, turn_index: int) -> random.Random:
    """A generator bound to one turn, independent of how many preceded it."""
    digest = hashlib.sha1(f"{seed}:{dialogue_id}:{turn_index}".encode(), usedforsecurity=False).hexdigest()
    return random.Random(int(digest[:16], 16))
