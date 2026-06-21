"""Agent tool-calling S2S sur backend vLLM-Omni (basse latence, RTF<1).

Contrairement à l'agent liquid (``ChatState`` persistant + KV cache), vLLM-Omni
régénère depuis un prompt complet à chaque appel. On orchestre donc le tour en
**passes** successives, pilotées par un point d'arrêt sur ``<|tool_call_end|>`` :

- **Pass A** (round 0) : génère le tour assistant en s'arrêtant sur
  ``<|tool_call_end|>`` OU ``<|im_end|>``.
  - *positif* → le modèle émet le tool call (TEXTE seul, pas d'audio — c'est le
    contrat d'entraînement Phase B) → on exécute + on réinjecte → Pass B ;
  - *négatif* → aucun tool call : le modèle parle directement, l'audio est
    streamé en temps réel → tour terminé en une passe.
- **Pass B** (round 1) : prompt = historique + assistant(tool call) + tool(résultat)
  → la **réponse parlée** S2S ancrée dans le résultat, streamée frame par frame.

Choix audio multi-passes : l'audio user n'est fourni qu'à la **Pass A** (chemin
prouvé : 1 placeholder en dernière position). Aux passes suivantes, le tour user
est stocké en texte ``(voice message)`` (sans audio) — la Pass B grounde sur le
RÉSULTAT d'outil (qui porte l'intention), ce qui évite le bug de placeholder vLLM
hors-dernière-position et réduit la latence.

Le backend est abstrait (``S2SStreamBackend``) → ce module est testable sans GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Protocol, Sequence

from s2s_toolcalling.data import chat_format
from s2s_toolcalling.orchestrator.events import (
    AgentEvent,
    AudioChunk,
    FillerSpeech,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from s2s_toolcalling.orchestrator.fillers import FillerBank
from s2s_toolcalling.orchestrator.tool_parser import StreamingToolCallParser
from s2s_toolcalling.tools.registry import ToolRegistry

if TYPE_CHECKING:
    import numpy as np

TOOL_CALL_START = chat_format.TOOL_CALL_START
TOOL_CALL_END = chat_format.TOOL_CALL_END
TEXT_END = "<|text_end|>"


@dataclass
class Turn:
    """Un tour de conversation rendu par le backend (``<|im_start|>role…``).

    ``audio`` (wave float32 mono, sample_rate) n'est porté que par le tour user
    COURANT (Pass A) ; mis à ``None`` ensuite (l'intention passe par le résultat).
    """

    role: str
    text: str = ""
    audio: "tuple[np.ndarray, int] | None" = None


class S2SStreamBackend(Protocol):
    """Contrat minimal attendu de l'adaptateur vLLM-Omni.

    ``stream`` rend des chunks wav (np.float32, 24 kHz) au fil de la génération et
    expose le texte généré dans ``last_text`` une fois le générateur épuisé.
    """

    system: str
    last_text: str
    tool_call_end_id: int
    im_end_id: int

    def stream(self, turns: Sequence[Turn], *, stop_token_ids: list[int]) -> Iterator["np.ndarray"]: ...


@dataclass(slots=True)
class VllmAgentConfig:
    system_instructions: str = chat_format.TOOLCALLING_EN_SYSTEM_INSTRUCTIONS
    max_tool_rounds: int = 1  # entraînement Phase B = 1 outil puis réponse


class VllmToolAgent:
    """Boucle tool-calling S2S 2-passes au-dessus d'un backend vLLM-Omni."""

    def __init__(
        self,
        backend: S2SStreamBackend,
        registry: ToolRegistry,
        *,
        config: VllmAgentConfig | None = None,
        fillers: FillerBank | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.config = config or VllmAgentConfig()
        self.fillers = fillers or FillerBank()
        # System = instructions + liste d'outils → contrat d'entraînement (Phase B).
        self.backend.system = chat_format.build_system_prompt(
            instructions=self.config.system_instructions,
            tool_definitions=registry.definitions(),
        )
        self.turns: list[Turn] = []

    def reset(self) -> None:
        self.turns = []

    def respond(self, wave: "np.ndarray", sample_rate: int) -> Iterator[AgentEvent]:
        """Tour complet : audio visiteur → événements (audio, tool calls, fin)."""
        user_turn = Turn(role="user", audio=(wave, sample_rate))
        self.turns.append(user_turn)
        audio_pending = True  # l'audio user n'est fourni qu'à la 1re passe

        for round_idx in range(self.config.max_tool_rounds + 1):
            stop = [self.backend.tool_call_end_id, self.backend.im_end_id]
            frames = 0
            for wav in self.backend.stream(self.turns, stop_token_ids=stop):
                frames += 1
                yield AudioChunk(samples=wav, sample_rate=24_000)

            if audio_pending:  # audio consommé → tours suivants en texte seul
                user_turn.audio = None
                user_turn.text = "(voice message)"
                audio_pending = False

            text = self.backend.last_text
            call = self._parse_call(text)

            if call is None:  # négatif / réponse finale : pas de tool call
                self.turns.append(Turn(role="assistant", text=text))
                yield TurnComplete(text=_visible(text), tool_rounds=round_idx, audio_frames=frames)
                return

            if round_idx == self.config.max_tool_rounds:  # garde-fou anti-boucle
                self.turns.append(Turn(role="assistant", text=text))
                yield TurnComplete(text=_visible(text), tool_rounds=round_idx, audio_frames=frames)
                return

            # Tool call : on stocke le span canonique, exécute, réinjecte le résultat.
            self.turns.append(Turn(role="assistant",
                                   text=chat_format.render_tool_calls([(call.name, call.arguments)])))
            yield ToolCallBegin(name=call.name, arguments=call.arguments)
            filler = self.fillers.get(call.name)
            yield FillerSpeech(phrase=filler.phrase, wav_path=filler.wav_path)

            result = self.registry.execute_sync(call.name, call.arguments)
            yield ToolCallResult(name=result.name, ok=result.ok, payload=result.payload(),
                                 elapsed_ms=result.elapsed_ms)
            self.turns.append(Turn(role="tool", text=chat_format.render_tool_response(result.payload())))

    # ------------------------------------------------------------------ #

    def _parse_call(self, text: str):
        """Extrait un tool call du texte de Pass A (None si aucun / malformé).

        vLLM peut stripper le stop token ``<|tool_call_end|>`` : si le span est
        ouvert sans fermeture, on l'ajoute avant de parser.
        """
        if TOOL_CALL_START not in text:
            return None
        buf = text if TOOL_CALL_END in text else text + TOOL_CALL_END
        parser = StreamingToolCallParser()
        completed = parser.feed(buf)
        if parser.errors or not completed:
            return None
        return completed[0]


def _visible(text: str) -> str:
    """Texte « parlable » : sans les marqueurs d'outil ni ``<|text_end|>``."""
    for tok in (TOOL_CALL_START, TOOL_CALL_END, TEXT_END):
        text = text.replace(tok, "")
    return text.strip()
