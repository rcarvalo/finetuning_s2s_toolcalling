"""Boucle agent S2S + tool calling (Phase 3).

Implémente la stratégie « thinking in text, speaking in audio » :

1. l'audio du visiteur entre dans le contexte (``ChatState.add_audio``) ;
2. ``generate_interleaved`` produit des tokens texte (1 élément) et des frames
   audio Mimi (8 éléments) — cf. ``liquid_audio.demo.chat`` pour la référence ;
3. le flux texte passe dans ``StreamingToolCallParser`` ; dès qu'un span
   ``<|tool_call_start|>...<|tool_call_end|>`` est complet, la génération est
   interrompue, un filler vocal est émis, l'outil est exécuté ;
4. le résultat est réinjecté dans un tour de rôle ``tool``
   (``<|tool_response_start|>...<|tool_response_end|>``) et la génération
   reprend sur un nouveau tour assistant — qui porte la réponse audio finale.

Nécessite GPU + extra ``train`` (liquid-audio). Le reste de l'orchestrateur
(parser, registre, fillers, événements) est testable sans GPU.

NOTE (risque tracé par la méthodologie) : l'émission d'un tool call en mode
interleaved n'est pas documentée par Liquid AI — ce module est aussi le banc
de dérisquage : ``max_tool_rounds``, erreurs de parsing renvoyées au modèle,
et fallback ``notify_receptionist`` si l'appel est invalide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

from s2s_toolcalling.data import chat_format
from s2s_toolcalling.orchestrator.events import (
    AgentError,
    AgentEvent,
    AudioChunk,
    FillerSpeech,
    TextDelta,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from s2s_toolcalling.orchestrator.fillers import FillerBank
from s2s_toolcalling.orchestrator.tool_parser import StreamingToolCallParser
from s2s_toolcalling.tools.registry import ToolRegistry

if TYPE_CHECKING:
    import torch
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

logger = logging.getLogger(__name__)

TEXT_END = "<|text_end|>"


@dataclass(slots=True)
class AgentConfig:
    max_new_tokens: int = 2048
    max_tool_rounds: int = 4
    text_temperature: float | None = None  # None = greedy (recommandé pour les tool calls)
    text_top_k: int | None = None
    audio_temperature: float | None = 1.0
    audio_top_k: int | None = 4
    decode_audio: bool = True
    system_instructions: str = chat_format.DEFAULT_SYSTEM_INSTRUCTIONS
    # HYBRIDE (défaut) : tour tool-call en generate_SEQUENTIAL (texte greedy
    # propre, pas d'interleave forcé qui shredderait le span) PUIS, après le
    # résultat d'outil, tour réponse en generate_INTERLEAVED (parole S2S basse
    # latence). hybrid=False → interleaved partout (parole instantanée mais tool
    # calls corrompus, cf. éval).
    hybrid: bool = True


class ReceptionAgent:
    def __init__(
        self,
        model: "LFM2AudioModel",
        processor: "LFM2AudioProcessor",
        registry: ToolRegistry,
        *,
        config: AgentConfig | None = None,
        fillers: FillerBank | None = None,
    ) -> None:
        self.model = model
        self.proc = processor
        self.registry = registry
        self.config = config or AgentConfig()
        self.fillers = fillers or FillerBank()
        self.mimi = processor.mimi.eval() if self.config.decode_audio else None

        from s2s_toolcalling.data.chat_format import verify_special_tokens

        bad = [tok for tok, ok in verify_special_tokens(processor.text).items() if not ok]
        if bad:
            raise RuntimeError(f"tokenizer is missing special tokens (not single ids): {bad}")

    # ------------------------------------------------------------------ #

    def new_session(self) -> "ChatState":
        """Nouveau contexte avec system prompt + liste d'outils (= contrat d'entraînement)."""
        from liquid_audio import ChatState

        chat = ChatState(self.proc)
        chat.new_turn("system")
        chat.add_text(
            chat_format.build_system_prompt(
                instructions=self.config.system_instructions,
                tool_definitions=self.registry.definitions(),
            )
        )
        chat.end_turn()
        return chat

    def respond(
        self,
        chat: "ChatState",
        wav: "torch.Tensor",
        sample_rate: int,
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AgentEvent]:
        """Tour complet : audio visiteur → événements (texte, audio, tool calls).

        ``should_stop`` est le hook barge-in (Phase 4) : vérifié à chaque token.
        Générateur synchrone : à exécuter dans un thread dédié côté serveur.
        """
        chat.new_turn("user")
        chat.add_audio(wav, sample_rate)
        chat.end_turn()
        chat.new_turn("assistant")

        full_text: list[str] = []
        total_audio_frames = 0
        tool_ran = False  # après un tool, le tour suivant = la réponse → on PARLE (interleaved)

        for round_idx in range(self.config.max_tool_rounds + 1):
            # HYBRIDE : tour tool-call (tool_ran=False) en sequential (texte propre) ;
            # tour réponse (après un outil) en interleaved (parole basse latence).
            speaking = tool_ran or not self.config.hybrid
            pending_calls, visible_delta_text, audio_frames, interrupted = yield from self._generate_round(
                chat, should_stop=should_stop, speaking=speaking
            )
            full_text.append(visible_delta_text)
            total_audio_frames += audio_frames

            if interrupted:
                chat.end_turn()
                break

            if not pending_calls:
                chat.end_turn()
                yield TurnComplete(text="".join(full_text).strip(), tool_rounds=round_idx, audio_frames=total_audio_frames)
                return

            if round_idx == self.config.max_tool_rounds:
                # Trop de rounds : escalade plutôt que boucle infinie.
                logger.warning("max tool rounds reached, escalating to receptionist")
                chat.end_turn()
                result = self.registry.execute_sync(
                    "notify_receptionist",
                    {"reason": "L'agent n'a pas réussi à conclure la demande du visiteur.", "urgency": "normal"},
                )
                yield ToolCallResult(name=result.name, ok=result.ok, payload=result.payload(), elapsed_ms=result.elapsed_ms)
                yield AgentError(message="max tool rounds reached", recoverable=True)
                return

            # --- Exécution des outils + réinjection rôle `tool` --------- #
            chat.end_turn()  # clôt le tour assistant texte-seul (tool call)

            payloads = []
            for call in pending_calls:
                yield ToolCallBegin(name=call.name, arguments=call.arguments)
                filler = self.fillers.get(call.name)
                yield FillerSpeech(phrase=filler.phrase, wav_path=filler.wav_path)

                result = self.registry.execute_sync(call.name, call.arguments)
                payloads.append(result.payload())
                yield ToolCallResult(name=result.name, ok=result.ok, payload=result.payload(), elapsed_ms=result.elapsed_ms)

            chat.new_turn("tool")  # rôle hors Literal amont, rendu correctement par f-string
            chat.add_text(chat_format.render_tool_response(payloads[0] if len(payloads) == 1 else payloads))
            chat.end_turn()
            chat.new_turn("assistant")
            tool_ran = True  # le prochain tour est la réponse → on parle (interleaved)

        yield TurnComplete(text="".join(full_text).strip(), tool_rounds=self.config.max_tool_rounds, audio_frames=total_audio_frames)

    # ------------------------------------------------------------------ #

    def _generate_round(self, chat: "ChatState", *, should_stop: Callable[[], bool] | None,
                        speaking: bool = False):
        """Une passe de génération ; s'arrête sur tool call complet, im_end ou barge-in.

        ``speaking`` : tour réponse → generate_interleaved (parole S2S) ; sinon
        tour tool-call → generate_sequential (texte propre). Yield les événements ;
        retourne (pending_calls, visible_text, audio_frames, interrupted).
        """
        import torch
        from liquid_audio import LFMModality

        cfg = self.config
        parser = StreamingToolCallParser()
        emitted = 0
        out_text: list[torch.Tensor] = []
        out_audio: list[torch.Tensor] = []
        out_modality: list[int] = []
        audio_frames = 0
        pending_calls: list = []
        interrupted = False

        import contextlib

        stream_ctx = self.mimi.streaming(1) if self.mimi is not None else contextlib.nullcontext()
        gen_fn = self.model.generate_interleaved if speaking else self.model.generate_sequential
        with torch.no_grad(), stream_ctx:
            for t in gen_fn(
                **chat,
                max_new_tokens=cfg.max_new_tokens,
                text_temperature=cfg.text_temperature,
                text_top_k=cfg.text_top_k,
                audio_temperature=cfg.audio_temperature,
                audio_top_k=cfg.audio_top_k,
            ):
                if should_stop is not None and should_stop():
                    interrupted = True
                    break

                if t.numel() == 1:  # token texte
                    out_text.append(t)
                    out_modality.append(int(LFMModality.TEXT))
                    piece = self.proc.text.decode(t)
                    completed = parser.feed(piece)

                    visible = parser.visible_text.replace(TEXT_END, "")
                    if len(visible) > emitted:
                        yield TextDelta(text=visible[emitted:])
                        emitted = len(visible)

                    if completed:
                        pending_calls = completed
                        break
                    if parser.errors:
                        # Appel malformé : on coupe et on laisse respond() escalader.
                        yield AgentError(message=f"tool call parse error: {parser.errors[-1]}", recoverable=True)
                        parser.errors.clear()

                elif t.numel() == 8:  # frame audio Mimi
                    out_audio.append(t)
                    out_modality.append(int(LFMModality.AUDIO_OUT))
                    audio_frames += 1
                    if self.mimi is not None and not bool((t == 2048).any()):
                        wav_chunk = self.mimi.decode(t[None, :, None])[0]
                        yield AudioChunk(samples=wav_chunk.detach().float().cpu(), sample_rate=24_000)
                else:
                    raise RuntimeError(f"unexpected token shape from generate_interleaved: {tuple(t.shape)}")

        # Réintègre les tokens générés dans le contexte (cf. demo/chat.py).
        if out_text or out_audio:
            device = chat.device
            text_t = torch.stack(out_text, 1) if out_text else torch.empty((1, 0), dtype=torch.long, device=device)
            audio_t = (
                torch.stack(out_audio, 1)
                if out_audio
                else torch.empty((chat.codebooks, 0), dtype=torch.long, device=device)
            )
            chat.append(text=text_t, audio_out=audio_t, modality_flag=torch.tensor([out_modality], device=device))

        visible_final = parser.visible_text.replace(TEXT_END, "")
        return pending_calls, visible_final, audio_frames, interrupted
