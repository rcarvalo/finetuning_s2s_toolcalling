"""Machine à états de modalité interleaved — port fidèle de
``liquid_audio.model.lfm2_audio.LFM2AudioModel.generate_interleaved``.

Python pur (aucun torch) : la modalité du PROCHAIN step se déduit
déterministiquement de la séquence d'ids du tour assistant courant, frames
audio représentées par leurs placeholders (cf. ``constants``). Cette pureté
est ce qui rend l'état rejouable après une préemption du scheduler vLLM et
compatible avec le prefix caching.

Sémantique portée (y compris ses subtilités, vérifiées dans le source) :

- état initial d'un tour assistant : TEXT, ``left = n_text``, ``text_done=False`` ;
- à chaque step, ``left`` est décrémenté PUIS le token est produit ;
- TEXT : ``<|text_end|>`` → ``text_done = True`` ; après le token, si
  ``left == 0`` ou ``text_done`` → bascule AUDIO avec ``left = n_audio`` ;
- AUDIO : après la frame, si ``left == 0`` et ``not text_done`` → bascule TEXT
  avec ``left = n_text`` ; une frame EOA (code 2048) bascule TEXT **sans
  réinitialiser ``left``** (quirk du code amont : ``left`` devient négatif et
  seule ``<|text_end|>`` re-déclenchera l'audio) — reproduit à l'identique ;
- ``<|im_end|>`` termine le tour (géré par les stop tokens du stage).

Extension hors-amont — **verrou tool-call** : un appel d'outil
(``<|tool_call_start|>``…``<|tool_call_end|>``) est TEXTE seul (contrat
d'entraînement Phase B : le tour tool-call est généré en *sequential*, sans
audio). Si ``tool_call_start_id`` est configuré, ce token arme un état qui
SUPPRIME la bascule audio périodique (le ``left == 0``) jusqu'à
``<|tool_call_end|>`` — sinon l'interleaving 6:12 hache le span structuré
``[fn(arg="…")]`` (token-salad ``<|audio_start|>``×12 au milieu de l'appel,
observé sur Colab). Inerte quand les ids d'outil valent ``None`` (parité amont
stricte) ; comme le reste de la machine, c'est une fonction pure du flux d'ids
(rejouable après préemption, compatible prefix caching).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from lfm2_audio.vllm_plugin.constants import (
    AUDIO_EOA_PLACEHOLDER_ID,
    AUDIO_FRAME_PLACEHOLDER_ID,
    DEFAULT_INTERLEAVED_N_AUDIO,
    DEFAULT_INTERLEAVED_N_TEXT,
    IM_END_TOKEN_ID,
    TEXT_END_TOKEN_ID,
)


class Modality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"


@dataclass(slots=True)
class ModalityConfig:
    n_text: int = DEFAULT_INTERLEAVED_N_TEXT
    n_audio: int = DEFAULT_INTERLEAVED_N_AUDIO
    text_end_id: int = TEXT_END_TOKEN_ID
    im_end_id: int = IM_END_TOKEN_ID
    frame_placeholder_id: int = AUDIO_FRAME_PLACEHOLDER_ID
    eoa_placeholder_id: int = AUDIO_EOA_PLACEHOLDER_ID
    # Verrou tool-call (texte-only). ``None`` → inerte (parité amont stricte).
    tool_call_start_id: int | None = None
    tool_call_end_id: int | None = None
    # Tour entier en texte seul — l'équivalent vLLM de ``generate_sequential``,
    # que la tâche ASR officielle du modèle exige (« Perform ASR. »). Sans lui,
    # l'interleaving force des frames audio tous les ``n_text`` tokens et hache
    # la transcription, comme il hachait les spans de tool call.
    # ``False`` → parité stricte avec le comportement interleavé existant.
    text_only: bool = False


@dataclass(slots=True)
class ModalityState:
    """État APRÈS avoir consommé un préfixe du tour assistant.

    ``current``/``left`` décrivent la modalité du prochain step à générer
    (``left`` = budget restant AVANT décrément, comme dans le code amont).
    """

    current: Modality = Modality.TEXT
    left: int = DEFAULT_INTERLEAVED_N_TEXT
    text_done: bool = False
    finished: bool = False
    in_tool_call: bool = False  # texte-only forcé : appel d'outil non encore fermé

    def is_audio_placeholder(self, token_id: int, cfg: ModalityConfig) -> bool:
        return token_id in (cfg.frame_placeholder_id, cfg.eoa_placeholder_id)


def initial_state(cfg: ModalityConfig) -> ModalityState:
    return ModalityState(current=Modality.TEXT, left=cfg.n_text, text_done=False, finished=False)


def advance(state: ModalityState, token_id: int, cfg: ModalityConfig) -> ModalityState:
    """Consomme un token du tour assistant et retourne le nouvel état.

    Reproduit exactement l'ordre des opérations de ``generate_interleaved``.
    """
    if state.finished:
        return state

    current, left, text_done, in_tool_call = state.current, state.left, state.text_done, state.in_tool_call
    left -= 1  # décrément en tête de boucle, comme amont

    if current is Modality.TEXT:
        if token_id == cfg.im_end_id:
            return ModalityState(
                current=current, left=left, text_done=text_done, finished=True, in_tool_call=in_tool_call
            )
        if state.is_audio_placeholder(token_id, cfg):
            raise ValueError("audio placeholder encountered during a TEXT step — id stream is inconsistent")
        if token_id == cfg.text_end_id:
            text_done = True
        # Verrou tool-call : `<|tool_call_start|>` arme le texte-only, `<|tool_call_end|>`
        # le lève (un appel d'outil n'émet jamais `<|text_end|>` ni de frame audio).
        # À la fermeture on réarme un budget texte frais : sinon un `left` résiduel
        # à 0 (appel pile à n_text) ferait basculer en audio juste après l'appel.
        if cfg.tool_call_start_id is not None and token_id == cfg.tool_call_start_id:
            in_tool_call = True
        elif cfg.tool_call_end_id is not None and token_id == cfg.tool_call_end_id:
            in_tool_call = False
            left = cfg.n_text
        # Bascule audio supprimée tant qu'un appel d'outil est ouvert : le span
        # structuré reste 100 % texte (sinon l'interleaving 6:12 le hache).
        if (left == 0 or text_done) and not in_tool_call and not cfg.text_only:
            current, left = Modality.AUDIO, cfg.n_audio
    else:  # AUDIO
        if not state.is_audio_placeholder(token_id, cfg):
            raise ValueError("text token encountered during an AUDIO step — id stream is inconsistent")
        # Ordre amont : la bascule fin-de-bloc est évaluée AVANT le test EOA.
        if left == 0 and not text_done:
            current, left = Modality.TEXT, cfg.n_text
        if token_id == cfg.eoa_placeholder_id:
            current = Modality.TEXT  # quirk amont : left N'EST PAS réinitialisé

    return ModalityState(current=current, left=left, text_done=text_done, finished=False, in_tool_call=in_tool_call)


def replay(turn_token_ids: Iterable[int], cfg: ModalityConfig) -> ModalityState:
    """Rejoue le tour assistant complet → état courant (fonction pure)."""
    state = initial_state(cfg)
    for token_id in turn_token_ids:
        state = advance(state, token_id, cfg)
    return state


def next_step_modality(turn_token_ids: Iterable[int], cfg: ModalityConfig) -> Modality:
    """Modalité du prochain token à générer pour ce tour assistant."""
    return replay(turn_token_ids, cfg).current


def expected_modalities(turn_token_ids: Iterable[int], cfg: ModalityConfig) -> list[Modality]:
    """Modalité de CHAQUE position du tour (diagnostic / vérification prefill)."""
    out: list[Modality] = []
    state = initial_state(cfg)
    for token_id in turn_token_ids:
        out.append(state.current)
        state = advance(state, token_id, cfg)
    return out
