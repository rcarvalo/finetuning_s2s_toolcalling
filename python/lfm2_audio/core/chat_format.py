"""Format ChatML LFM2.5 pour le tool calling.

Le backbone LFM2.5 (partagé par LFM2.5-Audio via son tokenizer) définit des
tokens spéciaux pour le tool calling :

- ``<|tool_list_start|>`` / ``<|tool_list_end|>`` : définitions JSON des outils,
  placées dans le system prompt ;
- ``<|tool_call_start|>`` / ``<|tool_call_end|>`` : appel "pythonic"
  (``[fn(arg="val")]``) émis par l'assistant dans le flux **texte** ;
- ``<|tool_response_start|>`` / ``<|tool_response_end|>`` : résultat d'exécution,
  réinjecté dans un tour de rôle ``tool``.

Ce module est en Python pur (aucune dépendance torch/liquid-audio) : il est
partagé entre la préparation de données (Phase 2) et l'orchestrateur (Phase 3).
"""

from __future__ import annotations

import json
import re
from typing import Any

from lfm2_audio.core.tokenizer import Tokenizer

TOOL_LIST_START = "<|tool_list_start|>"
TOOL_LIST_END = "<|tool_list_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_RESPONSE_START = "<|tool_response_start|>"
TOOL_RESPONSE_END = "<|tool_response_end|>"

# The WHOLE call span, markers included. Shared by the judge — which must not
# hand a tool call to a rubric written for speech — and by the failure
# diagnosis, which quotes the offending span. One definition, two readers.
TOOL_CALL_SPAN = re.compile(
    re.escape(TOOL_CALL_START) + r".*?" + re.escape(TOOL_CALL_END),
    flags=re.DOTALL,
)

SPECIAL_TOOL_TOKENS = (
    TOOL_LIST_START,
    TOOL_LIST_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
)

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "Tu es l'assistant vocal d'accueil de l'entreprise, embarqué dans le robot "
    "Reachy Mini. Tu accueilles les visiteurs en français, de façon chaleureuse "
    "et concise (réponses orales courtes). Tu disposes d'outils pour vérifier "
    "les rendez-vous, prévenir un collaborateur, orienter le visiteur, donner "
    "le wifi invité et interroger la base interne. N'appelle un outil que si "
    "c'est nécessaire pour répondre ; si tu ne peux pas aider, préviens le ou "
    "la réceptionniste avec notify_receptionist."
)

# Consigne anglaise pour la capacité tool calling vocal (web_search + db_query).
# Insiste sur le routage des outils (interne vs web) et l'abstention (négatifs) :
# c'est exactement ce que mesure relevance_accuracy dans eval_toolcalling.
TOOLCALLING_EN_SYSTEM_INSTRUCTIONS = (
    "You are a voice assistant that can call tools. Routing rules: use web_search for "
    "anything PUBLIC or CURRENT — weather, news, prices, sports, definitions, facts about "
    "the world. Use db_query ONLY for our INTERNAL company data (our customers, orders, "
    "products, employees, meetings). Never use db_query for public information like weather "
    "or news. Call at most one tool, and only when needed; for greetings or chit-chat, reply "
    "directly without any tool."
)


def render_python_literal(value: Any) -> str:  # noqa: ANN401 — littéral JSON quelconque
    """Rend une valeur en littéral pythonic (style LFM2 : strings JSON double-quotées,
    booléens/None en Python)."""
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_python_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{json.dumps(str(k), ensure_ascii=False)}: {render_python_literal(v)}" for k, v in value.items()
        )
        return "{" + items + "}"
    raise TypeError(f"unsupported literal type for tool argument: {type(value)!r}")


def render_tool_call(name: str, arguments: dict[str, Any] | None = None) -> str:
    """``check_appointment(visitor_name="Marie Dupont")`` (sans crochets ni marqueurs)."""
    args = ", ".join(f"{k}={render_python_literal(v)}" for k, v in (arguments or {}).items())
    return f"{name}({args})"


def render_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    """Bloc complet d'appels : ``<|tool_call_start|>[fn(...), fn2(...)]<|tool_call_end|>``."""
    inner = ", ".join(render_tool_call(name, args) for name, args in calls)
    return f"{TOOL_CALL_START}[{inner}]{TOOL_CALL_END}"


def render_tool_list(tool_definitions: list[dict[str, Any]]) -> str:
    """Bloc des définitions d'outils (JSON compact) pour le system prompt."""
    payload = json.dumps(tool_definitions, ensure_ascii=False, separators=(",", ":"))
    return f"{TOOL_LIST_START}{payload}{TOOL_LIST_END}"


def render_tool_response(payload: Any) -> str:  # noqa: ANN401 — payload JSON d'outil
    """Bloc résultat d'outil (JSON) pour un tour de rôle ``tool``."""
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{TOOL_RESPONSE_START}{body}{TOOL_RESPONSE_END}"


def build_system_prompt(
    instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> str:
    """System prompt complet : instructions + liste d'outils (si fournie)."""
    parts = [instructions]
    if tool_definitions:
        parts.append("Liste des outils disponibles : " + render_tool_list(tool_definitions))
    return "\n".join(parts)


def verify_special_tokens(tokenizer: Tokenizer) -> dict[str, bool]:
    """Vérifie que chaque marqueur d'outil est bien un token unique du tokenizer.

    À exécuter avant tout entraînement : si un marqueur se tokenise en plusieurs
    morceaux, le vocabulaire chargé n'est pas celui de LFM2.5 et la supervision
    des tool calls serait silencieusement dégradée.
    """
    result: dict[str, bool] = {}
    for token in (*SPECIAL_TOOL_TOKENS, "<|audio_start|>", "<|text_end|>", "<|im_end|>"):
        ids = tokenizer.encode(token, add_special_tokens=False)
        result[token] = len(ids) == 1
    return result
