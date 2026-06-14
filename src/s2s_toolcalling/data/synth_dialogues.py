"""Synthèse de dialogues tool-calling EN (Phase A, v1 single-turn).

Génère des cas ``(utterance_user, cible)`` pour ``web_search`` + ``db_query``,
calqué sur la méthodo du cookbook ``home-assistant`` :

- **taxonomie** équilibrée (outil cible × style de formulation × profondeur
  d'inférence) pour couvrir l'espace des entrées ;
- **vérification** de chaque cas via le parser + le registre EXISTANTS (contrat
  unique entraînement/inférence) → un cas non parsable ou hors-schéma est rejeté ;
- **filtre anti-contamination** (Jaccard trigram) contre un benchmark held-out,
  pour que les métriques restent honnêtes.

L'appel LLM lui-même est injecté (``generate_fn``) : tout ici est Python pur et
testable sans réseau ni GPU. Les utterances sont en TEXTE ; l'audio est ajouté
ensuite par ``scripts/synthesize_user_audio.py`` (TTS).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from s2s_toolcalling.data.chat_format import render_tool_calls
from s2s_toolcalling.orchestrator.tool_parser import StreamingToolCallParser
from s2s_toolcalling.tools.registry import ToolRegistry

ToolTarget = Literal["web_search", "db_query", "none"]

# Taxonomie (pondérations conformes au README : 20-30 % de négatifs).
TOOL_TARGETS: list[tuple[ToolTarget, float]] = [
    ("web_search", 0.35),
    ("db_query", 0.35),
    ("none", 0.30),
]
PHRASING_STYLES = ["direct command", "polite question", "indirect request", "with disfluency"]
INFERENCE_DEPTHS = ["explicit arguments", "requires inference"]

# Réponses négatives « à trous » (ex. « It's [current time]. ») = mauvaise cible.
_PLACEHOLDER = re.compile(r"\[[^\]]+\]")


@dataclass(slots=True)
class SynthCase:
    """Un cas vérifié : utterance + cible (tool call OU réponse sans appel)."""

    utterance: str
    target: ToolTarget
    arguments: dict[str, Any] = field(default_factory=dict)  # si target != "none"
    answer: str | None = None  # réponse normale si target == "none"
    style: str = ""
    depth: str = ""


# --------------------------------------------------------------------------- #
# Prompt LLM + parsing de la réponse (contrat JSON)
# --------------------------------------------------------------------------- #


def build_generation_prompt(
    *,
    target: ToolTarget,
    style: str,
    depth: str,
    n: int,
    tool_definitions: list[dict],
    blocklist: Iterable[str] = (),
) -> str:
    """Prompt demandant ``n`` cas pour une cellule de taxonomie (sortie JSON strict)."""
    tools_json = json.dumps(tool_definitions, ensure_ascii=False, indent=2)
    block = "\n".join(f"- {u}" for u in blocklist)
    if target == "none":
        target_spec = (
            'Generate NEGATIVE cases: the user says something that needs NO tool '
            '(greeting, chit-chat, or a question answerable directly). For each, give '
            'a short natural spoken "answer". Set "tool" to "none".'
        )
        shape = '{"utterance": "...", "tool": "none", "answer": "short spoken reply"}'
    else:
        target_spec = (
            f'Generate cases where the correct action is to call "{target}". For each, '
            f'give the exact "arguments" object for that tool.'
        )
        shape = f'{{"utterance": "...", "tool": "{target}", "arguments": {{...}}}}'
    return (
        "You generate training data for a voice assistant that calls tools.\n"
        f"Available tools (JSON schemas):\n{tools_json}\n\n"
        f"{target_spec}\n"
        f'Phrasing style: "{style}". Inference depth: "{depth}".\n'
        f"Produce exactly {n} DIVERSE, realistic English user utterances as spoken to a "
        "voice assistant (no markup, no tool tokens).\n"
        + (f"Avoid anything close to these held-out utterances:\n{block}\n" if block else "")
        + "Respond with ONLY a JSON array, each element shaped like:\n"
        f"{shape}\n"
    )


def parse_generation_response(text: str, *, target: ToolTarget, style: str, depth: str) -> list[SynthCase]:
    """Parse la réponse JSON du LLM en ``SynthCase`` (tolère un éventuel fence ```)."""
    payload = _extract_json_array(text)
    cases: list[SynthCase] = []
    for item in payload:
        utt = str(item.get("utterance", "")).strip()
        if not utt:
            continue
        tool = item.get("tool", target)
        if tool == "none":
            cases.append(SynthCase(utterance=utt, target="none", answer=str(item.get("answer", "")).strip(),
                                   style=style, depth=depth))
        else:
            cases.append(SynthCase(utterance=utt, target=tool, arguments=dict(item.get("arguments", {})),
                                   style=style, depth=depth))
    return cases


def _extract_json_array(text: str) -> list[dict]:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s[3:] else s
        s = s.removeprefix("json").strip().strip("`").strip()
    start, end = s.find("["), s.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


# --------------------------------------------------------------------------- #
# Vérification (réutilise le parser + le registre)
# --------------------------------------------------------------------------- #


def verify_case(case: SynthCase, registry: ToolRegistry) -> str | None:
    """Retourne un motif de rejet, ou None si le cas est valide.

    Pour un positif : le call doit (1) se rendre en pythonic puis re-parser à
    l'identique, et (2) passer ``registry.validate`` (nom/required/args connus).
    Pour un négatif : il faut une réponse non vide et aucun argument d'outil.
    """
    if not case.utterance.strip():
        return "empty utterance"

    if case.target == "none":
        if case.arguments:
            return "negative case must not carry tool arguments"
        if not (case.answer or "").strip():
            return "negative case needs a non-empty answer"
        # Rejette les réponses à trous (« It's [current time]. ») : ce sont des
        # questions temps-réel sans outil dédié → mauvaise cible d'entraînement.
        if _PLACEHOLDER.search(case.answer or ""):
            return "negative answer contains a [placeholder]"
        return None

    if case.target not in registry:
        return f"unknown tool: {case.target}"

    # round-trip : render pythonic → re-parse → mêmes nom + arguments.
    rendered = render_tool_calls([(case.target, case.arguments)])
    parser = StreamingToolCallParser()
    parsed = parser.feed(rendered)
    if parser.errors or len(parsed) != 1:
        return f"tool call does not round-trip: {parser.errors or parsed}"
    if parsed[0].name != case.target or parsed[0].arguments != case.arguments:
        return "round-trip mismatch (name/args)"

    return registry.validate(case.target, case.arguments)


# --------------------------------------------------------------------------- #
# Filtre anti-contamination (Jaccard trigram caractères)
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


def _char_trigrams(text: str) -> set[str]:
    s = _normalize(text)
    return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


def jaccard_trigram(a: str, b: str) -> float:
    ta, tb = _char_trigrams(a), _char_trigrams(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / len(ta | tb)


@dataclass
class ContaminationFilter:
    """Rejette une utterance trop proche d'un benchmark held-out (Jaccard > seuil)."""

    held_out: list[str] = field(default_factory=list)
    threshold: float = 0.5
    _trigrams: list[set[str]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._trigrams = [_char_trigrams(u) for u in self.held_out]

    def is_contaminated(self, utterance: str) -> bool:
        tu = _char_trigrams(utterance)
        if not tu:
            return False
        for tb in self._trigrams:
            inter = len(tu & tb)
            if inter and inter / len(tu | tb) > self.threshold:
                return True
        return False


# --------------------------------------------------------------------------- #
# Cas → dialogue_schema (single-turn)
# --------------------------------------------------------------------------- #


def case_to_dialogue(case: SynthCase, idx: int, *, tools: list[str]) -> dict[str, Any]:
    """Dialogue single-turn : user (texte, audio ajouté ensuite) → assistant."""
    if case.target == "none":
        assistant = {"role": "assistant", "text": case.answer}
    else:
        assistant = {"role": "assistant", "tool_calls": [{"name": case.target, "arguments": case.arguments}]}
    return {
        "id": f"tc_{idx:06d}_{case.target}",
        "tools": tools,
        "meta": {"style": case.style, "depth": case.depth, "target": case.target},
        "turns": [
            {"role": "user", "text": case.utterance},
            assistant,
        ],
    }


def dedup_cases(cases: Iterable[SynthCase]) -> list[SynthCase]:
    """Déduplique sur l'utterance normalisée (garde le premier)."""
    seen: set[str] = set()
    out: list[SynthCase] = []
    for c in cases:
        key = _normalize(c.utterance)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
