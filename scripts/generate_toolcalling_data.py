#!/usr/bin/env python3
"""Génère le dataset texte tool-calling EN (web_search + db_query) par synthèse LLM.

Parcourt la taxonomie (outil cible × style × profondeur), demande des cas à un
LLM (Gemini par défaut, Anthropic en option), VÉRIFIE chaque cas (parser +
registre EXISTANTS), filtre la contamination vs un benchmark held-out,
déduplique, et écrit un JSONL au ``dialogue_schema`` (single-turn, utterances en
TEXTE). L'audio est ajouté ensuite par ``scripts/synthesize_user_audio.py``.

    export GEMINI_API_KEY=...
    python scripts/generate_toolcalling_data.py --output data/tc_en_train.jsonl \
        --provider gemini --n-total 3000 \
        --held-out benchmark/toolcalling_en/cases.sample.jsonl

La logique pure (prompt, parse, verify, contamination) est dans
``s2s_toolcalling.data.synth_dialogues`` et testée sans réseau.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

from s2s_toolcalling.data import synth_dialogues as sd
from s2s_toolcalling.tools.schemas import (
    TOOLCALLING_EN_TOOL_DEFINITIONS,
    TOOLCALLING_EN_TOOL_NAMES,
)
from s2s_toolcalling.tools.toolcalling_en import build_toolcalling_en_registry

# Modèle par défaut par fournisseur. Gemini Flash = très bon marché pour la
# génération en volume (cf. estimation de coût en tête de notebook).
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-6"}
API_KEY_ENV = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "anthropic": ("ANTHROPIC_API_KEY",)}


def _gemini_generate_fn(model: str) -> Callable[[str], str]:
    """``generate_fn(prompt) -> str`` via le SDK Google GenAI (paresseux)."""
    from google import genai

    client = genai.Client()  # lit GEMINI_API_KEY / GOOGLE_API_KEY

    def generate(prompt: str) -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return generate


def _anthropic_generate_fn(model: str) -> Callable[[str], str]:
    """``generate_fn(prompt) -> str`` via le SDK Anthropic (paresseux)."""
    import anthropic

    client = anthropic.Anthropic()  # lit ANTHROPIC_API_KEY

    def generate(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    return generate


def _build_generate_fn(provider: str, model: str) -> Callable[[str], str]:
    if provider == "gemini":
        return _gemini_generate_fn(model)
    return _anthropic_generate_fn(model)


def _load_held_out(path: Path | None) -> list[str]:
    """Utterances du benchmark held-out (pour le filtre anti-contamination)."""
    if path is None or not path.exists():
        return []
    utts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # accepte soit {utterance:...} soit un dialogue {turns:[{role:user,text:...}]}
        if "utterance" in obj:
            utts.append(obj["utterance"])
        else:
            for t in obj.get("turns", []):
                if t.get("role") == "user" and t.get("text"):
                    utts.append(t["text"])
    return utts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n-total", type=int, default=3000, help="cas VÉRIFIÉS visés")
    ap.add_argument("--per-cell", type=int, default=12, help="cas demandés par appel LLM")
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", default=None, help="défaut : selon --provider")
    ap.add_argument("--held-out", type=Path, default=None, help="benchmark JSONL à éviter (contamination)")
    ap.add_argument("--contamination-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cells", type=int, default=10_000, help="garde-fou (coût)")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    if not any(os.environ.get(k) for k in API_KEY_ENV[args.provider]):
        print(f"ERREUR : clé API absente ({' ou '.join(API_KEY_ENV[args.provider])}).", file=sys.stderr)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    registry = build_toolcalling_en_registry()
    contamination = sd.ContaminationFilter(
        held_out=_load_held_out(args.held_out), threshold=args.contamination_threshold
    )
    generate_fn = _build_generate_fn(args.provider, model)

    targets = [t for t, _ in sd.TOOL_TARGETS]
    weights = [w for _, w in sd.TOOL_TARGETS]

    accepted: list[sd.SynthCase] = []
    seen_norm: set[str] = set()
    rejected = contaminated = cells = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    while len(accepted) < args.n_total and cells < args.max_cells:
        cells += 1
        target = rng.choices(targets, weights=weights, k=1)[0]
        style = rng.choice(sd.PHRASING_STYLES)
        depth = rng.choice(sd.INFERENCE_DEPTHS)
        prompt = sd.build_generation_prompt(
            target=target, style=style, depth=depth, n=args.per_cell,
            tool_definitions=TOOLCALLING_EN_TOOL_DEFINITIONS,
            blocklist=rng.sample(contamination.held_out, min(8, len(contamination.held_out))),
        )
        try:
            raw = generate_fn(prompt)
        except Exception as e:  # noqa: BLE001 — un appel raté ne doit pas tout arrêter
            print(f"[cell {cells}] appel LLM échoué : {e}", file=sys.stderr)
            continue

        for case in sd.parse_generation_response(raw, target=target, style=style, depth=depth):
            reason = sd.verify_case(case, registry)
            if reason:
                rejected += 1
                continue
            key = sd._normalize(case.utterance)
            if key in seen_norm:
                continue
            if contamination.is_contaminated(case.utterance):
                contaminated += 1
                continue
            seen_norm.add(key)
            accepted.append(case)

        if cells % 10 == 0:
            print(f"[cell {cells}] acceptés={len(accepted)} rejetés={rejected} contaminés={contaminated}", flush=True)

    with args.output.open("w", encoding="utf-8") as f:
        for i, case in enumerate(accepted):
            dialogue = sd.case_to_dialogue(case, i, tools=TOOLCALLING_EN_TOOL_NAMES)
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")

    pos = sum(1 for c in accepted if c.target != "none")
    print(f"\nécrit {args.output} : {len(accepted)} cas "
          f"({pos} positifs / {len(accepted) - pos} négatifs) — "
          f"rejetés={rejected}, contaminés={contaminated}, cellules={cells}")
    print("TTS ensuite : python scripts/synthesize_user_audio.py "
          f"--dialogues {args.output} --audio-root data/audio_tc_en")


if __name__ == "__main__":
    main()
