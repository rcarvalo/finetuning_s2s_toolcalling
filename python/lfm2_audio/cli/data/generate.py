#!/usr/bin/env python3
"""Génère le dataset texte tool-calling EN (web_search + db_query) par synthèse LLM.

Parcourt la taxonomie (outil cible × style × profondeur), demande des cas à un
LLM (Gemini par défaut, Anthropic en option), VÉRIFIE chaque cas (parser +
registre EXISTANTS), filtre la contamination vs un benchmark held-out,
déduplique, et écrit un JSONL au ``dialogue_schema`` (single-turn, utterances en
TEXTE). L'audio est ajouté ensuite par ``lfm2-synthesize-audio``.

    export GEMINI_API_KEY=...
    lfm2-generate-data --output data/tc_en_train.jsonl \
        --provider gemini --n-total 3000 \
        --held-out benchmark/toolcalling_en/cases.sample.jsonl

La logique pure (prompt, parse, verify, contamination) est dans
``lfm2_audio.data_prep.synth_dialogues`` et testée sans réseau.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from google import genai

from lfm2_audio.data_prep import synth_dialogues as sd
from lfm2_audio.tools.schemas import (
    TOOLCALLING_EN_TOOL_DEFINITIONS,
    TOOLCALLING_EN_TOOL_NAMES,
)
from lfm2_audio.tools.toolcalling_en import build_toolcalling_en_registry

# Modèle par défaut par fournisseur. Gemini Flash = très bon marché pour la
# génération en volume (cf. estimation de coût en tête de notebook).
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-6"}
API_KEY_ENV = {"gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "anthropic": ("ANTHROPIC_API_KEY",)}


def _gemini_generate_fn(model: str) -> Callable[[str], str]:
    """``generate_fn(prompt) -> str`` via le SDK Google GenAI (paresseux)."""

    client = genai.Client()  # lit GEMINI_API_KEY / GOOGLE_API_KEY

    def generate(prompt: str) -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return resp.text or ""

    return generate


def _anthropic_generate_fn(model: str) -> Callable[[str], str]:
    """``generate_fn(prompt) -> str`` via le SDK Anthropic (paresseux)."""

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
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
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
    ap.add_argument("--per-cell", type=int, default=20, help="cas demandés par appel LLM (↑ = moins d'appels)")
    ap.add_argument("--concurrency", type=int, default=8, help="appels LLM en parallèle (↓ si rate-limit 429)")
    ap.add_argument(
        "--mode",
        choices=["single", "loop"],
        default="single",
        help="single = Phase A (audio→tool call) ; loop = Phase B S2S (+ tool_result + réponse parlée ancrée)",
    )
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", default=None, help="défaut : selon --provider")
    ap.add_argument(
        "--lang",
        choices=["en", "fr"],
        default="en",
        help="langue des énoncés parlés ; les outils et leurs arguments ne changent pas",
    )
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
    weights_by = dict(sd.TOOL_TARGETS)

    seen_norm: set[str] = set()
    written = pos = rejected = contaminated = cells = 0
    alloc = {t: 0.0 for t in targets}  # état du round-robin pondéré (balance globale)

    def _next_target() -> sd.ToolTarget:
        """Round-robin pondéré : la cible la plus EN RETARD sur son poids cible.
        Garantit le mix ~35/35/30 quelle que soit la taille (≠ tirage aléatoire
        qui se déséquilibre sur les petits runs)."""
        t = min(targets, key=lambda t: (alloc[t] + 1) / weights_by[t])
        alloc[t] += 1
        return t

    def _make_cell(target: sd.ToolTarget) -> tuple[sd.ToolTarget, str, str, str, str]:
        """Construit une cellule (style/profondeur/forme aléatoires).

        La forme interrogative est tirée INDÉPENDAMMENT de la cible : c'est ce
        qui empêche « when » d'appartenir à la base de données, le raccourci
        que v4 avait appris.
        """
        style = rng.choice(sd.PHRASING_STYLES)
        depth = rng.choice(sd.INFERENCE_DEPTHS)
        form = rng.choice(sd.QUESTION_FORMS)
        prompt = sd.build_generation_prompt(
            target=target,
            style=style,
            depth=depth,
            form=form,
            n=args.per_cell,
            tool_definitions=TOOLCALLING_EN_TOOL_DEFINITIONS,
            blocklist=rng.sample(contamination.held_out, min(8, len(contamination.held_out))),
            mode=args.mode,
            language={"en": "English", "fr": "French"}[args.lang],
        )
        return target, style, depth, form, prompt

    def _safe_gen(prompt: str) -> str | Exception:
        try:
            return generate_fn(prompt)
        except Exception as e:
            return e

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Écriture AU FUR ET À MESURE : chaque cas accepté est flushé sur disque tout
    # de suite → on peut arrêter (Ctrl-C) à tout moment sans rien perdre, et
    # suivre l'avancement. Appels LLM EN PARALLÈLE par vagues de `concurrency`
    # (goulot = latence réseau). Parsing/verif/dedup séquentiel (rapide).
    with args.output.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while written < args.n_total and cells < args.max_cells:
            batch = [_make_cell(_next_target()) for _ in range(args.concurrency)]
            cells += len(batch)
            for (target, style, depth, form, _), raw in zip(
                batch, pool.map(_safe_gen, [c[4] for c in batch]), strict=True
            ):
                if isinstance(raw, Exception):
                    print(f"[appel échoué] {raw}", file=sys.stderr)
                    continue
                for case in sd.parse_generation_response(
                    raw, target=target, style=style, depth=depth, form=form, mode=args.mode
                ):
                    if written >= args.n_total:
                        break
                    if sd.verify_case(case, registry, mode=args.mode):
                        rejected += 1
                        continue
                    key = sd._normalize(case.utterance)
                    if key in seen_norm:
                        continue
                    if contamination.is_contaminated(case.utterance):
                        contaminated += 1
                        continue
                    seen_norm.add(key)
                    dialogue = sd.case_to_dialogue(
                        case,
                        written,
                        tools=TOOLCALLING_EN_TOOL_NAMES,
                        mode=args.mode,
                        lang=args.lang,
                        prefix=f"tc{args.lang}" if args.lang != "en" else "tc",
                    )
                    out.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
                    written += 1
                    pos += case.target != "none"
            out.flush()
            print(
                f"[{cells} appels] écrits={written}/{args.n_total} "
                f"(pos={pos} neg={written - pos}) rejetés={rejected} contaminés={contaminated}",
                flush=True,
            )

    print(
        f"\nécrit {args.output} : {written} cas "
        f"({pos} positifs / {written - pos} négatifs) — "
        f"rejetés={rejected}, contaminés={contaminated}, appels={cells}"
    )
    print(f"TTS ensuite : lfm2-synthesize-audio --dialogues {args.output} --audio-root data/audio_tc_en")


if __name__ == "__main__":
    main()
