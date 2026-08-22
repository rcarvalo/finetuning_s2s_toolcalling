#!/usr/bin/env python3
"""Harnais d'éval AUDIO → tool-call (le maillon manquant de la chaîne d'éval).

Pour chaque cas held-out ``{id, audio, expected_calls}``, exécute le modèle sur
l'AUDIO, capture le flux TEXTE généré (qui contient le span
``<|tool_call_start|>…<|tool_call_end|>``), et écrit le JSONL
``{id, expected_calls, predicted_text}`` que ``eval_toolcalling`` score
(parse/relevance/name/call). Rapporte directement le résumé.

    lfm2-eval-audio --backend vllm \
        --checkpoint exports/lfm25_tc_en --cases benchmark/toolcalling_en/cases.jsonl \
        --audio-root data/audio_tc_en --out eval_tc_en.jsonl --arg-match token_f1

La fonction de prédiction est injectable (``run_audio_eval``) → testable avec un
modèle mocké, sans GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lfm2_audio.core.chat_format import (
    TOOLCALLING_EN_SYSTEM_INSTRUCTIONS,
    build_system_prompt,
)
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.ds.generation_config import GenerationConfig
from lfm2_audio.ds.inference_config import EngineConfig
from lfm2_audio.evaluation.audio_harness import PredictFn, load_cases, run_audio_eval
from lfm2_audio.serving import LFM2Audio
from lfm2_audio.tools.schemas import TOOLCALLING_EN_TOOL_DEFINITIONS


def _build_backend_predict_fn(args: argparse.Namespace) -> PredictFn:
    """Fonction de prédiction adossée à un modèle réel.

    Le system prompt est celui de l'entraînement (instructions + liste d'outils) :
    l'évaluer avec un autre prompt mesurerait autre chose que ce qui a été appris.
    """

    audio_root = Path(args.audio_root)
    model = LFM2Audio.from_pretrained(
        args.checkpoint,
        backend=args.backend,
        adapter=args.adapter,
        system=build_system_prompt(TOOLCALLING_EN_SYSTEM_INSTRUCTIONS, TOOLCALLING_EN_TOOL_DEFINITIONS),
        engine=EngineConfig(deploy_config=None) if args.no_deploy_config else EngineConfig(),
        # Un tool call est du texte pur : l'interleave audio shredderait le span
        # `[fn(arg="…")]`. Sans effet côté vLLM, où il sort déjà en texte.
        generation=GenerationConfig(text_only=args.backend == "liquid"),
    )

    def predict(case: dict[str, Any]) -> str:
        model.reset()
        # raw_text : les marqueurs <|tool_call_*|> doivent survivre au scoring.
        model.reply(audio=Waveform.from_file(audio_root / case["audio"]))
        return model.last_text

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="JSONL held-out {id, audio, expected_calls}")
    ap.add_argument("--audio-root", default="data/audio_tc_en")
    ap.add_argument("--out", required=True, help="JSONL prédictions (consommable par eval_toolcalling)")
    ap.add_argument("--backend", choices=["auto", "liquid", "vllm"], default="vllm")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--adapter", default=None, help="dir d'adaptateur LoRA à fusionner (backend liquid) — éval d'un fine-tune"
    )
    ap.add_argument("--no-deploy-config", action="store_true")
    ap.add_argument("--arg-match", choices=["exact", "token_f1", "semantic"], default="token_f1")
    ap.add_argument("--arg-threshold", type=float, default=0.7)
    args = ap.parse_args()

    cases = load_cases(args.cases)
    predict_fn = _build_backend_predict_fn(args)
    predictions, report = run_audio_eval(cases, predict_fn, arg_match=args.arg_match, threshold=args.arg_threshold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(json.dumps({"summary": report.summary()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
