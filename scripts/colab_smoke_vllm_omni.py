#!/usr/bin/env python3
"""Smoke-test progressif du plugin vllm_omni_lfm2_audio (cible : Colab GPU).

Valide couche par couche, avec un diagnostic exploitable à chaque échec :

    1. imports     — torch / vllm / vllm-omni / liquid-audio + CUDA
    2. plugin      — entry point découvert, architecture + pipeline enregistrés
    3. contract    — le contrat runtime v0.22.0 dont dépend le plugin
                     (OmniOutput, hook sample()/prefer_model_sampler, champs
                     StagePipelineConfig) est bien présent
    4. checkpoint  — config.json + préfixes de poids du checkpoint converti
                     (--checkpoint)
    5. engine      — Omni(model=...) démarre et génère N tokens greedy
                     (--engine, GPU requis)

Usage (Colab) :
    python scripts/colab_smoke_vllm_omni.py
    python scripts/colab_smoke_vllm_omni.py --checkpoint exports/full_omni --engine

Code retour = nombre de stages en échec (0 si tout passe).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import sys
import traceback
from pathlib import Path

PLUGIN_ENTRY_POINT = "lfm2_audio"
PLUGIN_GROUP = "vllm_omni.general_plugins"
ARCHITECTURE = "Lfm2AudioOmniModel"
MODEL_TYPE = "lfm2_audio"

# Champs de StagePipelineConfig utilisés par vllm_omni_lfm2_audio/pipeline.py.
STAGE_CONFIG_FIELDS_USED = (
    "stage_id",
    "model_stage",
    "execution_type",
    "input_sources",
    "final_output",
    "final_output_type",
    "owns_tokenizer",
    "requires_multimodal_data",
    "engine_output_type",
    "sampling_constraints",
    "custom_process_input_func",
    "custom_process_next_stage_input_func",
    "async_chunk_process_next_stage_input_func",
    "sync_process_input_func",
)

# Préfixes de poids attendus dans le checkpoint converti (layout liquid-audio).
WEIGHT_PREFIXES = ("lfm.", "conformer.", "audio_adapter.")


def _fail(stage: str, msg: str) -> str:
    print(f"[FAIL] {stage}: {msg}")
    return msg


def _ok(stage: str, msg: str) -> None:
    print(f"[ ok ] {stage}: {msg}")


def check_imports() -> list[str]:
    problems: list[str] = []
    for mod in ("torch", "vllm", "vllm_omni", "liquid_audio"):
        try:
            m = importlib.import_module(mod)
            _ok("imports", f"{mod} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 — diagnostic exhaustif voulu ici
            problems.append(_fail("imports", f"{mod}: {type(exc).__name__}: {exc}"))
    if not problems:
        import torch

        if torch.cuda.is_available():
            _ok("imports", f"CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)}")
        else:
            print("[warn] imports: pas de GPU CUDA (stage engine indisponible)")
    return problems


def check_plugin() -> list[str]:
    problems: list[str] = []

    eps = {ep.name: ep for ep in importlib.metadata.entry_points(group=PLUGIN_GROUP)}
    if PLUGIN_ENTRY_POINT not in eps:
        return [
            _fail(
                "plugin",
                f"entry point {PLUGIN_ENTRY_POINT!r} absent du groupe {PLUGIN_GROUP!r} "
                f"(trouvés : {sorted(eps)}) — `pip install -e .` du repo manquant ?",
            )
        ]
    _ok("plugin", f"entry point {PLUGIN_ENTRY_POINT!r} découvert ({eps[PLUGIN_ENTRY_POINT].value})")

    try:
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        _ok("plugin", "load_omni_general_plugins() exécuté")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return problems + [_fail("plugin", f"chargement des plugins : {type(exc).__name__}: {exc}")]

    try:
        from vllm_omni.config.stage_config import _PIPELINE_REGISTRY

        if MODEL_TYPE in list(_PIPELINE_REGISTRY.keys()):
            _ok("plugin", f"pipeline {MODEL_TYPE!r} présent dans le registre")
        else:
            problems.append(_fail("plugin", f"pipeline {MODEL_TYPE!r} absent du registre après chargement"))
    except Exception as exc:  # noqa: BLE001
        problems.append(_fail("plugin", f"introspection du registre de pipelines : {exc}"))

    try:
        from vllm_omni.model_executor.models.registry import OmniModelRegistry

        registered = ARCHITECTURE in repr(vars(OmniModelRegistry)) or any(
            ARCHITECTURE in str(v) for v in vars(OmniModelRegistry).values()
        )
        if registered:
            _ok("plugin", f"architecture {ARCHITECTURE!r} visible dans OmniModelRegistry")
        else:
            problems.append(
                _fail("plugin", f"architecture {ARCHITECTURE!r} introuvable dans OmniModelRegistry")
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(_fail("plugin", f"OmniModelRegistry : {exc}"))

    return problems


def check_contract() -> list[str]:
    """Contrat runtime v0.22.0 — chaque assertion correspond à un mécanisme
    dont le plugin dépend (vérifiées dans le wheel PyPI, on re-vérifie ici
    contre la version réellement installée)."""
    problems: list[str] = []

    from vllm_omni.model_executor.models.output_templates import OmniOutput

    fields = getattr(OmniOutput, "_fields", ())
    for f in ("text_hidden_states", "multimodal_outputs"):
        if f not in fields:
            problems.append(_fail("contract", f"OmniOutput.{f} absent (champs : {fields})"))
    if not problems:
        _ok("contract", f"OmniOutput{fields}")

    # Hook sampler custom : le runner doit router vers model.sample() quand
    # prefer_model_sampler est vrai (idiome cosyvoice3/glm_tts — c'est LE
    # mécanisme qui porte notre machine à états interleaved).
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    src = inspect.getsource(GPUARModelRunner._sample)
    if "prefer_model_sampler" in src and "_sampling_metadata_for_model_sampler" in src:
        _ok("contract", "hook sample()/prefer_model_sampler présent dans GPUARModelRunner._sample")
    else:
        problems.append(
            _fail("contract", "hook prefer_model_sampler absent de GPUARModelRunner._sample (version incompatible ?)")
        )

    import dataclasses

    from vllm_omni.config.stage_config import StagePipelineConfig, register_pipeline  # noqa: F401

    available = {f.name for f in dataclasses.fields(StagePipelineConfig)}
    missing = [f for f in STAGE_CONFIG_FIELDS_USED if f not in available]
    if missing:
        problems.append(_fail("contract", f"champs StagePipelineConfig manquants : {missing}"))
    else:
        _ok("contract", "tous les champs StagePipelineConfig utilisés par pipeline.py existent")

    return problems


def check_checkpoint(checkpoint: Path) -> list[str]:
    problems: list[str] = []
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        return [_fail("checkpoint", f"{config_path} introuvable")]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("architectures") != [ARCHITECTURE] or config.get("model_type") != MODEL_TYPE:
        problems.append(
            _fail(
                "checkpoint",
                f"config.json non converti (architectures={config.get('architectures')}, "
                f"model_type={config.get('model_type')}) — lancer convert_checkpoint",
            )
        )
    for key in ("interleaved_n_text", "interleaved_n_audio", "audio_frame_token_id", "audio_eoa_token_id"):
        if key not in config:
            problems.append(_fail("checkpoint", f"clé {key!r} absente du config.json"))
    if not problems:
        _ok(
            "checkpoint",
            f"config OK (ratio {config['interleaved_n_text']}:{config['interleaved_n_audio']})",
        )

    index_path = checkpoint / "model.safetensors.index.json"
    single = checkpoint / "model.safetensors"
    names: list[str] = []
    if index_path.exists():
        names = list(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    elif single.exists():
        from safetensors import safe_open

        with safe_open(single, framework="pt") as f:
            names = list(f.keys())
    if names:
        for prefix in WEIGHT_PREFIXES:
            if not any(n.startswith(prefix) for n in names):
                problems.append(_fail("checkpoint", f"aucun poids avec le préfixe {prefix!r}"))
        if not problems:
            _ok("checkpoint", f"{len(names)} tenseurs, préfixes attendus présents")
    else:
        problems.append(_fail("checkpoint", "aucun fichier de poids safetensors trouvé"))
    return problems


def check_engine(checkpoint: Path, max_tokens: int, dtype: str) -> list[str]:
    import torch

    if not torch.cuda.is_available():
        return [_fail("engine", "GPU CUDA requis")]
    try:
        from vllm_omni import Omni

        kwargs: dict[str, Any] = {} if dtype == "auto" else {"dtype": dtype}
        # T4/GPU lents : profiling + capture > 300 s par stage.
        # enforce_eager : la capture CUDA graph du stage code2wav demande un
        # wrapper dédié (cf. mimo cuda_graph_decoder_wrapper) — hors scope smoke.
        omni = Omni(
            model=str(checkpoint),
            stage_init_timeout=1200,
            init_timeout=1800,
            enforce_eager=True,
            # les 2 stages partagent le même GPU : 0.92 par défaut → le stage 1
            # n'a plus de VRAM (vu sur T4 : 1.1 GiB libre au launch du stage 1)
            gpu_memory_utilization=0.42,
            # historique tronqué pour le sampler custom en async (cf. probe)
            async_scheduling=False,
            **kwargs,
        )
        _ok("engine", "Omni(...) initialisé — stages chargés")
        outputs = omni.generate(
            "Bonjour, qui es-tu ?",
            None,  # défauts par stage ; greedy à ajuster une fois le sampler branché
        )
        _ok("engine", f"generate() → {len(outputs)} sortie(s)")
        for out in outputs:
            print(f"        sortie : {out}")
        return []
    except Exception as exc:  # noqa: BLE001 — le but du smoke est le diagnostic complet
        traceback.print_exc()
        return [_fail("engine", f"{type(exc).__name__}: {exc}")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint converti (convert_checkpoint)")
    parser.add_argument("--engine", action="store_true", help="démarre l'engine Omni (GPU)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="auto", help="ex. float16 sur T4 (pas de bf16)")
    args = parser.parse_args()

    stages: list[tuple[str, list[str]]] = []
    stages.append(("imports", check_imports()))
    if not stages[-1][1]:
        stages.append(("plugin", check_plugin()))
        stages.append(("contract", check_contract()))
        if args.checkpoint is not None:
            stages.append(("checkpoint", check_checkpoint(args.checkpoint)))
        if args.engine:
            if args.checkpoint is None:
                stages.append(("engine", ["--engine requiert --checkpoint"]))
            else:
                stages.append(("engine", check_engine(args.checkpoint, args.max_tokens, args.dtype)))

    failed = [name for name, problems in stages if problems]
    print()
    if failed:
        print(f"RÉSULTAT : {len(failed)} stage(s) en échec : {', '.join(failed)}")
    else:
        print(f"RÉSULTAT : OK ({', '.join(name for name, _ in stages)})")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
