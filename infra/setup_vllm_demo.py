"""Installer la pile vLLM-Omni de la démo tool-calling sur une VM Colab (L4).

Recette **vérifiée couche par couche** le 26/08/2026 sur une image vierge
(Python 3.13, torch 2.11, CUDA 12). Elle encode trois pièges qui ont coûté une
journée :

1. **Jamais de shell.** Les specifiers pip contiennent ``<`` : passé à bash,
   ``vllm-omni>=0.22,<0.23`` devient une redirection depuis un fichier « 0.23 »
   et l'installation échoue *en silence*. Tout passe par ``subprocess`` sans
   shell.

2. **Retirer torchao.** ``peft`` exige ``torchao>=0.16`` dès qu'on touche au
   chemin LoRA (fusion d'adaptateur), mais ces versions embarquent des
   extensions compilées pour CPython 3.10 : sur l'image 3.13 elles font
   **segfaulter** le processus (rc=-11, aucune trace Python). Absent, torchao
   n'est plus contrôlé — et aucun de nos chemins ne l'utilise.

3. **vLLM-Omni impose son propre ensemble** (diffusers 0.38, transformers
   5.8.1) et il est cohérent : ne rien y « corriger ». Les conflits apparents
   peft/torchao/diffusers observés avant cette recette venaient tous de
   l'étape 1 ratée, pas d'une incompatibilité réelle.

Ordre d'import à respecter côté exécution : le module de démo se charge
**avant** le moteur (c'est le cas naturel, la démo étant le point d'entrée).
L'ordre inverse segfaute.

    python infra/setup_vllm_demo.py --repo /content/repo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

VLLM_WHEEL = (
    "vllm @ https://github.com/vllm-project/vllm/releases/download/"
    "v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
CU129_INDEX = "https://download.pytorch.org/whl/cu129"
DEMO_EXTRAS = ["fastrtc", "gradio", "ddgs", "tavily-python", "qwen-tts"]


def pip(*args: str) -> list[str]:
    return [sys.executable, "-m", "pip", *args]


def run(step: list[str], label: str) -> None:
    print(f"\n=== {label}", flush=True)
    code = subprocess.run(step, check=False).returncode
    if code != 0:
        raise SystemExit(f"échec : {label} (rc={code})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="/content/repo", type=Path)
    args = parser.parse_args()

    run(pip("install", VLLM_WHEEL, "--extra-index-url", CU129_INDEX), "vLLM 0.22.1+cu129")
    run(pip("install", "vllm-omni>=0.22,<0.23"), "vLLM-Omni (sans shell : le '<' est littéral)")
    run(pip("install", "-e", f"{args.repo}[serving]"), "lfm2-audio[serving]")
    run(pip("install", *DEMO_EXTRAS), "extras démo (UI voix, outils, TTS fillers)")
    # Désinstallation tolérante : torchao peut déjà être absent.
    subprocess.run(pip("uninstall", "-y", "torchao"), check=False)
    print("\n=== torchao retiré (segfault en 3.13, cf. docstring)", flush=True)

    verify = (
        "from lfm2_audio.cli.serve.demo_toolcall import build_agent\n"
        "from vllm_omni.entrypoints.omni import Omni\n"
        "from lfm2_audio.training.lora import inject_lora\n"
        "print('SETUP_VLLM_DEMO_OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", verify], cwd=args.repo, capture_output=True, text=True, check=False)
    if "SETUP_VLLM_DEMO_OK" not in out.stdout:
        raise SystemExit(f"vérification d'import échouée (rc={out.returncode})\n{out.stderr[-600:]}")
    print("SETUP_VLLM_DEMO_OK", flush=True)


if __name__ == "__main__":
    main()
