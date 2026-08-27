"""Voxtral-4B-TTS through vLLM-Omni — the third candidate of the brick A bake-off.

Its own job, not a branch of ``voice_bakeoff``: Voxtral-TTS has no
transformers-format weights (the first attempt died on a missing
``model.safetensors``) and runs on vLLM-Omni, whose dependency set replaces
transformers wholesale. Sharing an environment with ``qwen-tts`` would break
one of the two, and a bake-off where candidates sabotage each other measures
the installer, not the voices.

The install follows ``infra/setup_vllm_demo.py``, verified layer by layer on
26/08 — in particular **never through a shell**: a pip specifier contains ``<``,
which bash turns into a redirection and the install then fails silently.

Synthesises the same ten sentences as the Qwen candidates. Scoring is off-GPU,
against the same yardsticks.

Licence note recorded while reading the model card: the non-commercial clause
comes from the voice references shipped with the model (EARS, CML-TTS and
others under CC-BY-NC). The weights themselves are published CC-BY-NC, so
output stays encumbered whatever reference is used — which is why this
candidate is a yardstick rather than a default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "bakeoff" / "voxtral_tts"

MODEL = "mistralai/Voxtral-4B-TTS-2603"
VOICE = os.environ.get("VOXTRAL_VOICE", "casual_male")
BASE_URL = "http://127.0.0.1:8000/v1"
READY_TIMEOUT_S = 900

SENTENCES = [
    "Bonjour, comment puis-je vous aider aujourd'hui ?",
    "Il est quinze heures trente, votre rendez-vous est dans une demi-heure.",
    "Je n'ai pas trouvé ce nom dans l'annuaire. Pouvez-vous me l'épeler ?",
    "D'accord, je préviens votre interlocutrice tout de suite.",
    "Le code du wifi invité est affiché sur le panneau derrière vous.",
    "Attendez, je vérifie... Oui, c'est bien confirmé pour jeudi.",
    "Désolé, je n'ai pas bien entendu. Vous pouvez répéter ?",
    "Avec plaisir ! Bonne journée et à bientôt.",
    "Alors, il y a deux possibilités : soit vous patientez, soit je vous rappelle.",
    "Je vous mets en relation, ne quittez pas.",
]


def pip(*args: str) -> None:
    """Install without a shell — a ``<`` in a specifier would become a redirect."""
    subprocess.run([sys.executable, "-m", "pip", *args], check=False)


def install_stack() -> None:
    pip("install", "-q", "-U", "vllm")
    pip("install", "-q", "vllm-omni>=0.18")
    # torchao ships extensions built for CPython 3.10 and segfaults the
    # interpreter on the 3.13 images; nothing here uses it.
    pip("uninstall", "-y", "-q", "torchao")


def serve() -> subprocess.Popen[bytes]:
    log = OUT.parent / "vllm_serve.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        ["vllm", "serve", MODEL, "--omni"],
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_ready() -> bool:
    """Poll until the server answers, or give up with the reason visible."""
    import httpx

    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/models", timeout=5.0).status_code == 200:
                print("serveur vLLM prêt", flush=True)
                return True
        except Exception:
            pass
        time.sleep(10)
    print(f"serveur vLLM non prêt après {READY_TIMEOUT_S}s", flush=True)
    return False


def synthesise() -> int:
    import httpx
    import soundfile as sf

    OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, sentence in enumerate(SENTENCES):
        target = OUT / f"s{index:02d}.wav"
        if target.exists():
            written += 1
            continue
        response = httpx.post(
            f"{BASE_URL}/audio/speech",
            json={"input": sentence, "model": MODEL, "response_format": "wav", "voice": VOICE},
            timeout=180.0,
        )
        response.raise_for_status()
        target.write_bytes(response.content)
        (OUT / f"s{index:02d}.txt").write_text(sentence, encoding="utf-8")
        info = sf.info(str(target))
        print(f"  s{index:02d} {info.duration:.1f}s @ {info.samplerate} Hz", flush=True)
        written += 1
    return written


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    install_stack()
    server = serve()
    result: dict[str, object] = {"candidate": "voxtral_tts", "voice": VOICE, "clips": 0}
    try:
        if wait_ready():
            result["clips"] = synthesise()
            result["status"] = "ok"
        else:
            result["status"] = "serveur non prêt"
    except Exception:
        result["status"] = traceback.format_exc(limit=2)
        print("ÉCHEC :", result["status"], flush=True)
    finally:
        server.terminate()

    (OUT / "voxtral.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT voxtral.json===", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
