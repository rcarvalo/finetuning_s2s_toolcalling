"""Voxtral-4B-TTS through vLLM-Omni — the third candidate of the brick A bake-off.

Its own job, not a branch of ``voice_bakeoff``: Voxtral-TTS ships no
transformers-format weights (a first attempt died on a missing
``model.safetensors``) and runs on vLLM-Omni, whose dependency set replaces
transformers wholesale. Sharing an environment with ``qwen-tts`` would break one
of the two, and a bake-off whose candidates sabotage each other measures the
installer, not the voices.

**In-process, never a served child.** The obvious route is
``vllm serve --omni`` plus HTTP calls, and it is what the model card shows. This
project already has an open wound there: on the v4 demo the vLLM engine started
as a child process and *died without leaving a trace*. ``Omni`` runs the engine
inside this process instead, so a failure produces a traceback in the job log
rather than silence.

Same SIWIS reference as the Qwen candidate, through ``ref_audio``: comparing two
engines on one voice isolates the engine, where comparing two different voices
would confound engine and timbre.

Resumable clip by clip — Colab pruned two sessions in an hour on 27/08, so a
lost VM must cost one sentence, not ten.

Licence, read from the model card: the non-commercial clause comes from the
voice references shipped with the model, but the weights themselves are
CC-BY-NC, so output stays encumbered whatever reference is supplied. That is
why this candidate is a yardstick rather than a default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "bakeoff" / "voxtral_tts"

MODEL = "mistralai/Voxtral-4B-TTS-2603"
SIWIS_REPO = "Aviv-anthonnyolime/SIWIS_French_Speech_Synthesis_Database"
SAMPLE_RATE = 24_000

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
    """Install without a shell — a ``<`` in a specifier becomes a redirection."""
    subprocess.run([sys.executable, "-m", "pip", *args], check=False)


# The exact pair verified in infra/setup_vllm_demo.py. Taking the latest of
# each instead broke on `No module named vllm.entrypoints.serve.utils
# .error_response`: vllm-omni tracks a specific vLLM API and the two release
# on their own schedules, so "newest of both" is not a combination anyone has
# tested.
VLLM_WHEEL = (
    "vllm @ https://github.com/vllm-project/vllm/releases/download/"
    "v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
CU129_INDEX = "https://download.pytorch.org/whl/cu129"


def install_stack() -> None:
    pip("install", "-q", VLLM_WHEEL, "--extra-index-url", CU129_INDEX)
    pip("install", "-q", "vllm-omni>=0.22,<0.23")
    pip("install", "-q", "-U", "mistral-common")
    # torchao ships extensions built for CPython 3.10 and segfaults the 3.13
    # images; nothing on this path uses it.
    pip("uninstall", "-y", "-q", "torchao")


def siwis_reference() -> bytes | None:
    """The same SIWIS clip the Qwen candidate clones, as raw bytes."""
    sys.path.insert(0, str(ROOT / "python"))
    from lfm2_audio.data_prep.siwis_reference import SiwisError, resolve_reference

    try:
        reference = resolve_reference()
    except (SiwisError, OSError, ValueError):
        print("référence SIWIS indisponible :", traceback.format_exc(limit=1), flush=True)
        return None
    print(f"référence SIWIS : {reference.stem} — « {reference.text[:70]} »", flush=True)
    return reference.audio_bytes


def build_inputs(reference: bytes) -> list[dict]:
    """One speech request per sentence, cloning the reference voice."""
    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_hf_hub(MODEL).instruct_tokenizer
    inputs = []
    for sentence in SENTENCES:
        tokenized = tokenizer.encode_speech_request(SpeechRequest(input=sentence, ref_audio=reference))
        inputs.append({"prompt_token_ids": tokenized.tokens})
    return inputs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"candidate": "voxtral_tts", "clips": 0, "reference": "siwis"}
    try:
        install_stack()
        reference = siwis_reference()
        if reference is None:
            raise RuntimeError("pas de référence SIWIS")

        import soundfile as sf
        from vllm import SamplingParams
        from vllm_omni.entrypoints.omni import Omni

        inputs = build_inputs(reference)
        engine = Omni(model=MODEL)
        outputs = engine.generate(inputs, [SamplingParams(max_tokens=4096)] * len(inputs))

        for index, output in enumerate(outputs):
            audio = output.multimodal_output["audio"].tolist()
            path = OUT / f"s{index:02d}.wav"
            sf.write(str(path), audio, SAMPLE_RATE)
            (OUT / f"s{index:02d}.txt").write_text(SENTENCES[index], encoding="utf-8")
            print(f"  s{index:02d} {len(audio) / SAMPLE_RATE:.1f}s", flush=True)
        result["clips"] = len(list(OUT.glob("*.wav")))
        result["status"] = "ok"
    except Exception:
        result["clips"] = len(list(OUT.glob("*.wav")))
        result["status"] = traceback.format_exc(limit=3)
        print("ÉCHEC :", result["status"], flush=True)

    (OUT / "voxtral.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT voxtral.json===", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "status"}, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
