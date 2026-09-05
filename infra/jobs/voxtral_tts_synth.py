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


# The pair Colab is shipping today, and which is therefore known to work
# together: vllm 0.26.0 with vllm-omni 0.26.0, on torch 2.11.
#
# The 0.22.1+cu129 recipe from notebooks/colab_vllm_omni_integration.ipynb was
# correct for the image it was written against, in August. Applied to a current
# image it DOWNGRADES a healthy environment: it drags in a different torch,
# which breaks torchaudio (undefined symbol: torch_library_impl), which breaks
# everything that reads a wav. Every failure of that evening came from the
# previous fix. An install recipe is dated by the image it was verified on.
VLLM_PIN = "vllm==0.26.0"
VLLM_OMNI_PIN = "vllm-omni==0.26.0"


def install_stack(progress: object | None = None) -> None:
    """Install the stack out loud.

    The quiet version cost a 45-minute run that could not be told apart from a
    hang: pip printed nothing, and RunPod's CPU/GPU metrics read near zero for a
    healthy pod as well. Multi-gigabyte steps now stream a line periodically.
    """
    from lfm2_audio.core.progress import Progress, stream_command

    reporter = progress if isinstance(progress, Progress) else Progress("voxtral-stack")
    reporter.step(f"pip install {VLLM_PIN} + {VLLM_OMNI_PIN} (plusieurs Go)")
    stream_command([sys.executable, "-m", "pip", "install", VLLM_PIN, VLLM_OMNI_PIN], reporter)
    reporter.step("pip install mistral-common")
    stream_command([sys.executable, "-m", "pip", "install", "-U", "mistral-common"], reporter)

    # Realign torchaudio ONLY if its import actually fails. The unconditional
    # version — copied from a Colab cell that was healing a broken torchaudio —
    # BROKE a healthy one: PyPI's default wheel is cu13, and on a cu128 torch
    # the reinstall traded a working pair for a CUDA mismatch that killed
    # `vllm serve` at import. A repair applied to a healthy patient is an
    # injury; probe first, and pull the wheel from the matching CUDA channel.
    reporter.step("torchaudio : contrôle d'import")
    check = subprocess.run([sys.executable, "-c", "import torchaudio"], capture_output=True, text=True, check=False)
    if check.returncode == 0:
        reporter.note("torchaudio sain — pas de réalignement")
        return

    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__.split('+')[0], torch.version.cuda)"],
        capture_output=True,
        text=True,
        check=False,
    )
    torch_version, cuda = [*probe.stdout.split(), "", ""][:2]
    channel = f"https://download.pytorch.org/whl/cu{cuda.replace('.', '')}" if cuda else "https://pypi.org/simple"
    reporter.note(f"torchaudio cassé — réalignement sur torch {torch_version} (canal {channel})")
    stream_command([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"], reporter)
    stream_command(
        [sys.executable, "-m", "pip", "install", f"torchaudio=={torch_version}", "--index-url", channel],
        reporter,
    )
    recheck = subprocess.run([sys.executable, "-c", "import torchaudio"], capture_output=True, text=True, check=False)
    if recheck.returncode != 0:
        # Last resort: PyPI default wheel (matches a cu13 torch).
        stream_command([sys.executable, "-m", "pip", "install", f"torchaudio=={torch_version}"], reporter)


def start_server(progress, *, port: int = 8001, timeout_s: int = 900):  # noqa: ANN001, ANN201 — Progress, (Popen, str)
    """``vllm serve --omni`` as a child, polled until its API answers.

    The only path that has ever produced a clip here: in-process ``Omni()``
    hung forever after stage-0 warmup on two runs, silently. The server owns
    its stage orchestration; callers only speak HTTP.
    """
    import subprocess
    import time

    import httpx

    out_dir = Path(os.environ.get("LFM2_OUT", "/workspace/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = (out_dir / "vllm_serve.log").open("w")
    # On a 24 GB card the default 0.9 leaves ~2 GB for the Whisper check that
    # runs alongside; VOXTRAL_GPU_UTIL=0.8 keeps both on the same GPU.
    util = os.environ.get("VOXTRAL_GPU_UTIL", "")
    extra = ["--gpu-memory-utilization", util] if util else []
    server = subprocess.Popen(
        ["vllm", "serve", MODEL, "--omni", "--port", str(port), *extra],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
    )
    base_url = f"http://127.0.0.1:{port}/v1"
    started = time.monotonic()
    for tick in range(timeout_s // 10):
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve mort (code {server.returncode}) — voir vllm_serve.log")
        try:
            if httpx.get(f"{base_url}/models", timeout=2.0).status_code == 200:
                progress.note(f"serveur prêt en {time.monotonic() - started:.0f}s")
                return server, base_url
        except httpx.HTTPError:
            pass
        if tick % 6 == 5:
            progress.note(f"serveur pas encore prêt ({(tick + 1) * 10}s)")
        time.sleep(10)
    raise RuntimeError(f"vllm serve jamais prêt en {timeout_s}s — voir vllm_serve.log")


def preload_cuda13() -> None:
    """Load the cu13 shared objects before importing vllm_omni.

    The wheel is built against CUDA 13 while the image ships cu12x, so
    ``import vllm_omni`` fails on a missing ``libcudart.so.13``. Preloading the
    ``nvidia/cu13`` libraries with RTLD_GLOBAL satisfies it for this process,
    and exporting LD_LIBRARY_PATH satisfies the stage subprocesses the engine
    spawns. Must run **before** any vllm_omni import — the notebook that first
    got this working states exactly that.
    """
    import contextlib
    import ctypes
    import glob

    # site-packages rather than a hardcoded path: the notebook's
    # /usr/local/lib/python*/dist-packages is Colab's layout and does not exist
    # on the pod image, where the guard then reported "not found" for a package
    # that was installed all along.
    import site

    roots = [*site.getsitepackages(), site.getusersitepackages()]
    found = [
        directory
        for root in roots
        for directory in glob.glob(f"{root}/nvidia/cu13/lib") + glob.glob(f"{root}/nvidia/*/lib")
    ]
    if not found:
        print(f"libs cu13 introuvables sous {roots} — import vllm_omni probablement voué à l'échec", flush=True)
        return
    directory = found[0]
    os.environ["LD_LIBRARY_PATH"] = directory + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    for shared_object in sorted(glob.glob(directory + "/lib*.so*")):
        with contextlib.suppress(OSError):
            ctypes.CDLL(shared_object, mode=ctypes.RTLD_GLOBAL)
    print(f"libs cu13 préchargées depuis {directory}", flush=True)


def siwis_reference() -> bytes | None:
    """The same SIWIS clip the Qwen candidate clones, as raw bytes."""
    sys.path.insert(0, str(ROOT / "python"))
    from lfm2_audio.data_prep.siwis_reference import resolve_reference

    try:
        reference = resolve_reference()
    except Exception:
        print("référence SIWIS indisponible :", traceback.format_exc(limit=3), flush=True)
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


def claim_cuda() -> None:
    """Create the CUDA context before the pip subprocesses.

    Same failure and same cure as the ASR job: both fork before touching the
    GPU and both died on "CUDA unknown error" at the first CUDA call, twice
    each. Claiming the device first fixed it there, and costs nothing here.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("aucun GPU visible")
    torch.zeros(1, device="cuda")
    print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {"candidate": "voxtral_tts", "clips": 0, "reference": "siwis"}
    try:
        claim_cuda()
        # Reference FIRST: installing the vLLM stack swaps torch for a cu129
        # build and breaks torchaudio (undefined symbol: torch_library_impl),
        # which the resolver needs to read and transcribe the clip. Resolve it
        # while the environment is still intact, then let the install do what
        # it likes.
        reference = siwis_reference()
        if reference is None:
            raise RuntimeError("pas de référence SIWIS")
        install_stack()
        preload_cuda13()

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
