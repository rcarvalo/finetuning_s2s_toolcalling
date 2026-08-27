"""Voice bake-off for brick A: the same French sentences, every candidate voice.

"Quality only" is a decision, so it is measured. The assistant voice is the most
consequential data choice of the corpus: brick A teaches the model what to SAY,
and every clip in it is imitated. A voice that drifts from its text teaches
drift — the very failure the French workstream exists to remove.

Three candidates, deliberately chosen to separate two different questions.

  qwen_siwis   Qwen3-TTS **Base** cloning a SIWIS clip. SIWIS is studio French
               from a professional speaker (CC-BY-4.0) and the model is
               Apache 2.0, so this path is both native French and free of any
               non-commercial clause.
  qwen_preset  Qwen3-TTS **CustomVoice** with an English-native preset speaking
               French. The CustomVoice catalogue has no French timbre, so this
               is the control for "a multilingual voice doing French" — exactly
               what cloning is supposed to beat.
Voxtral-4B-TTS is the third candidate but lives in its own job
(`voxtral_tts_synth`): it runs on vLLM-Omni, whose dependency set replaces
transformers wholesale and would break qwen-tts if they shared an environment.

The incumbent — the voice already in french-dialogue-tts-1000h, UTMOS 3.73 —
is not synthesised here: its clips exist. It enters the comparison at scoring
time, and without it a bake-off can only say which newcomer wins, not whether
changing anything is worth it.

This job SYNTHESISES only. Scoring runs off-GPU on the gate stack (VERSA for
naturalness, an independent ASR for text fidelity), the split that has kept
every measurement of this project honest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "bakeoff"

sys.path.insert(0, str(ROOT / "python"))

SIWIS_REPO = "Aviv-anthonnyolime/SIWIS_French_Speech_Synthesis_Database"
SIWIS_CLIP = ""  # apparié dynamiquement : tous les wav n'ont pas de .lab
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PRESET_SPEAKER = "Aiden"

# Conversational register, not read prose: brick A feeds a voice assistant, and
# a voice that shines on literary sentences may not on "Il est quinze heures".
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


def install_qwen_tts() -> None:
    """Qwen3-TTS ships its own runtime; transformers alone does not know it.

    Installed after the entrypoint's regression assertion has run, so a
    dependency bump here cannot invalidate the check that guards the campaign.
    """
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-tts"], check=False)


def siwis_reference() -> tuple[str, str] | None:
    """A SIWIS clip and its transcript — cloning needs both.

    The pair is found by intersecting the wav and lab file lists rather than
    hardcoded: the repository ships 314 wavs and 400 labs and they do not cover
    each other, so a name picked by eye can point at a clip whose transcript is
    missing. That is how the first run lost its main candidate.
    """
    from huggingface_hub import HfApi, hf_hub_download

    try:
        files = HfApi().list_repo_files(SIWIS_REPO, repo_type="dataset")
        wavs = {Path(f).stem: f for f in files if f.endswith(".wav")}
        labs = {Path(f).stem: f for f in files if f.endswith(".lab")}
        paired = sorted(set(wavs) & set(labs))
        if not paired:
            print("aucune paire wav/lab dans SIWIS", flush=True)
            return None
        stem = paired[0]
        wav = hf_hub_download(SIWIS_REPO, wavs[stem], repo_type="dataset")
        lab = hf_hub_download(SIWIS_REPO, labs[stem], repo_type="dataset")
    except Exception:
        print("référence SIWIS indisponible :", traceback.format_exc(limit=1), flush=True)
        return None
    text = Path(lab).read_text(encoding="utf-8", errors="replace").strip()
    print(f"référence SIWIS : {Path(wav).name} ({len(paired)} paires) — « {text[:70]} »", flush=True)
    return wav, text


def save(name: str, index: int, wave, sample_rate: int) -> None:  # noqa: ANN001 — tableau numpy
    import soundfile as sf

    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    sf.write(str(target / f"s{index:02d}.wav"), wave, sample_rate, subtype="PCM_16")
    (target / f"s{index:02d}.txt").write_text(SENTENCES[index], encoding="utf-8")


def run_candidate(name: str, synthesise) -> dict[str, object]:  # noqa: ANN001 — callable
    """Run one candidate; a failure is recorded, never fatal to the others."""
    try:
        synthesise()
        clips = len(list((OUT / name).glob("*.wav")))
        status = "ok"
    except Exception:
        clips = len(list((OUT / name).glob("*.wav"))) if (OUT / name).exists() else 0
        status = traceback.format_exc(limit=2)
        print(f"[{name}] ÉCHEC : {status}", flush=True)
    print(f"[{name}] {clips}/{len(SENTENCES)} clips", flush=True)
    return {"candidate": name, "clips": clips, "status": status}


def qwen_clone(reference: tuple[str, str]) -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    wav, text = reference
    model = Qwen3TTSModel.from_pretrained(QWEN_BASE, device_map="cuda:0", dtype=torch.bfloat16)
    # Built once and reused: the reference features are identical for every
    # sentence, and recomputing them per clip is pure waste.
    prompt = model.create_voice_clone_prompt(ref_audio=wav, ref_text=text, x_vector_only_mode=False)
    waves, sample_rate = model.generate_voice_clone(
        text=list(SENTENCES),
        language=["French"] * len(SENTENCES),
        voice_clone_prompt=prompt,
    )
    for index, wave in enumerate(waves):
        save("qwen_siwis", index, wave, sample_rate)


def qwen_preset() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(QWEN_CUSTOM, device_map="cuda:0", dtype=torch.bfloat16)
    print("timbres disponibles :", model.get_supported_speakers(), flush=True)
    waves, sample_rate = model.generate_custom_voice(
        text=list(SENTENCES),
        language=["French"] * len(SENTENCES),
        speaker=[PRESET_SPEAKER] * len(SENTENCES),
    )
    for index, wave in enumerate(waves):
        save("qwen_preset", index, wave, sample_rate)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    install_qwen_tts()

    results = []
    reference = siwis_reference()
    if reference is not None:
        results.append(run_candidate("qwen_siwis", lambda: qwen_clone(reference)))
    results.append(run_candidate("qwen_preset", qwen_preset))

    (OUT / "bakeoff.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT bakeoff.json===", flush=True)
    print(json.dumps([{k: v for k, v in r.items() if k != "status"} for r in results], ensure_ascii=False), flush=True)
    print("===END===", flush=True)
    print(f"clips sous {OUT} — notation hors GPU", flush=True)


if __name__ == "__main__":
    main()
