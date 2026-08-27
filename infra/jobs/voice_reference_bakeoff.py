"""Which SOURCE gives a natural assistant voice? Engine fixed, reference varied.

The first bake-off compared engines on one reference. Listening to that
reference settled a different question: SIWIS is a *speech synthesis database*
— every register in it is READ speech, neutral reading included. Cloning it
faithfully reproduces someone reading a book, which is why the clone was heard
as "robotic" even after the register fix. The problem was never only the
engine; it is the source.

So: same engine (Qwen3-TTS, the measured winner), same ten sentences, three
different ways of sourcing the voice.

  siwis_neut     real human, studio, but READ register — today's baseline
  dialogue_conv  a clip from french-dialogue-tts-1000h: synthetic origin, but
                 genuinely conversational register ("Ah tu sais, j'ai rellu
                 Proust l'autre jour et franchement…"). Tests whether register
                 matters more than authenticity.
  voice_design   no reference at all — Qwen3-TTS VoiceDesign builds a voice from
                 a French description. Neither cloning nor a preset, and it
                 cannot inherit a reading register because there is nothing to
                 inherit from.

The comparison is decided by ear. UTMOS scored the emphatic clone 4.02 and the
neutral one 3.99 while a listener separated them instantly — naturalness metrics
do not know what a receptionist should sound like.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "ref_bakeoff"

sys.path.insert(0, str(ROOT / "python"))

QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DIALOGUE_REPO = "Rcarvalo/french-dialogue-tts-1000h"

VOICE_DESCRIPTION = (
    "Voix féminine française, chaleureuse et posée, timbre naturel de conversation. "
    "Débit normal, articulation claire sans emphase, ton d'accueil professionnel et amical. "
    "Elle parle, elle ne lit pas."
)
"""The last sentence is the whole point: everything sourced from SIWIS reads."""

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-tts"], check=False)


def dialogue_reference() -> tuple[str, str] | None:
    """A conversational clip and its text, from the dialogue corpus.

    Picked for length rather than position: cloning wants a few seconds, and
    the corpus mixes one-liners with long monologues.
    """
    from huggingface_hub import hf_hub_download

    manifest = Path(hf_hub_download(DIALOGUE_REPO, "metadata.jsonl", repo_type="dataset"))
    for line in manifest.read_text(encoding="utf-8").splitlines()[:200]:
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        if not 60 <= len(text) <= 220:
            continue
        wav = hf_hub_download(DIALOGUE_REPO, row["file_name"], repo_type="dataset")
        import soundfile as sf

        duration = sf.info(wav).duration
        if 3.0 <= duration <= 15.0:
            print(f"référence dialogue : {row['file_name']} ({duration:.1f}s) — « {text[:60]} »", flush=True)
            return wav, text
    return None


def save(name: str, index: int, wave, sample_rate: int) -> None:  # noqa: ANN001 — tableau
    import soundfile as sf

    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    sf.write(str(target / f"s{index:02d}.wav"), wave, sample_rate, subtype="PCM_16")
    (target / f"s{index:02d}.txt").write_text(SENTENCES[index], encoding="utf-8")


def clone_arm(name: str, wav: str, text: str) -> int:
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(QWEN_BASE, device_map="cuda:0", dtype=torch.bfloat16)
    prompt = model.create_voice_clone_prompt(ref_audio=wav, ref_text=text)
    waves, rate = model.generate_voice_clone(
        text=list(SENTENCES), language=["French"] * len(SENTENCES), voice_clone_prompt=prompt
    )
    for index, wave in enumerate(waves):
        save(name, index, wave, rate)
    del model
    torch.cuda.empty_cache()
    return len(waves)


def design_arm(name: str) -> int:
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(QWEN_DESIGN, device_map="cuda:0", dtype=torch.bfloat16)
    waves, rate = model.generate_voice_design(
        text=list(SENTENCES),
        language=["French"] * len(SENTENCES),
        instruct=[VOICE_DESCRIPTION] * len(SENTENCES),
    )
    for index, wave in enumerate(waves):
        save(name, index, wave, rate)
    del model
    torch.cuda.empty_cache()
    return len(waves)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    install_qwen_tts()

    from lfm2_audio.data_prep.siwis_reference import resolve_reference

    results = []
    arms: list[tuple[str, object]] = []

    siwis = resolve_reference()
    arms.append(("siwis_neut", lambda: clone_arm("siwis_neut", str(siwis.wav_path), siwis.text)))

    dialogue = dialogue_reference()
    if dialogue is not None:
        arms.append(("dialogue_conv", lambda: clone_arm("dialogue_conv", dialogue[0], dialogue[1])))

    arms.append(("voice_design", lambda: design_arm("voice_design")))

    for name, run in arms:
        done = len(list((OUT / name).glob("*.wav"))) if (OUT / name).exists() else 0
        if done >= len(SENTENCES):
            print(f"[{name}] déjà complet", flush=True)
            results.append({"arm": name, "clips": done})
            continue
        try:
            clips = run()  # type: ignore[operator]
            results.append({"arm": name, "clips": clips})
        except Exception as error:
            print(f"[{name}] ÉCHEC : {error}", flush=True)
            results.append({"arm": name, "clips": 0, "error": str(error)[:200]})
        print(f"[{name}] terminé", flush=True)
        (OUT / "ref_bakeoff.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    print("===RESULT ref_bakeoff.json===", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
