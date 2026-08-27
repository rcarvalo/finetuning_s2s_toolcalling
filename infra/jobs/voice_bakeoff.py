"""Voice bake-off for brick A: same French sentences, every candidate voice.

"Quality only" is a decision, not a preference, so it is measured. The assistant
voice is the single most consequential data choice of the corpus: brick A
teaches the model what to SAY, and every clip it contains is imitated. A voice
that drifts from its text teaches drift — which is precisely the failure the
whole French workstream is trying to remove.

Candidates, all synthesising the SAME sentences:

  qwen_siwis    Qwen3-TTS CustomVoice cloning SIWIS (Apache 2.0 + CC-BY-4.0)
  qwen_base     Qwen3-TTS default French voice
  voxtral_tts   Voxtral-4B-TTS (CC-BY-NC — kept as a yardstick, not a default)
  incumbent     the existing french-dialogue-tts-1000h voice, UTMOS 3.73

The incumbent matters: a bake-off without the current best can only tell you
which newcomer wins, not whether changing anything is worth it.

This job SYNTHESISES only. Scoring happens off-GPU with the gate stack (VERSA
for naturalness, an independent ASR for text fidelity), the same split that has
kept every measurement honest so far.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "bakeoff"

sys.path.insert(0, str(ROOT / "python"))

SIWIS_DATASET = "Aviv-anthonnyolime/SIWIS_French_Speech_Synthesis_Database"
QWEN_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOXTRAL_TTS = "mistralai/Voxtral-4B-TTS-2603"
INCUMBENT = "Rcarvalo/french-dialogue-tts-1000h"

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


def reference_clip() -> Path | None:
    """One SIWIS clip to clone from — the voice identity of the assistant."""
    import soundfile as sf
    from datasets import load_dataset

    try:
        rows = load_dataset(SIWIS_DATASET, split="train", streaming=True)
        row = next(iter(rows))
    except Exception:
        print("SIWIS indisponible :", traceback.format_exc(limit=1), flush=True)
        return None
    audio = row.get("audio")
    if not isinstance(audio, dict) or "array" not in audio:
        print("schéma SIWIS inattendu :", list(row)[:8], flush=True)
        return None
    path = OUT / "siwis_reference.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio["array"], audio["sampling_rate"], subtype="PCM_16")
    print(f"référence SIWIS : {path} ({len(audio['array']) / audio['sampling_rate']:.1f}s)", flush=True)
    return path


def synthesise(name: str, fn) -> dict[str, object]:  # noqa: ANN001 — callable de synthèse
    """Run one candidate, recording why it failed rather than aborting."""
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for index, sentence in enumerate(SENTENCES):
            path = target / f"s{index:02d}.wav"
            fn(sentence, path)
            (target / f"s{index:02d}.txt").write_text(sentence, encoding="utf-8")
            written.append(path.name)
        status = "ok"
    except Exception:
        status = traceback.format_exc(limit=2)
        print(f"[{name}] ÉCHEC : {status}", flush=True)
    result = {"candidate": name, "clips": len(written), "status": status}
    print(f"[{name}] {len(written)}/{len(SENTENCES)} clips", flush=True)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    reference = reference_clip()

    if reference is not None:
        results.append(synthesise("qwen_siwis", _qwen_cloner(reference)))
    results.append(synthesise("qwen_base", _qwen_cloner(None)))
    results.append(synthesise("voxtral_tts", _voxtral_speaker()))

    (OUT / "bakeoff.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT bakeoff.json===", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("===END===", flush=True)
    print(f"clips à récupérer sous {OUT} — notation hors GPU", flush=True)


def _qwen_cloner(reference: Path | None):  # noqa: ANN202 — fabrique de closure
    """Qwen3-TTS, cloning ``reference`` when given one."""
    from transformers import AutoModel, AutoProcessor

    model_id = QWEN_CUSTOM if reference else QWEN_BASE
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

    def speak(text: str, path: Path) -> None:
        import soundfile as sf

        kwargs = {"text": text}
        if reference is not None:
            kwargs["reference_audio"] = str(reference)
        inputs = processor(**kwargs, return_tensors="pt")
        with_audio = model.generate(**inputs)
        waveform = with_audio["waveform"] if isinstance(with_audio, dict) else with_audio
        sf.write(str(path), waveform.squeeze().float().cpu().numpy(), 24000, subtype="PCM_16")

    return speak


def _voxtral_speaker():  # noqa: ANN202 — fabrique de closure
    """Voxtral-4B-TTS, kept as a yardstick despite its non-commercial licence."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(VOXTRAL_TTS, trust_remote_code=True)
    model = AutoModel.from_pretrained(VOXTRAL_TTS, trust_remote_code=True).eval()

    def speak(text: str, path: Path) -> None:
        import soundfile as sf

        inputs = processor(text=text, return_tensors="pt")
        output = model.generate(**inputs)
        waveform = output["waveform"] if isinstance(output, dict) else output
        sf.write(str(path), waveform.squeeze().float().cpu().numpy(), 24000, subtype="PCM_16")

    return speak


if __name__ == "__main__":
    main()
