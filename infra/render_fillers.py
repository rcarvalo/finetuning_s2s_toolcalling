"""Pré-rend les phrases d'attente en WAV avec la voix Aiden (Qwen3-TTS).

La voix des fillers DOIT être celle que le modèle a apprise en Phase B :
c'est ce qui rend la transition filler → réponse invisible à l'oreille. Une
autre voix ferait entendre deux locuteurs dans le même tour.

Sortie : ``<tool>_<i>.wav`` à 24 kHz dans --out, le nommage exact que
``FillerBank`` résout (l'index suit l'ordre des phrases).

    python infra/render_fillers.py --out data/fillers_en
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lfm2_audio.orchestrator.fillers import EN_FILLER_PHRASES

TARGET_RATE = 24_000  # cadence de sortie de la démo (SR_OUT)
INSTRUCT = "Speak as a friendly, clear voice assistant: warm, natural pace, no whispering."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/fillers_en", type=Path)
    parser.add_argument("--voice", default="Aiden")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    args.out.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0", dtype=torch.bfloat16
    )

    for tool, phrases in EN_FILLER_PHRASES.items():
        stem = "default" if tool == "_default" else tool
        for index, phrase in enumerate(phrases):
            wavs, rate = model.generate_custom_voice(
                text=phrase, language="English", speaker=args.voice, instruct=INSTRUCT
            )
            wave = wavs[0]
            if rate != TARGET_RATE:
                import torchaudio

                wave = torchaudio.functional.resample(torch.as_tensor(wave).reshape(1, -1), rate, TARGET_RATE)[
                    0
                ].numpy()
            target = args.out / f"{stem}_{index}.wav"
            sf.write(str(target), wave, TARGET_RATE, subtype="PCM_16")
            print(f"  {target.name}: {phrase!r}", flush=True)
    print("FILLERS_DONE", flush=True)


if __name__ == "__main__":
    main()
