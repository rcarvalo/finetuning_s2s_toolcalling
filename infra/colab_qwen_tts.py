"""Voice the Phase B ASSISTANT answer turns with Qwen3-TTS (Colab L4).

Why this model for these turns: in interleaved Phase B training the model
learns to PREDICT the audio codes of the assistant answers, so their voice
becomes v3's voice. Qwen3-TTS is the strongest permissively-licensed TTS of
2026 (Apache 2.0, lowest open-model WER) and needs none of the vLLM stack.

User turns are voiced separately (Kokoro, 8 voices, locally): input-side audio
rewards diversity, and engine transfer is proven (v2 = 0.833 on unseen-engine
voices).

CustomVoice English presets are male: Ryan (rhythmic) and Aiden (sunny
American). Default Aiden; override with ASSISTANT_VOICE. Resumable: existing
WAVs are kept.

Outputs ``data/audio_phase_b_assistant/<dialogue_id>_a<turn_index>.wav`` and
pushes a tarball to the Hub.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import soundfile as sf
import torch

SRC = Path("data/phase_b_train_src.jsonl")
AUDIO = Path("data/audio_phase_b_assistant")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
HUB_REPO = "Rcarvalo/tc-en-voice-agent-v1"
INSTRUCT = "Speak as a friendly, clear voice assistant: warm, natural pace, no whispering."


def spoken_assistant_turns() -> list[tuple[str, int, str]]:
    """(dialogue_id, turn_index, text) for every assistant turn that speaks.

    Tool-call turns carry no text to speak; tool turns are backend payloads.
    """
    jobs = []
    with SRC.open(encoding="utf-8") as handle:
        for line in handle:
            dialogue = json.loads(line)
            for index, turn in enumerate(dialogue["turns"]):
                if turn.get("role") == "assistant" and turn.get("text") and not turn.get("tool_calls"):
                    jobs.append((dialogue["id"], index, turn["text"]))
    return jobs


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    jobs = spoken_assistant_turns()
    pending = [(d, i, t) for d, i, t in jobs if not (AUDIO / f"{d}_a{i}.wav").exists()]
    print(f"{len(jobs)} spoken assistant turns, {len(pending)} to synthesize", flush=True)

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16)
    speaker = os.environ.get("ASSISTANT_VOICE", "Aiden")

    batch_size = int(os.environ.get("TTS_BATCH", "8"))
    done = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [text for _, _, text in batch]
        wavs, rate = model.generate_custom_voice(
            text=texts,
            language=["English"] * len(texts),
            speaker=[speaker] * len(texts),
            instruct=[INSTRUCT] * len(texts),
        )
        for (dialogue_id, index, _), wav in zip(batch, wavs, strict=True):
            sf.write(str(AUDIO / f"{dialogue_id}_a{index}.wav"), wav, rate, subtype="PCM_16")
        done += len(batch)
        if done % 96 < batch_size or done == len(pending):
            print(f"  {done}/{len(pending)}", flush=True)

    tarball = "/tmp/phase_b_assistant_audio.tar.gz"
    with tarfile.open(tarball, "w:gz") as archive:
        archive.add(str(AUDIO), arcname=AUDIO.name)
    from huggingface_hub import HfApi

    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=tarball, path_in_repo="phase_b/assistant_audio.tar.gz", repo_id=HUB_REPO, repo_type="dataset"
    )
    print("QWEN_TTS_DONE", flush=True)


if __name__ == "__main__":
    main()
