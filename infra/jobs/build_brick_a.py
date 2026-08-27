"""Build brick A: the assistant's French speech, one voice, verified clip by clip.

Every assistant turn of `C_dialogues` is spoken by the retained voice — SIWIS
cloned with Qwen3-TTS — and **each clip is transcribed back and compared to its
text before being kept**. That check is the point of the brick: the model
imitates what it is shown, so a clip whose speech drifts from its transcript
teaches drift, which is the exact defect the French workstream exists to remove.
A corpus of 95 verified clips is worth more than 100 where five lie.

Resumable clip by clip: a lost machine costs one sentence.

Output is the corpus layout (`manifest.jsonl` + `audio/`), so
`lfm2-corpus-push --brick A` can publish it unchanged.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

Synthesiser = Callable[[list[str], list[str]], tuple[list, int]]
"""Speak a batch of texts; returns waveforms and their sample rate."""

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "A_assistant_speech"

sys.path.insert(0, str(ROOT / "python"))

# Versioned under corpus/, not data/: data/ is gitignored and the pod, which
# clones the repo, found nothing there.
DIALOGUES = ROOT / "corpus/C_dialogues/dialogues.jsonl"
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SAMPLE_RATE = 24_000
LIMIT = int(os.environ.get("BRICK_A_LIMIT", "0")) or None
BATCH = int(os.environ.get("BRICK_A_BATCH", "16"))

ENGINE = os.environ.get("BRICK_A_ENGINE", "qwen")
"""``qwen`` or ``voxtral`` — both clone the same SIWIS voice.

Choosing ``voxtral`` puts the whole brick, and any model trained on it, under
the CC-BY-NC clause its weights carry. ``qwen`` (Apache 2.0, on a CC-BY-4.0
reference) leaves the corpus unencumbered.
"""

MAX_VERIFY_WER = float(os.environ.get("BRICK_A_MAX_WER", "0.15"))
"""Above this, the clip does not say its text and is dropped.

Not zero: the check transcribes with an independent ASR, whose own errors
would otherwise reject good clips. 0.15 leaves room for that while still
catching a clip that drifted.
"""


def turns_to_speak(limit: int | None) -> list[tuple[str, str, str]]:
    """``(clip_id, text, lang)`` for every assistant turn to synthesise."""
    items: list[tuple[str, str, str]] = []
    for line in DIALOGUES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        for index, turn in enumerate(case.get("turns", [])):
            if turn.get("role") != "assistant" or not turn.get("text", "").strip():
                continue
            items.append((f"{case['id']}_t{index}", turn["text"].strip(), turn.get("lang", "fr")))
    return items[:limit] if limit else items


def install_qwen_tts() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-tts"], check=False)


def qwen_synthesiser(reference) -> Synthesiser:  # noqa: ANN001 — VoiceReference
    """Qwen3-TTS cloning the reference. Apache 2.0, corpus unencumbered."""
    import torch
    from qwen_tts import Qwen3TTSModel

    install_qwen_tts()
    model = Qwen3TTSModel.from_pretrained(QWEN_BASE, device_map="cuda:0", dtype=torch.bfloat16)
    prompt = model.create_voice_clone_prompt(ref_audio=str(reference.wav_path), ref_text=reference.text)

    def speak(texts: list[str], langs: list[str]):  # noqa: ANN202
        return model.generate_voice_clone(
            text=texts,
            language=["French" if lang == "fr" else "English" for lang in langs],
            voice_clone_prompt=prompt,
        )

    return speak


def voxtral_synthesiser(reference) -> Synthesiser:  # noqa: ANN001 — VoiceReference
    """Voxtral-4B-TTS on the same reference, through vLLM-Omni.

    Reuses the bake-off job rather than re-deriving its install: that recipe
    took several attempts to get right (paired vllm/vllm-omni pins, cu13
    preloading, and resolving the reference before the install breaks
    torchaudio), and a second copy would drift from it.
    """
    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    import voxtral_tts_synth as vox

    vox.install_stack()
    vox.preload_cuda13()

    from mistral_common.protocol.speech.request import SpeechRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from vllm import SamplingParams
    from vllm_omni.entrypoints.omni import Omni

    tokenizer = MistralTokenizer.from_hf_hub(vox.MODEL).instruct_tokenizer
    engine = Omni(model=vox.MODEL)
    audio_bytes = reference.audio_bytes

    def speak(texts: list[str], langs: list[str]):  # noqa: ANN202 — langue portée par le texte
        inputs = [
            {"prompt_token_ids": tokenizer.encode_speech_request(SpeechRequest(input=t, ref_audio=audio_bytes)).tokens}
            for t in texts
        ]
        outputs = engine.generate(inputs, [SamplingParams(max_tokens=4096)] * len(inputs))
        return [o.multimodal_output["audio"].tolist() for o in outputs], vox.SAMPLE_RATE

    return speak


def main() -> None:
    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    from lfm2_audio.data_prep.corpus_layout import CorpusEntry, write_manifest
    from lfm2_audio.data_prep.siwis_reference import resolve_reference
    from lfm2_audio.ds.audio import Waveform
    from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
    from lfm2_audio.scorer.audio.wer import word_error_rate

    # Resolved before any engine install: the vLLM stack replaces torch and
    # breaks the torchaudio this needs.
    reference = resolve_reference()
    print(f"voix : SIWIS {reference.stem} [{ENGINE}] — « {reference.text[:60]} »", flush=True)

    items = turns_to_speak(LIMIT)
    todo = [(cid, text, lang) for cid, text, lang in items if not (audio_dir / f"{cid}.wav").exists()]
    print(f"{len(items)} tours assistant, {len(items) - len(todo)} déjà faits, {len(todo)} à produire", flush=True)

    transcriber = FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
    speak = voxtral_synthesiser(reference) if ENGINE == "voxtral" else qwen_synthesiser(reference)

    kept, dropped, rates = [], 0, []
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        waves, sample_rate = speak([text for _, text, _ in chunk], [lang for _, _, lang in chunk])
        for (clip_id, text, lang), wave in zip(chunk, waves, strict=False):
            path = audio_dir / f"{clip_id}.wav"
            sf.write(str(path), wave, sample_rate, subtype="PCM_16")
            heard = transcriber.transcribe(Waveform.from_file(str(path)), language=lang)
            rate = word_error_rate(text, heard)
            rates.append(rate)
            if rate > MAX_VERIFY_WER:
                path.unlink()
                dropped += 1
                continue
            kept.append(
                CorpusEntry(
                    id=clip_id,
                    audio=f"audio/{clip_id}.wav",
                    text=text,
                    lang=lang,
                    duration_s=round(len(wave) / sample_rate, 3),
                    role="assistant",
                    speaker=f"siwis_{reference.stem}",
                    source=f"{ENGINE}-tts-clone",
                    voxtral_wer=round(rate, 4),
                )
            )
        print(
            f"  {start + len(chunk)}/{len(todo)} — gardés {len(kept)}, écartés {dropped}",
            flush=True,
        )
        write_manifest(kept, OUT / "manifest.jsonl")

    summary = {
        "engine": ENGINE,
        "clips": len(kept),
        "dropped": dropped,
        "hours": round(sum(e.duration_s for e in kept) / 3600, 3),
        "median_verify_wer": round(statistics.median(rates), 4) if rates else None,
        "voice": reference.stem,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT brick_a===", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
