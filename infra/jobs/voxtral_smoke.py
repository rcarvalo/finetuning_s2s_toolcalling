"""Voxtral smoke test: does it clone the approved voice, and at what price?

Twelve sentences, not a corpus. This run exists to answer three questions
before any 100 h synthesis is paid for:

1. **Does the stack come up at all?** Six previous runs died inside vLLM-Omni
   init; the plugin fix (c6db14b) and the phase logging make this run either
   work or say precisely which stage stalled.
2. **Does the voice identity survive the engine change?** The user approved a
   *voice* — the ``ref_dialogue_conv`` clip — heard through a Qwen clone.
   Voxtral clones the SAME reference here; the clips go back for the ear and
   for a VERSA ``spk_similarity`` check against the stock corpus (locally: the
   VERSA venv lives on the Mac, not on pods).
3. **What does a clip cost?** Measured RTF → projected GPU-hours and dollars
   for the 103.6 h the corpus plan still needs. Estimated so far; decided here.

Two English sentences ride along because the plan's EN slice assumes Voxtral
holds English in the same timbre — an assumption nobody has tested.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "voxtral_smoke"

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "infra" / "jobs"))

REMAINING_CORPUS_H = 103.6
"""What configs/corpus/fr_150h.yaml still needs synthesised — the projection target."""

GPU_DOLLARS_PER_H = float(os.environ.get("SMOKE_GPU_RATE", "0.35"))

FR_SENTENCES = [
    # The ten receptionist sentences of every bake-off, so the ear compares
    # this run with the Qwen arms it has already heard.
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
EN_SENTENCES = [
    "Sure, I can check that for you right away.",
    "Your meeting room is on the second floor, just past the elevators.",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    from lfm2_audio.core.progress import Progress

    with Progress("voxtral-smoke") as progress:
        # Reference FIRST: the vLLM install replaces torch and breaks the
        # torchaudio this resolution path needs.
        progress.step("résolution de la référence vocale (avant toute install)")
        from lfm2_audio.data_prep.voice_reference import resolve_voice_reference

        reference = resolve_voice_reference("dialogue")
        audio_bytes = reference.audio_bytes
        progress.note(f"{reference.stem} — « {reference.text[:60]} »")

        import voxtral_tts_synth as vox

        vox.install_stack(progress)
        progress.step("préchargement des bibliothèques CUDA 13")
        vox.preload_cuda13()

        progress.step("import de vllm_omni")
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from vllm import SamplingParams
        from vllm_omni.entrypoints.omni import Omni

        progress.step(f"tokenizer {vox.MODEL}")
        tokenizer = MistralTokenizer.from_hf_hub(vox.MODEL).instruct_tokenizer

        progress.step("démarrage du moteur Omni (poids ~8 Go)")
        engine_t0 = time.monotonic()
        engine = Omni(model=vox.MODEL)
        engine_seconds = time.monotonic() - engine_t0
        progress.note(f"moteur prêt en {engine_seconds:.0f}s")

        progress.step("synthèse des 12 phrases")
        texts = [*FR_SENTENCES, *EN_SENTENCES]
        synth_t0 = time.monotonic()
        inputs = [
            {"prompt_token_ids": tokenizer.encode_speech_request(SpeechRequest(input=t, ref_audio=audio_bytes)).tokens}
            for t in texts
        ]
        outputs = engine.generate(inputs, [SamplingParams(max_tokens=4096)] * len(inputs))
        waves = [o.multimodal_output["audio"].tolist() for o in outputs]
        synth_seconds = time.monotonic() - synth_t0

        progress.step("écriture + contre-transcription")
        import soundfile as sf

        from lfm2_audio.ds.audio import Waveform
        from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
        from lfm2_audio.scorer.audio.wer import word_error_rate

        transcriber = FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
        clips, audio_seconds = [], 0.0
        for index, (text, wave) in enumerate(zip(texts, waves, strict=True)):
            lang = "fr" if index < len(FR_SENTENCES) else "en"
            path = OUT / f"s{index:02d}_{lang}.wav"
            sf.write(str(path), wave, vox.SAMPLE_RATE, subtype="PCM_16")
            (OUT / f"s{index:02d}_{lang}.txt").write_text(text, encoding="utf-8")
            duration = len(wave) / vox.SAMPLE_RATE
            audio_seconds += duration
            heard = transcriber.transcribe(Waveform.from_file(str(path)), language=lang)
            wer = word_error_rate(text, heard)
            clips.append({"clip": path.name, "lang": lang, "duration_s": round(duration, 2), "wer": round(wer, 3)})
            progress.note(f"{path.name} {duration:4.1f}s WER {wer:.2f} — « {heard[:50]} »")

        # RTF = audio seconds produced per wall second; the projection prices
        # the remaining corpus at this pod's hourly rate.
        rtf = audio_seconds / synth_seconds if synth_seconds else 0.0
        projected_gpu_h = (REMAINING_CORPUS_H * 3600 / rtf) / 3600 if rtf else None
        fr_wers = [c["wer"] for c in clips if c["lang"] == "fr"]
        summary = {
            "engine_start_s": round(engine_seconds, 1),
            "synthesis_s": round(synth_seconds, 1),
            "audio_s": round(audio_seconds, 1),
            "rtf": round(rtf, 2),
            "median_wer_fr": round(statistics.median(fr_wers), 3) if fr_wers else None,
            "wer_en": [c["wer"] for c in clips if c["lang"] == "en"],
            "projected_gpu_h_for_corpus": round(projected_gpu_h, 1) if projected_gpu_h else None,
            "projected_dollars": round(projected_gpu_h * GPU_DOLLARS_PER_H, 2) if projected_gpu_h else None,
            "clips": clips,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    print("===RESULT voxtral_smoke===", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
