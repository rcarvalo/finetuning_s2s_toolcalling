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

        # SERVER mode, not in-process Omni(): the in-process path hung forever
        # after stage-0 warmup — stage-1 (the audio decoder) never started, no
        # error, no log. `vllm serve --omni` is the recipe the user PROVED on
        # Colab (single L4, same 0.26 pair); the server owns its stage
        # orchestration, we only speak HTTP to it.
        progress.step("démarrage du serveur vllm serve --omni (poids ~8 Go)")
        import base64
        import subprocess

        import httpx

        engine_t0 = time.monotonic()
        server_log = (OUT / "vllm_serve.log").open("w")
        server = subprocess.Popen(
            ["vllm", "serve", vox.MODEL, "--omni", "--port", "8001"],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
        )
        base_url = "http://127.0.0.1:8001/v1"
        ready = False
        for tick in range(90):  # 15 min max
            if server.poll() is not None:
                raise RuntimeError(f"vllm serve mort (code {server.returncode}) — voir vllm_serve.log")
            try:
                if httpx.get(f"{base_url}/models", timeout=2.0).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            if tick % 6 == 5:
                progress.note(f"serveur pas encore prêt ({(tick + 1) * 10}s)")
            time.sleep(10)
        if not ready:
            raise RuntimeError("vllm serve jamais prêt en 15 min — voir vllm_serve.log")
        engine_seconds = time.monotonic() - engine_t0
        progress.note(f"serveur prêt en {engine_seconds:.0f}s")

        progress.step("synthèse des 12 phrases (clone par ref_audio, repli preset)")
        texts = [*FR_SENTENCES, *EN_SENTENCES]
        # The server is explicit about the shape it accepts: "ref_audio must be
        # a URL (http/https), base64 data URL (data:...), or file URI" — raw
        # base64 came back 400. Hence the data URL, mime taken from the file.
        mime = "audio/wav" if reference.wav_path.suffix == ".wav" else "audio/mpeg"
        reference_b64 = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode()}"
        voice_mode: str | None = None
        waves = []
        synth_t0 = time.monotonic()

        import io

        import soundfile as sf

        with httpx.Client(timeout=180.0) as client:
            for text in texts:
                # ref_audio first — the whole point is cloning the approved
                # voice. If this server build rejects it, fall back to a preset
                # so the run still measures throughput and intelligibility, and
                # SAYS which mode produced the clips.
                payloads = [
                    (
                        {"input": text, "model": vox.MODEL, "response_format": "wav", "ref_audio": reference_b64},
                        "ref_audio",
                    ),
                    ({"input": text, "model": vox.MODEL, "response_format": "wav", "voice": "casual_female"}, "preset"),
                ]
                if voice_mode is not None:  # stick to the mode that worked
                    payloads = [p for p in payloads if p[1] == voice_mode]
                wave = None
                for payload, mode in payloads:
                    response = client.post(f"{base_url}/audio/speech", json=payload)
                    if response.status_code == 200:
                        data, rate = sf.read(io.BytesIO(response.content), dtype="float32")
                        wave = (data, rate)
                        if voice_mode is None:
                            voice_mode = mode
                            progress.note(f"mode voix retenu : {mode}")
                        break
                    progress.note(f"{mode} refusé ({response.status_code}) : {response.text[:120]}")
                if wave is None:
                    raise RuntimeError("les deux modes de voix ont été refusés — voir les notes ci-dessus")
                waves.append(wave)
        synth_seconds = time.monotonic() - synth_t0
        server.terminate()

        progress.step("écriture + contre-transcription")
        import soundfile as sf

        from lfm2_audio.ds.audio import Waveform
        from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
        from lfm2_audio.scorer.audio.wer import word_error_rate

        transcriber = FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
        clips, audio_seconds = [], 0.0
        for index, (text, (wave, rate)) in enumerate(zip(texts, waves, strict=True)):
            lang = "fr" if index < len(FR_SENTENCES) else "en"
            path = OUT / f"s{index:02d}_{lang}.wav"
            sf.write(str(path), wave, int(rate), subtype="PCM_16")
            (OUT / f"s{index:02d}_{lang}.txt").write_text(text, encoding="utf-8")
            duration = len(wave) / rate
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
            "voice_mode": voice_mode,
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
