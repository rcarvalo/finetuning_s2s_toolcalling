"""Voxtral voice run: the truly French voices, cloning, and batched throughput.

Run 3 proved the engine (server up in 261 s, EN WER 0.00/0.08) but its clips
used ``casual_female`` — an English-register preset whose French carries the
accent the listener immediately flagged. The model repo ships twenty
``voice_embedding/*.pt`` and among them **``fr_female`` and ``fr_male``** —
native French voices from CML-TTS. This run produces all the candidates the
ear needs to pick one:

  fr_female / fr_male   the native presets
  clone                 ref_audio on the approved ``ref_dialogue_conv`` clip,
                        as a data URL — run 3's 400 stated that exact contract

And it measures what run 3 could not: **batched RTF**. The model card claims
RTF 0.302 at concurrency 32 on H200; run 3 posted sequentially and got 4.26.
Sentences here go out concurrently per arm, so the corpus projection uses a
throughput someone actually measured on our stack.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "voxtral_voices"

sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "infra" / "jobs"))

REMAINING_CORPUS_H = 103.6
GPU_DOLLARS_PER_H = float(os.environ.get("SMOKE_GPU_RATE", "1.19"))
CONCURRENCY = int(os.environ.get("SMOKE_CONCURRENCY", "12"))

FR_SENTENCES = [
    # The ten receptionist sentences of every bake-off, so the ear compares
    # arms with everything it has heard before.
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    from lfm2_audio.core.progress import Progress

    with Progress("voxtral-voices") as progress:
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

        # Server mode — the only path that has ever produced a clip here. The
        # in-process Omni() hung forever after stage-0 warmup on two runs.
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
        progress.note(f"serveur prêt en {time.monotonic() - engine_t0:.0f}s")

        # Run 3's 400 spelled out the contract: URL, data URL, or file URI.
        mime = "audio/wav" if reference.wav_path.suffix == ".wav" else "audio/mpeg"
        data_url = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode()}"
        arms: list[tuple[str, dict[str, str]]] = [
            ("fr_female", {"voice": "fr_female"}),
            ("fr_male", {"voice": "fr_male"}),
            ("clone", {"ref_audio": data_url}),
        ]

        import io

        import soundfile as sf

        def speak(client: httpx.Client, text: str, voice_args: dict[str, str]):  # noqa: ANN202
            payload = {"input": text, "model": vox.MODEL, "response_format": "wav", **voice_args}
            response = client.post(f"{base_url}/audio/speech", json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code}: {response.text[:160]}")
            return sf.read(io.BytesIO(response.content), dtype="float32")

        results = []
        with httpx.Client(timeout=300.0) as client:
            for arm_name, voice_args in arms:
                progress.step(f"bras {arm_name} — {len(FR_SENTENCES)} phrases, {CONCURRENCY} en parallèle")
                arm_dir = OUT / arm_name
                arm_dir.mkdir(parents=True, exist_ok=True)
                batch_t0 = time.monotonic()
                try:
                    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
                        waves = list(pool.map(lambda t, args=voice_args: speak(client, t, args), FR_SENTENCES))
                except RuntimeError as error:
                    progress.note(f"ÉCHEC : {error}")
                    results.append({"arm": arm_name, "error": str(error)[:200]})
                    continue
                batch_seconds = time.monotonic() - batch_t0

                audio_seconds = 0.0
                for index, (text, (wave, rate)) in enumerate(zip(FR_SENTENCES, waves, strict=True)):
                    sf.write(str(arm_dir / f"s{index:02d}.wav"), wave, int(rate), subtype="PCM_16")
                    (arm_dir / f"s{index:02d}.txt").write_text(text, encoding="utf-8")
                    audio_seconds += len(wave) / rate
                rtf = audio_seconds / batch_seconds if batch_seconds else 0.0
                results.append(
                    {
                        "arm": arm_name,
                        "batch_s": round(batch_seconds, 1),
                        "audio_s": round(audio_seconds, 1),
                        "batched_rtf": round(rtf, 2),
                    }
                )
                progress.note(f"{audio_seconds:.0f}s d'audio en {batch_seconds:.1f}s — RTF batché {rtf:.1f}")

        server.terminate()

        progress.step("contre-transcription")
        from lfm2_audio.ds.audio import Waveform
        from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
        from lfm2_audio.scorer.audio.wer import word_error_rate

        transcriber = FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
        for arm_result in results:
            if "error" in arm_result:
                continue
            rates = []
            for index, text in enumerate(FR_SENTENCES):
                wav = OUT / str(arm_result["arm"]) / f"s{index:02d}.wav"
                heard = transcriber.transcribe(Waveform.from_file(str(wav)), language="fr")
                rates.append(word_error_rate(text, heard))
            arm_result["median_wer_fr"] = round(statistics.median(rates), 3)
            progress.note(f"{arm_result['arm']} : WER médian {arm_result['median_wer_fr']}")

        # The projection uses the best measured batched RTF: that is the number
        # the 103.6 h purchase decision rests on.
        rtfs = [r["batched_rtf"] for r in results if "batched_rtf" in r]
        best = max(rtfs) if rtfs else None
        summary = {
            "arms": results,
            "concurrency": CONCURRENCY,
            "best_batched_rtf": best,
            "projected_gpu_h_for_corpus": round(REMAINING_CORPUS_H / best, 1) if best else None,
            "projected_dollars": round(REMAINING_CORPUS_H / best * GPU_DOLLARS_PER_H, 2) if best else None,
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    print("===RESULT voxtral_voices===", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
