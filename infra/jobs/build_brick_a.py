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
# One folder and one Hub path per brick: the EN wave pushing into
# A_assistant_speech overwrote the French manifest with its own (29/08).
BRICK_PATH = os.environ.get("BRICK_A_HF_PATH", "A_assistant_speech")
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / BRICK_PATH

sys.path.insert(0, str(ROOT / "python"))

# Versioned under corpus/, not data/: data/ is gitignored and the pod, which
# clones the repo, found nothing there. Every source is spoken by the SAME
# voice — one assistant identity across conversation and tool calling.
SOURCES = [
    ROOT / p
    for p in os.environ.get("BRICK_A_SOURCES", "corpus/C_dialogues/dialogues.jsonl,corpus/TC_fr/tc_fr_v1.jsonl").split(
        ","
    )
]
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SAMPLE_RATE = 24_000
LIMIT = int(os.environ.get("BRICK_A_LIMIT", "0")) or None
BATCH = int(os.environ.get("BRICK_A_BATCH", "32"))

VOICE_SOURCE = os.environ.get("BRICK_A_VOICE", "dialogue")
"""``dialogue`` won the reference bake-off by ear: SIWIS is read speech and its
clone was judged robotic, while a conversational clip gives a spoken register."""

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


def _resolve_source(source: Path) -> Path | None:
    """A source lives in git if small, on the corpus HF repo otherwise.

    The 2.7 MB TC corpus tripped the repo's large-file hook — data does not
    belong in git. The pod pulls it from the Hub by the same relative path.
    """
    if source.exists():
        return source
    try:
        from huggingface_hub import hf_hub_download

        relative = source.relative_to(ROOT / "corpus")
        return Path(hf_hub_download("Rcarvalo/lfm25-fr-corpus-v1", str(relative), repo_type="dataset"))
    except Exception as error:
        print(f"source absente (git ET hub), ignorée : {source} ({error})", flush=True)
        return None


def turns_to_speak(limit: int | None) -> list[tuple[str, str, str]]:
    """``(clip_id, text, lang)`` for every assistant turn to synthesise."""
    items: list[tuple[str, str, str]] = []
    for declared in SOURCES:
        source = _resolve_source(declared)
        if source is None:
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            lang = str(case.get("meta", {}).get("lang", "fr"))
            for index, turn in enumerate(case.get("turns", [])):
                if turn.get("role") != "assistant" or not turn.get("text", "").strip():
                    continue
                items.append((f"{case['id']}_t{index}", turn["text"].strip(), turn.get("lang", lang)))
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
    """Voxtral-4B-TTS in server mode, requests fired concurrently.

    Server, never in-process: ``Omni()`` hung forever after stage-0 warmup on
    two runs. And concurrency is not an optimisation, it is the price of the
    corpus: sequential posting measured RTF 4.26 while the model card reaches
    0.302 at concurrency 32 — an order of magnitude on the GPU bill.

    The voice: ``fr_female``/``fr_male`` ride the native preset embeddings;
    anything else clones ``reference`` through a ref_audio data URL.
    """
    import base64
    from concurrent.futures import ThreadPoolExecutor

    import httpx

    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    import voxtral_tts_synth as vox

    from lfm2_audio.core.progress import Progress

    progress = Progress("voxtral")
    vox.install_stack(progress)
    progress.step("préchargement des bibliothèques CUDA 13")
    vox.preload_cuda13()
    progress.step("démarrage du serveur vllm serve --omni (poids ~8 Go)")
    _server, base_url = vox.start_server(progress)  # gardé : le processus vit tant que le job vit
    progress.step("serveur prêt — synthèse")

    if VOICE_SOURCE in ("fr_female", "fr_male"):
        voice_args = {"voice": VOICE_SOURCE}
    else:
        mime = "audio/wav" if reference.wav_path.suffix == ".wav" else "audio/mpeg"
        voice_args = {"ref_audio": f"data:{mime};base64,{base64.b64encode(reference.audio_bytes).decode()}"}

    concurrency = int(os.environ.get("BRICK_A_CONCURRENCY", "16"))
    client = httpx.Client(timeout=300.0)

    def one(text: str):  # noqa: ANN202 — (ndarray, int)
        import io

        import soundfile as sf

        payload = {"input": text, "model": vox.MODEL, "response_format": "wav", **voice_args}
        response = client.post(f"{base_url}/audio/speech", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"{response.status_code}: {response.text[:160]}")
        return sf.read(io.BytesIO(response.content), dtype="float32")

    def speak(texts: list[str], langs: list[str]):  # noqa: ANN202 — langue portée par le texte
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, texts))
        rate = int(results[0][1]) if results else vox.SAMPLE_RATE
        return [wave for wave, _ in results], rate

    return speak


def main() -> None:
    import logging

    # The pusher's messages ARE the job's safety record: they must reach the
    # pod log, which is all that survives a deleted pod.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    from lfm2_audio.data_prep.corpus_layout import CorpusEntry, write_manifest
    from lfm2_audio.data_prep.voice_reference import resolve_voice_reference
    from lfm2_audio.ds.audio import Waveform
    from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber
    from lfm2_audio.scorer.audio.wer import word_error_rate

    # A preset voice needs no reference at all — cloning is impossible on the
    # open Voxtral checkpoint anyway (no encoder weights). A clone source is
    # resolved before any engine install: the vLLM stack replaces torch and
    # breaks the torchaudio the resolution needs.
    if VOICE_SOURCE in ("fr_female", "fr_male"):
        reference = None
        print(f"voix : preset natif {VOICE_SOURCE} [{ENGINE}]", flush=True)
    else:
        reference = resolve_voice_reference(VOICE_SOURCE)
        print(f"voix : {VOICE_SOURCE}/{reference.stem} [{ENGINE}] — « {reference.text[:60]} »", flush=True)

    items = turns_to_speak(LIMIT)
    todo = [(cid, text, lang) for cid, text, lang in items if not (audio_dir / f"{cid}.wav").exists()]
    print(f"{len(items)} tours assistant, {len(items) - len(todo)} déjà faits, {len(todo)} à produire", flush=True)

    # The Hub is the primary store, verified BEFORE any GPU spending: a RunPod
    # balance reaching zero deletes the pod and its disk, and this run is meant
    # to be left alive until exactly that happens.
    from lfm2_audio.data_prep.streaming_push import StreamingPusher

    pusher = StreamingPusher(OUT, os.environ.get("BRICK_A_HF_REPO", "Rcarvalo/lfm25-fr-corpus-v1"), BRICK_PATH)
    pusher.verify()
    push_every = int(os.environ.get("BRICK_A_PUSH_EVERY", "5"))

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
                    speaker=VOICE_SOURCE if reference is None else f"{VOICE_SOURCE}_{reference.stem}",
                    source=f"{ENGINE}-tts-clone",
                    voxtral_wer=round(rate, 4),
                )
            )
        print(
            f"  {start + len(chunk)}/{len(todo)} — gardés {len(kept)}, écartés {dropped}",
            flush=True,
        )
        write_manifest(kept, OUT / "manifest.jsonl")
        if (start // BATCH) % push_every == push_every - 1:
            pusher.push(message=f"brick A [{VOICE_SOURCE}]: {len(kept)} clips")

    summary = {
        "engine": ENGINE,
        "clips": len(kept),
        "dropped": dropped,
        "hours": round(sum(e.duration_s for e in kept) / 3600, 3),
        "median_verify_wer": round(statistics.median(rates), 4) if rates else None,
        "voice": VOICE_SOURCE if reference is None else reference.stem,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    pusher.push(message=f"brick A [{VOICE_SOURCE}]: final — {len(kept)} clips, {summary['hours']} h")
    print("===RESULT brick_a===", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
