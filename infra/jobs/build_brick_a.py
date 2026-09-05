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
import time
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


WAIT_SOURCES_MIN = int(os.environ.get("BRICK_A_WAIT_SOURCES_MIN", "0"))
"""How long to wait for a source still being produced upstream (the merged
dialogue file lands on the Hub when its text campaign ends). 0 = skip it with
a warning, as before; a silently skipped source once ended a run at a tenth
of its work while the pod's bootstrap had already been paid."""


def _resolve_source(source: Path, *, sleep=time.sleep) -> Path | None:  # noqa: ANN001 — Callable[[float], None]
    """A source lives in git if small, on the corpus HF repo otherwise.

    The 2.7 MB TC corpus tripped the repo's large-file hook — data does not
    belong in git. The pod pulls it from the Hub by the same relative path.
    """
    if source.exists():
        return source
    from huggingface_hub import hf_hub_download

    relative = source.relative_to(ROOT / "corpus")
    deadline = WAIT_SOURCES_MIN
    while True:
        try:
            return Path(hf_hub_download("Rcarvalo/lfm25-fr-corpus-v1", str(relative), repo_type="dataset"))
        except Exception as error:  # absent, or the Hub is down: both mean "not now"
            if deadline <= 0:
                print(f"source absente (git ET hub), ignorée : {source} ({type(error).__name__})", flush=True)
                return None
            print(f"source {relative} pas encore sur le Hub — attente ({deadline} min restantes)", flush=True)
            sleep(60)
            deadline -= 1


ROLE = os.environ.get("BRICK_A_ROLE", "assistant")
"""Which side of the dialogue to voice. ``user`` feeds the fr_asr_user slice:
what the model must HEAR, in voices other than the assistant's."""

SHARD = os.environ.get("BRICK_A_SHARD", "")
"""``k/n`` takes every n-th turn starting at k. Lets the user-speech corpus be
spoken by several preset voices without per-clip plumbing: one run per voice,
each on its own shard, each manifest recording its voice."""


KINDS = {k for k in os.environ.get("BRICK_A_KINDS", "").split(",") if k}
SKIP_KINDS = {k for k in os.environ.get("BRICK_A_SKIP_KINDS", "").split(",") if k}
"""Dialogue families (``meta.kind``) to keep / to leave out. The merged v2 file
mixes French families with the English preservation share, and those go to
different bricks: one run per brick, each told which families are its own."""


def wanted(case: dict) -> bool:  # a JSONL row
    kind = str(case.get("meta", {}).get("kind", ""))
    if KINDS and kind not in KINDS:
        return False
    return kind not in SKIP_KINDS


def turns_to_speak(limit: int | None) -> list[tuple[str, str, str]]:
    """``(clip_id, text, lang)`` for every turn of ``ROLE`` to synthesise."""
    items: list[tuple[str, str, str]] = []
    for declared in SOURCES:
        source = _resolve_source(declared)
        if source is None:
            continue
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            if not wanted(case):
                continue
            lang = str(case.get("meta", {}).get("lang", "fr"))
            for index, turn in enumerate(case.get("turns", [])):
                if turn.get("role") != ROLE or not turn.get("text", "").strip():
                    continue
                items.append((f"{case['id']}_t{index}", turn["text"].strip(), turn.get("lang", lang)))
    if SHARD:
        offset, stride = (int(p) for p in SHARD.split("/"))
        items = items[offset::stride]
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

    Failure paths are exercised, not hoped away (tests/test_voxtral_client.py):
    a request is retried then skipped, a dead server is restarted.
    """
    import base64

    import httpx

    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    import voxtral_tts_synth as vox
    from _voxtral_client import ServerGuard, VoxtralClient

    from lfm2_audio.core.progress import Progress

    progress = Progress("voxtral")
    vox.install_stack(progress)
    progress.step("préchargement des bibliothèques CUDA 13")
    vox.preload_cuda13()
    progress.step("démarrage du serveur vllm serve --omni (poids ~8 Go)")
    guard = ServerGuard(start=lambda: vox.start_server(progress), note=progress.note)
    guard.ensure_alive()
    progress.step("serveur prêt — synthèse")

    if VOICE_SOURCE not in ("dialogue", "siwis"):  # tout preset embarqué du dépôt Voxtral
        voice_args = {"voice": VOICE_SOURCE}
    else:
        mime = "audio/wav" if reference.wav_path.suffix == ".wav" else "audio/mpeg"
        voice_args = {"ref_audio": f"data:{mime};base64,{base64.b64encode(reference.audio_bytes).decode()}"}

    client = VoxtralClient(
        http=httpx.Client(timeout=300.0),
        model=vox.MODEL,
        voice_args=voice_args,
        concurrency=int(os.environ.get("BRICK_A_CONCURRENCY", "16")),
    )

    def speak(texts: list[str], langs: list[str]):  # noqa: ANN202 — langue portée par le texte
        client.base_url = guard.ensure_alive()
        waves, rate = client.speak(texts)
        return waves, rate or vox.SAMPLE_RATE

    return speak


def merge_existing(manifest: Path | None):  # noqa: ANN201 — list[CorpusEntry]
    """The brick's entries already on the Hub, to be kept in the manifest we push.

    Every push re-sends the whole manifest, and the local one only knew this
    run's clips: a relaunch would have shrunk the brick to its newest wave and
    orphaned thousands of clips. Start from what is there.
    """
    from lfm2_audio.data_prep.corpus_layout import read_manifest

    if manifest is None or not manifest.exists():
        return []
    return list(read_manifest(manifest))


def hub_manifest() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download

        repo = os.environ.get("BRICK_A_HF_REPO", "Rcarvalo/lfm25-fr-corpus-v1")
        return Path(hf_hub_download(repo, f"{BRICK_PATH}/manifest.jsonl", repo_type="dataset"))
    except Exception as error:  # absent brick, or Hub down: start from nothing, say so
        print(f"manifeste Hub absent ou illisible ({type(error).__name__}) — la brique repart de zéro", flush=True)
        return None


def main() -> None:
    import logging

    # The pusher's messages ARE the job's safety record: they must reach the
    # pod log, which is all that survives a deleted pod.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    audio_dir = OUT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Fail in two seconds, not after eight gigabytes: a community pod sometimes
    # boots without a working GPU (measured 27/08).
    sys.path.insert(0, str(ROOT / "infra" / "jobs"))
    from voxtral_tts_synth import claim_cuda

    claim_cuda()

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
    if VOICE_SOURCE not in ("dialogue", "siwis"):  # tout preset embarqué du dépôt Voxtral
        reference = None
        print(f"voix : preset natif {VOICE_SOURCE} [{ENGINE}], rôle {ROLE}", flush=True)
    else:
        reference = resolve_voice_reference(VOICE_SOURCE)
        print(f"voix : {VOICE_SOURCE}/{reference.stem} [{ENGINE}] — « {reference.text[:60]} »", flush=True)

    # The Hub is the primary store, verified BEFORE any GPU spending — and the
    # ONLY resume point: Colab sessions die within hours and their disks with
    # them, so "already done" is defined by the repo, not the local folder.
    from lfm2_audio.data_prep.streaming_push import StreamingPusher

    pusher = StreamingPusher(OUT, os.environ.get("BRICK_A_HF_REPO", "Rcarvalo/lfm25-fr-corpus-v1"), BRICK_PATH)
    pusher.verify()
    on_hub = pusher.preload_existing()
    push_every = int(os.environ.get("BRICK_A_PUSH_EVERY", "5"))

    kept = merge_existing(hub_manifest())
    known = {entry.id for entry in kept}
    print(f"manifeste Hub : {len(kept)} entrées conservées", flush=True)

    items = turns_to_speak(LIMIT)
    todo = [
        (cid, text, lang)
        for cid, text, lang in items
        if cid not in known and not (audio_dir / f"{cid}.wav").exists() and f"{cid}.wav" not in on_hub
    ]
    print(f"{len(items)} tours {ROLE}, {len(items) - len(todo)} déjà faits, {len(todo)} à produire", flush=True)

    transcriber = FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")
    speak = voxtral_synthesiser(reference) if ENGINE == "voxtral" else qwen_synthesiser(reference)

    new_clips, dropped, missing, rates = 0, 0, 0, []
    for start in range(0, len(todo), BATCH):
        chunk = todo[start : start + BATCH]
        waves, sample_rate = speak([text for _, text, _ in chunk], [lang for _, _, lang in chunk])
        for (clip_id, text, lang), wave in zip(chunk, waves, strict=False):
            if wave is None:
                missing += 1  # counted, logged by the client, never fatal
                continue
            path = audio_dir / f"{clip_id}.wav"
            sf.write(str(path), wave, sample_rate, subtype="PCM_16")
            heard = transcriber.transcribe(Waveform.from_file(str(path)), language=lang)
            rate = word_error_rate(text, heard)
            rates.append(rate)
            if rate > MAX_VERIFY_WER:
                path.unlink()
                dropped += 1
                continue
            new_clips += 1
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
            f"  {start + len(chunk)}/{len(todo)} — gardés {new_clips}, écartés {dropped}, sans réponse {missing}",
            flush=True,
        )
        write_manifest(kept, OUT / "manifest.jsonl")
        if (start // BATCH) % push_every == push_every - 1:
            pusher.push(message=f"brick A [{VOICE_SOURCE}]: {len(kept)} clips")

    summary = {
        "engine": ENGINE,
        "clips": len(kept),
        "new_clips": new_clips,
        "dropped": dropped,
        "missing": missing,
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
