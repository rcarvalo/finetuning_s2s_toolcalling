"""Assemble the FR corpus from what already exists, verified clip by clip.

The bake-off asked which voice to *synthesise*. Measuring the answer exposed a
better one: `french-dialogue-tts-1000h` already holds **26.4 h in a single
voice** (`fr_female`, 15 164 clips, conversational register, 4.4 s median) — and
that voice is the one the listening test retained. Ingesting it beats
synthesising an hour and a half of the same thing.

  A_assistant_speech  ingest dialogue-1000h (26 h) — what the model must SAY
  B_user_speech       ingest the student's Common Voice — what it must HEAR
  D_english           the English share that protects the frozen anchors

Brick C is text and already published; the turns it holds are synthesised
separately by ``build_brick_a`` and merge into the same manifest.

**Every clip is transcribed back and compared to its text.** Brick A because
the model imitates what it is shown, so a clip that drifts teaches drift; brick
B because a label the audio does not say teaches the wrong mapping. The audit
measured 13 % label WER on the student corpus with a small ASR — that number is
the reason this filter exists.

Resumable: each brick writes its manifest as it goes and skips what it has.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out"))

sys.path.insert(0, str(ROOT / "python"))

BRICKS = os.environ.get("CORPUS_BRICKS", "A,B").split(",")
TARGET_HOURS_A = float(os.environ.get("CORPUS_HOURS_A", "30"))
TARGET_HOURS_B = float(os.environ.get("CORPUS_HOURS_B", "15"))
MAX_VERIFY_WER = float(os.environ.get("CORPUS_MAX_WER", "0.20"))
"""A clip whose transcript the audio does not say is dropped.

Loose enough to absorb the checking ASR's own errors, tight enough to catch a
clip that says something else. Brick A synthesis uses a stricter 0.15: there we
control the text exactly, so any gap is the TTS drifting.
"""

DIALOGUE_REPO = "Rcarvalo/french-dialogue-tts-1000h"
CV_REPO = "baptistefrancois1/s2s-fr-finetuning"
MIN_DISTILLMOS = float(os.environ.get("CORPUS_MIN_MOS", "3.5"))
MAX_PER_SPEAKER = int(os.environ.get("CORPUS_MAX_PER_SPEAKER", "40"))


def ensure_deps() -> None:
    """The two packages the pod's extras do not carry.

    ``pod_entrypoint.sh`` installs ``[serving-liquid,eval,inspect]``, which
    brings the ASR but neither the HF streaming reader nor the resampler.
    """
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets>=3.0", "soxr>=0.5"], check=False)


def transcriber():  # noqa: ANN201 — type interne au job
    from lfm2_audio.scorer.audio.faster_whisper_transcriber import FasterWhisperTranscriber

    return FasterWhisperTranscriber(model_size="small", device="cuda", compute_type="float16")


def verify(check, wav: Path, text: str, lang: str) -> float:  # noqa: ANN001 — transcripteur
    from lfm2_audio.ds.audio import Waveform
    from lfm2_audio.scorer.audio.wer import word_error_rate

    heard = check.transcribe(Waveform.from_file(str(wav)), language=lang)
    return word_error_rate(text, heard)


def brick_a(check) -> dict[str, object]:  # noqa: ANN001 — transcripteur
    """Ingest the single-voice dialogue corpus: the assistant's speech."""
    import soundfile as sf
    from huggingface_hub import snapshot_download

    from lfm2_audio.data_prep.corpus_layout import CorpusEntry, read_manifest, write_manifest

    target = OUT / "A_assistant_speech"
    audio_dir = target / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.jsonl"

    kept = list(read_manifest(manifest_path)) if manifest_path.exists() else []
    if kept:
        print(f"[A] reprise : {len(kept)} clips déjà retenus", flush=True)
    seen = {entry.id for entry in kept}
    hours = sum(entry.duration_s for entry in kept) / 3600
    dropped, rates = 0, []

    # One bulk snapshot, not 15 164 file requests: the per-file round-trip
    # dominates at this count, and the HF cache makes a restart free.
    print("[A] téléchargement du corpus dialogue…", flush=True)
    local = Path(snapshot_download(DIALOGUE_REPO, repo_type="dataset"))
    print(f"[A] corpus local : {local}", flush=True)

    for line in (local / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if hours >= TARGET_HOURS_A:
            break
        if not line.strip():
            continue
        row = json.loads(line)
        clip_id = f"a_{Path(row['file_name']).stem}"
        text = " ".join(str(row.get("text", "")).split())
        source_wav = local / row["file_name"]
        if clip_id in seen or len(text) < 20 or not source_wav.exists():
            continue

        rate = verify(check, source_wav, text, "fr")
        rates.append(rate)
        if rate > MAX_VERIFY_WER:
            dropped += 1
            continue

        (audio_dir / f"{clip_id}.wav").write_bytes(source_wav.read_bytes())
        duration = sf.info(str(source_wav)).duration
        kept.append(
            CorpusEntry(
                id=clip_id,
                audio=f"audio/{clip_id}.wav",
                text=text,
                lang="fr",
                duration_s=round(duration, 3),
                role="assistant",
                speaker=str(row.get("voice", "fr_female")),
                source="french-dialogue-tts-1000h",
                voxtral_wer=round(rate, 4),
            )
        )
        hours += duration / 3600
        if len(kept) % 100 == 0:
            write_manifest(kept, manifest_path)
            print(f"[A] {len(kept)} clips · {hours:.2f} h · {dropped} écartés", flush=True)

    write_manifest(kept, manifest_path)
    return {
        "brick": "A",
        "clips": len(kept),
        "hours": round(hours, 2),
        "dropped": dropped,
        "median_wer": round(statistics.median(rates), 4) if rates else None,
    }


def brick_b(check) -> dict[str, object]:  # noqa: ANN001 — transcripteur
    """Ingest the student's Common Voice: what the model must hear.

    Requirements are the opposite of brick A's — maximum speaker diversity, not
    one voice — so a per-speaker cap applies, and the held-out benchmark
    speakers and clip ids are excluded across every source, because the FR
    corpora overlap on Common Voice.
    """
    import io

    import numpy as np
    import soundfile as sf
    import soxr
    from datasets import Audio, load_dataset

    from lfm2_audio.data_prep.asr_bench import AsrCandidate, AsrClipSelector
    from lfm2_audio.data_prep.corpus_layout import CorpusEntry, write_manifest
    from lfm2_audio.data_prep.holdout import HoldoutFilter

    target = OUT / "B_user_speech"
    audio_dir = target / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    holdout = HoldoutFilter.from_benchmarks(
        [ROOT / "benchmark/cv_fr_asr", ROOT / "benchmark/fleurs_fr_asr", ROOT / "benchmark/fr_s2s"]
    )
    print(f"[B] hold-out : {len(holdout.speakers)} locuteurs, {len(holdout.source_ids)} ids", flush=True)

    # The quality + diversity rule is the benchmark builder's, reused rather than
    # restated: brick B and the held-out benchmark must select alike, or "held
    # out" stops meaning anything. Its budget counts clips, so the hour target is
    # converted at the corpus's own 4.4 s median.
    selector = AsrClipSelector(
        limit=int(TARGET_HOURS_B * 3600 / 4.4),
        min_score=MIN_DISTILLMOS,
        max_per_speaker=MAX_PER_SPEAKER,
    )

    kept: list[CorpusEntry] = []
    hours, dropped, rates = 0.0, 0, []

    rows = load_dataset(CV_REPO, "common_voice_fr", split="train", streaming=True)
    rows = rows.cast_column("audio", Audio(decode=False))
    for index, row in enumerate(rows):
        if hours >= TARGET_HOURS_B or selector.full:
            break
        text = " ".join(str(row.get("transcript", "") or "").split())
        speaker = str(row.get("speaker_id", "") or "")
        if holdout.excludes({"speaker": speaker, "id": str(row.get("clip_id", "") or ""), "text": text}):
            continue
        score = row.get("distillmos")
        candidate = AsrCandidate(
            sample_id=f"b_{index:06d}",
            transcript=text,
            speaker=speaker,
            score=float(score) if score is not None else None,
        )
        if len(text) < 15 or not selector.offer(candidate):
            continue

        destination = audio_dir / f"{candidate.sample_id}.wav"
        data, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if rate != 16_000:
            data = soxr.resample(data, rate, 16_000)
        sf.write(str(destination), data, 16_000, subtype="PCM_16")

        wer = verify(check, destination, text, "fr")
        rates.append(wer)
        if wer > MAX_VERIFY_WER:
            destination.unlink()
            dropped += 1
            continue

        duration = len(data) / 16_000
        kept.append(
            CorpusEntry(
                id=candidate.sample_id,
                audio=f"audio/{candidate.sample_id}.wav",
                text=text,
                lang="fr",
                duration_s=round(duration, 3),
                role="user",
                speaker=speaker,
                source="s2s-fr-finetuning/common_voice_fr",
                voxtral_wer=round(wer, 4),
                extra={"distillmos": round(float(score), 3)} if score is not None else {},
            )
        )
        hours += duration / 3600
        if len(kept) % 100 == 0:
            write_manifest(kept, target / "manifest.jsonl")
            print(
                f"[B] {len(kept)} clips · {hours:.2f} h · {len(selector.speakers)} locuteurs · {dropped} écartés",
                flush=True,
            )

    write_manifest(kept, target / "manifest.jsonl")
    return {
        "brick": "B",
        "clips": len(kept),
        "hours": round(hours, 2),
        "speakers": len(selector.speakers),
        "dropped": dropped,
        "median_wer": round(statistics.median(rates), 4) if rates else None,
        "holdout": holdout.stats.summary(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_deps()
    check = transcriber()
    results = []
    for brick in BRICKS:
        builder = {"A": brick_a, "B": brick_b}.get(brick.strip().upper())
        if builder is None:
            continue
        results.append(builder(check))
        (OUT / "corpus_summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    print("===RESULT corpus_summary.json===", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
