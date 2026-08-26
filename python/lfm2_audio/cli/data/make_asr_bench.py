"""Build an ASR benchmark (JSONL + WAV) from a HF audio dataset.

Entry point: ``lfm2-asr-bench``. The JSONL goes under ``benchmark/`` (in git);
the WAVs go to ``--audio-out`` (never in git). Two exclusion files are written
next to the JSONL so the training mixer can hold the benchmark out of EVERY
source — the user's own datasets and the student's overlap (both draw from
Common Voice FR), so excluding by speaker in one source is not enough:
``speakers.txt`` (selected speakers) and ``source_ids.txt`` (original clip ids
via ``--id-column``).

    lfm2-asr-bench --repo-id google/fleurs --config fr_fr --split test \\
      --text-column transcription --id-column id --limit 200 --prefix fleurs_fr \\
      --out benchmark/fleurs_fr_asr --audio-out data/benchmark_audio/fleurs_fr

    lfm2-asr-bench --repo-id baptistefrancois1/s2s-fr-finetuning \\
      --config common_voice_fr --split train --text-column transcript \\
      --id-column clip_id --speaker-column speaker_id \\
      --score-column distillmos --min-score 3.5 --max-per-speaker 3 \\
      --limit 300 --prefix cv_fr \\
      --out benchmark/cv_fr_asr --audio-out data/benchmark_audio/cv_fr
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
from datasets import Audio, load_dataset

from lfm2_audio.data_prep.asr_bench import AsrCandidate, AsrClipSelector, asr_dialogue

TARGET_RATE = 16_000
"""What the mel encoder expects; re-encoding once here beats doing it per eval."""


def build(args: argparse.Namespace) -> int:
    rows = load_dataset(args.repo_id, args.config, split=args.split, streaming=True)
    # decode=False: datasets 5.x decodes through torchcodec, which the shared
    # venv does not carry; soundfile reads the raw bytes just as well.
    rows = rows.cast_column(args.audio_column, Audio(decode=False))

    out = Path(args.out)
    audio_out = Path(args.audio_out)
    out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    selector = AsrClipSelector(
        limit=args.limit,
        min_score=args.min_score if args.score_column else None,
        max_per_speaker=args.max_per_speaker,
    )
    source_ids: list[str] = []
    with (out / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if selector.full:
                break
            candidate = _to_candidate(row, index, args)
            if not selector.offer(candidate):
                continue
            wav_name = f"{candidate.sample_id}.wav"
            _write_wav(audio_out / wav_name, row[args.audio_column])
            case = asr_dialogue(candidate.sample_id, candidate.transcript, wav_name, args.lang)
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
            if args.id_column:
                source_ids.append(str(row.get(args.id_column, "")))

    (out / "speakers.txt").write_text("".join(f"{s}\n" for s in sorted(selector.speakers)), encoding="utf-8")
    if args.id_column:
        (out / "source_ids.txt").write_text("".join(f"{i}\n" for i in source_ids), encoding="utf-8")
    print(f"{selector.accepted}/{args.limit} clips → {out / 'questions.jsonl'}")
    print(f"{len(selector.speakers)} locuteurs held-out → {out / 'speakers.txt'}")
    print(f"audio (hors git) → {audio_out}")
    return 0 if selector.full else 1


def _write_wav(path: Path, audio: dict[str, Any]) -> None:
    data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if rate != TARGET_RATE:
        data = soxr.resample(data, rate, TARGET_RATE)
    sf.write(str(path), data, TARGET_RATE, subtype="PCM_16")


def _to_candidate(row: dict[str, Any], index: int, args: argparse.Namespace) -> AsrCandidate:
    score = row.get(args.score_column) if args.score_column else None
    speaker = str(row.get(args.speaker_column, "")) if args.speaker_column else ""
    return AsrCandidate(
        sample_id=f"{args.prefix}_{index:05d}",
        transcript=str(row.get(args.text_column, "") or ""),
        speaker=speaker,
        score=float(score) if score is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--id-column", default=None, help="colonne d'id source (traçabilité anti-recoupement)")
    parser.add_argument("--speaker-column", default=None)
    parser.add_argument("--score-column", default=None, help="colonne qualité (ex. distillmos)")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--max-per-speaker", type=int, default=None)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--prefix", required=True, help="préfixe des ids (ex. fleurs_fr)")
    parser.add_argument("--lang", default="fr")
    parser.add_argument("--out", required=True, help="dossier du questions.jsonl (dans git)")
    parser.add_argument("--audio-out", required=True, help="dossier des WAV (hors git)")
    args = parser.parse_args()
    raise SystemExit(build(args))


if __name__ == "__main__":
    main()
