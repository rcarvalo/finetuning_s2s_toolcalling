"""Run the FR source audit: sample, measure, and report.

Entry point: ``lfm2-fr-audit``. Reads ``configs/audit/fr_sources.yaml``,
samples each source, runs VERSA (pseudo-MOS + NISQA) on the raw-audio ones,
checks label cleanliness with faster-whisper (fr), and writes the comparison
to ``docs/fr_data_audit.md`` plus raw per-clip JSON next to the WAVs.

CPU-friendly by design: whisper `small` on int8 and VERSA's CPU path — the
audit compares sources against each other, not against an absolute bar.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import soxr
import yaml
from datasets import Audio, load_dataset

from lfm2_audio.data_prep.asr_bench import AsrCandidate, AsrClipSelector
from lfm2_audio.data_prep.fr_source_audit import ClipAudit, SourceAudit, audit_markdown
from lfm2_audio.evaluation.versa_runner import MOS_CONFIG, VersaRunner, nisqa_config
from lfm2_audio.scorer.audio.wer import word_error_rate

logger = logging.getLogger(__name__)

TARGET_RATE = 16_000


def audit_source(spec: dict[str, Any], sample_size: int, audio_root: Path, versa: VersaRunner) -> SourceAudit:
    audit = SourceAudit(
        name=spec["name"],
        register=spec.get("register", ""),
        metadata_only=bool(spec.get("metadata_only", False)),
    )
    if spec.get("kind") == "audiofolder":
        rows: Any = _iter_audiofolder(spec, sample_size)
    else:
        rows = load_dataset(spec["repo_id"], spec.get("config"), split=spec["split"], streaming=True)
        if not audit.metadata_only:
            rows = rows.cast_column(spec["audio_column"], Audio(decode=False))

    out_dir = audio_root / audit.name
    out_dir.mkdir(parents=True, exist_ok=True)
    wavs: dict[str, Path] = {}
    # Same diversity guard as the benchmarks: streamed corpora are often
    # ordered by speaker, and the first N clips would audit one voice.
    selector = AsrClipSelector(limit=sample_size, max_per_speaker=spec.get("max_per_speaker", 3))
    for index, row in enumerate(rows):
        if selector.full:
            break
        clip_id = f"{audit.name}_{index:04d}"
        speaker = str(row.get(spec.get("speaker_column", ""), "") or "")
        transcript = str(row.get(spec["text_column"], "") or "")
        if not selector.offer(AsrCandidate(sample_id=clip_id, transcript=transcript, speaker=speaker)):
            continue
        duration = _duration(row, spec, out_dir, clip_id, wavs, metadata_only=audit.metadata_only)
        audit.add(
            ClipAudit(
                sample_id=clip_id,
                duration_s=duration,
                speaker=speaker,
                transcript=transcript,
            )
        )

    if not audit.metadata_only and wavs:
        _measure(audit, wavs, versa)
    return audit


def _duration(
    row: dict[str, Any],
    spec: dict[str, Any],
    out_dir: Path,
    clip_id: str,
    wavs: dict[str, Path],
    *,
    metadata_only: bool,
) -> float | None:
    if metadata_only:
        column = spec.get("duration_column")
        return float(row[column]) if column and row.get(column) is not None else None
    data, rate = sf.read(io.BytesIO(row[spec.get("audio_column", "audio")]["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if rate != TARGET_RATE:
        data = soxr.resample(data, rate, TARGET_RATE)
    path = out_dir / f"{clip_id}.wav"
    sf.write(str(path), data, TARGET_RATE, subtype="PCM_16")
    wavs[clip_id] = path
    return len(data) / TARGET_RATE


def _iter_audiofolder(spec: dict[str, Any], sample_size: int) -> Iterator[dict[str, Any]]:
    """Rows of a wav + metadata.jsonl repo, sampled evenly across the corpus.

    An even stride matters here: audiofolder chunks are often grouped by
    recording batch, and the first N clips would all share one voice.
    """
    from huggingface_hub import hf_hub_download

    manifest = Path(hf_hub_download(spec["repo_id"], spec["metadata_file"], repo_type="dataset"))
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    stride = max(1, len(entries) // sample_size)
    for entry in entries[::stride][:sample_size]:
        wav = Path(hf_hub_download(spec["repo_id"], entry["file_name"], repo_type="dataset"))
        yield {"audio": {"bytes": wav.read_bytes()}, **{k: v for k, v in entry.items() if k != "file_name"}}


def _measure(audit: SourceAudit, wavs: dict[str, Path], versa: VersaRunner) -> None:
    mos = versa.score(wavs, MOS_CONFIG)
    nisqa = versa.score(wavs, nisqa_config(versa.root))
    label_wer = _label_wer(audit, wavs)
    audit.clips = [
        ClipAudit(
            sample_id=c.sample_id,
            duration_s=c.duration_s,
            speaker=c.speaker,
            transcript=c.transcript,
            dnsmos=mos.get(c.sample_id, {}).get("dns_overall"),
            utmos=mos.get(c.sample_id, {}).get("utmos"),
            nisqa=nisqa.get(c.sample_id, {}).get("nisqa_mos_pred"),
            label_wer=label_wer.get(c.sample_id),
        )
        for c in audit.clips
    ]


def _label_wer(audit: SourceAudit, wavs: dict[str, Path]) -> dict[str, float]:
    """WER between the shipped transcript and an independent ASR's reading."""
    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8")
    scores: dict[str, float] = {}
    for clip in audit.clips:
        path = wavs.get(clip.sample_id)
        if path is None or not clip.transcript.strip():
            continue
        segments, _ = model.transcribe(str(path), language="fr", beam_size=1)
        heard = " ".join(segment.text for segment in segments)
        scores[clip.sample_id] = word_error_rate(clip.transcript, heard)
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=Path("configs/audit/fr_sources.yaml"), type=Path)
    parser.add_argument("--out", default=Path("docs/fr_data_audit.md"), type=Path)
    parser.add_argument("--sample-size", type=int, default=None, help="écrase la valeur du YAML (smoke)")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sample_size = args.sample_size or int(config["sample_size"])
    audio_root = Path(config["audio_out"])
    versa = VersaRunner()

    audits: list[SourceAudit] = []
    for spec in config["sources"]:
        logger.info("audit de %s (%d clips)…", spec["name"], sample_size)
        audit = audit_source(spec, sample_size, audio_root, versa)
        audits.append(audit)
        raw = [dataclasses.asdict(clip) for clip in audit.clips]
        (audio_root / audit.name / "clips.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        logger.info("→ %s : %s", audit.name, audit.summary())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_report(audits, sample_size), encoding="utf-8")
    print(f"rapport → {args.out}")


def _report(audits: list[SourceAudit], sample_size: int) -> str:
    intro = (
        "# Audit comparatif des sources FR\n\n"
        f"Échantillon de {sample_size} clips par source (streaming, premiers clips valides).\n"
        "Métriques audio via VERSA (autorité des gates) ; « WER labels » = écart entre le\n"
        "transcript fourni et une ré-écoute faster-whisper small fr (proxy de propreté des\n"
        "labels, pas un gate). `lfm2-bilingual-pilot-125h` n'est pas ré-audité : pré-packé et\n"
        "déjà validé par le pilote (val_loss 2.02). `emilia_yodas_fr` ne distribue que des\n"
        "codes codec : métadonnées seules, décoder pour juger le corpus jugerait le codec.\n\n"
    )
    warning = (
        "\n\n## Recoupement inter-sources\n\n"
        "Les sources Rcarvalo et le dataset étudiant se recoupent (Common Voice FR).\n"
        "L'exclusion du hold-out (benchmark/*/speakers.txt + source_ids.txt) doit être\n"
        "appliquée à TOUTES les sources au moment du mix, pas seulement à la source d'origine.\n"
    )
    return intro + audit_markdown(audits) + warning


if __name__ == "__main__":
    main()
