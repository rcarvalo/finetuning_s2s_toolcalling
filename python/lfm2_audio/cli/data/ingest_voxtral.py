"""Turn a folder of Voxtral clips (``sample_*.wav`` + ``.json``) into a corpus brick.

Entry point: ``lfm2-ingest-voxtral``. Every clip is re-listened to by an
independent ASR and kept only if it says its text (same rule as brick A);
refusals go to ``dropped.jsonl`` with what was heard. The result is a brick
folder ready for ``lfm2-corpus-push``::

    lfm2-ingest-voxtral --source ../voxtral_data/audio --out data/corpus/E_long_form

Resumable: clips already in the manifest or the rejection log are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lfm2_audio.data_prep.clip_verification import accepted, verification_rates
from lfm2_audio.data_prep.corpus_layout import MANIFEST_NAME, CorpusEntry, read_manifest, write_manifest
from lfm2_audio.data_prep.rejection_log import RejectionLog

if TYPE_CHECKING:
    from collections.abc import Callable

    from avet.scorers.asr.transcriber import Transcriber


@dataclass(frozen=True)
class IngestConfig:
    source: Path
    out: Path
    id_prefix: str = "wiki_fr"
    lang: str = "fr"
    speaker: str = "fr_female"
    source_tag: str = "voxtral-tts-wiki"
    max_wer: float = 0.15
    max_cer: float = 0.15
    limit: int | None = None
    checkpoint_every: int = 50


@dataclass(frozen=True)
class Sample:
    stem: str
    wav: Path
    text: str

    @property
    def clip_id(self) -> str:
        return self.stem.replace("sample_", "")


@dataclass(frozen=True)
class IngestSummary:
    clips: int
    new_clips: int
    dropped: int
    hours: float


class VoxtralFolderIngester:
    """Verifies each clip of a Voxtral folder and builds the brick folder."""

    def __init__(self, config: IngestConfig, transcriber: Transcriber, *, log: Callable[[str], None] = print) -> None:
        self._config = config
        self._transcriber = transcriber
        self._log = log

    def samples(self) -> list[Sample]:
        """Every ``.json`` with its ``.wav`` beside it, in stem order."""
        found = []
        for meta in sorted(self._config.source.glob("sample_*.json")):
            wav = meta.with_suffix(".wav")
            if not wav.exists():
                continue
            payload = json.loads(meta.read_text(encoding="utf-8"))
            found.append(Sample(stem=meta.stem, wav=wav, text=str(payload["text"]).strip()))
        return found[: self._config.limit] if self._config.limit else found

    def run(self) -> IngestSummary:
        from lfm2_audio.ds.audio import Waveform

        out, prefix = self._config.out, self._config.id_prefix
        audio_dir = out / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        manifest = out / MANIFEST_NAME
        kept = list(read_manifest(manifest)) if manifest.exists() else []
        known = {entry.id for entry in kept}
        rejections = RejectionLog(out / "dropped.jsonl").load(out / "dropped.jsonl")
        todo = [
            s
            for s in self.samples()
            if f"{prefix}_{s.clip_id}" not in known and rejections.attempts(f"{prefix}_{s.clip_id}") == 0
        ]
        self._log(f"{len(kept)} clips déjà vérifiés, {len(rejections)} refus connus, {len(todo)} à vérifier")

        new_clips = dropped = 0
        started, audio_seconds = time.time(), 0.0
        for index, sample in enumerate(todo, start=1):
            clip_id = f"{prefix}_{sample.clip_id}"
            waveform = Waveform.from_file(str(sample.wav))
            duration = round(waveform.duration_s, 3)
            audio_seconds += duration
            heard = self._transcriber.transcribe(waveform, language=self._config.lang)
            wer, cer = verification_rates(sample.text, heard, self._config.lang)
            if not accepted(wer, cer, max_wer=self._config.max_wer, max_cer=self._config.max_cer):
                dropped += 1
                rejections.record(clip_id, text=sample.text, heard=heard, wer=wer, cer=cer)
            else:
                shutil.copyfile(sample.wav, audio_dir / f"{clip_id}.wav")
                kept.append(self._entry(clip_id, sample.text, duration, wer, cer))
                new_clips += 1
            if index % self._config.checkpoint_every == 0 or index == len(todo):
                write_manifest(kept, manifest)
                speed = audio_seconds / max(time.time() - started, 1e-6)
                self._log(f"  {index}/{len(todo)} — gardés {new_clips}, écartés {dropped}, {speed:.1f}× temps réel")

        hours = round(sum(entry.duration_s for entry in kept) / 3600, 3)
        return IngestSummary(clips=len(kept), new_clips=new_clips, dropped=dropped, hours=hours)

    def _entry(self, clip_id: str, text: str, duration: float, wer: float, cer: float) -> CorpusEntry:
        return CorpusEntry(
            id=clip_id,
            audio=f"audio/{clip_id}.wav",
            text=text,
            lang=self._config.lang,
            duration_s=duration,
            role="assistant",
            speaker=self._config.speaker,
            source=self._config.source_tag,
            voxtral_wer=round(wer, 4),
            voxtral_cer=round(cer, 4),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="dossier des sample_*.wav + .json")
    parser.add_argument("--out", required=True, type=Path, help="dossier de la brique à construire")
    parser.add_argument("--id-prefix", default="wiki_fr")
    parser.add_argument("--speaker", default="fr_female")
    parser.add_argument("--source-tag", default="voxtral-tts-wiki")
    parser.add_argument("--lang", default="fr", choices=["fr", "en"])
    parser.add_argument("--max-wer", type=float, default=0.15)
    parser.add_argument("--max-cer", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-size", default="small")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.is_dir():
        print(f"❌ dossier source introuvable : {args.source}", file=sys.stderr)
        raise SystemExit(1)
    from avet.scorers.asr.faster_whisper_transcriber import FasterWhisperTranscriber

    compute = "float16" if args.device == "cuda" else "int8"
    transcriber = FasterWhisperTranscriber(model_size=args.model_size, device=args.device, compute_type=compute)
    config = IngestConfig(
        source=args.source,
        out=args.out,
        id_prefix=args.id_prefix,
        lang=args.lang,
        speaker=args.speaker,
        source_tag=args.source_tag,
        max_wer=args.max_wer,
        max_cer=args.max_cer,
        limit=args.limit,
    )
    summary = VoxtralFolderIngester(config, transcriber).run()
    print("===RESULT ingest_voxtral===", flush=True)
    print(json.dumps(summary.__dict__, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
