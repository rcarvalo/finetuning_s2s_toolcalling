"""Does French listening fail, or only French listening WHILE speaking?

The D1 verdict — French ASR at 0.50 median WER against 0.15 in English — was
measured through the normal interleaved path, like everything else that day.
Then the decoding-mode diagnostic showed the model writes clean, full-length
French in text-only mode and contaminated, half-length French when audio is
interleaved. If listening suffers the same way, the 0.50 is an artefact of the
regime, French hearing was never the problem, and rung R1 disappears.

Same 100 FLEURS-fr clips, same prompt, one variable: ``text_only``.

  interleaved  the path D1 was measured on
  text_only    no audio generated at all

An English control runs both ways too: if text-only helps French and leaves
English flat, the gain is about French; if it lifts both, the interleaved path
degrades transcription in general and that is a different finding again.

Transcription is compared against the reference with the same word error rate
the campaigns use, so the numbers sit next to D1's without conversion.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LFM2_ROOT", "/workspace/repo"))
OUT = Path(os.environ.get("LFM2_OUT", "/workspace/out")) / "asr_modes"

sys.path.insert(0, str(ROOT / "python"))

LIMIT = int(os.environ.get("ASR_LIMIT", "100"))
# The instruction must live in the SYSTEM prompt: in the user turn the model
# answers the question instead of transcribing (probe of 25/08).
ASR_SYSTEM = "Transcribe the audio faithfully. Output only the transcription."

BENCHMARKS = {
    "fr": ("benchmark/fleurs_fr_asr/questions.jsonl", "data/benchmark_audio/fleurs_fr"),
    "en": ("benchmark/fleurs_en_asr/questions.jsonl", "data/benchmark_audio/fleurs_en"),
}


def cases(questions: Path, audio_root: Path, limit: int) -> list[tuple[str, Path, str]]:
    out = []
    for line in questions.read_text(encoding="utf-8").splitlines()[:limit]:
        case = json.loads(line)
        user = next(t for t in case["turns"] if t["role"] == "user")
        assistant = next(t for t in case["turns"] if t["role"] == "assistant")
        out.append((case["id"], audio_root / user["audio"], assistant["text"]))
    return out


def build_benchmarks() -> None:
    """FLEURS audio is not in git; rebuild it with the same deterministic CLI."""
    import subprocess

    specs = [
        (
            "fleurs_fr",
            [
                "--repo-id",
                "google/fleurs",
                "--config",
                "fr_fr",
                "--split",
                "test",
                "--text-column",
                "transcription",
                "--id-column",
                "id",
                "--limit",
                "200",
                "--prefix",
                "fleurs_fr",
                "--lang",
                "fr",
                "--out",
                "benchmark/fleurs_fr_asr",
                "--audio-out",
                "data/benchmark_audio/fleurs_fr",
            ],
        ),
        (
            "fleurs_en",
            [
                "--repo-id",
                "google/fleurs",
                "--config",
                "en_us",
                "--split",
                "test",
                "--text-column",
                "transcription",
                "--id-column",
                "id",
                "--limit",
                "100",
                "--prefix",
                "fleurs_en",
                "--lang",
                "en",
                "--out",
                "benchmark/fleurs_en_asr",
                "--audio-out",
                "data/benchmark_audio/fleurs_en",
            ],
        ),
    ]
    for name, args in specs:
        if (ROOT / "data" / "benchmark_audio" / name).exists():
            continue
        print(f"construction de {name}…", flush=True)
        subprocess.run([sys.executable, "-m", "lfm2_audio.cli.data.make_asr_bench", *args], cwd=str(ROOT), check=False)


def transcribe(lang: str, text_only: bool) -> dict[str, object]:
    from lfm2_audio.ds.audio import Waveform
    from lfm2_audio.ds.generation_config import GenerationConfig
    from lfm2_audio.scorer.audio.wer import word_error_rate
    from lfm2_audio.serving.model import LFM2Audio

    questions, audio_root = BENCHMARKS[lang]
    items = cases(ROOT / questions, ROOT / audio_root, LIMIT)
    mode = "text_only" if text_only else "interleaved"
    model = LFM2Audio.from_pretrained(
        "LiquidAI/LFM2.5-Audio-1.5B",
        backend="liquid",
        system=ASR_SYSTEM,
        generation=GenerationConfig(text_only=text_only),
    )
    rates, rows = [], []
    with model:
        for case_id, wav, reference in items:
            model.reset()
            reply = model.reply(audio=Waveform.from_file(str(wav)))
            hypothesis = reply.text or ""
            rate = word_error_rate(reference, hypothesis)
            rates.append(rate)
            rows.append({"id": case_id, "wer": round(rate, 4), "hypothesis": hypothesis[:200]})

    # Median, never mean: a few samples degenerate into repetition loops and a
    # mean would mostly report how often that happened.
    summary = {
        "lang": lang,
        "mode": mode,
        "n": len(rates),
        "median_wer": round(statistics.median(rates), 4) if rates else None,
        "clean_rate": round(sum(1 for r in rates if r <= 0.30) / len(rates), 3) if rates else None,
        "loop_rate": round(sum(1 for r in rates if r > 1.0) / len(rates), 3) if rates else None,
    }
    (OUT / f"{lang}_{mode}.json").write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def claim_cuda() -> None:
    """Create the CUDA context before anything forks.

    This job builds its benchmarks through subprocesses and only then loads the
    model — and it died twice on ``CUDA unknown error`` at that point, on a
    fresh pod as well as a restarted one, while the bake-off job (which touches
    CUDA first and forks nothing) never did. Claiming the device up front costs
    nothing and removes the ordering as a suspect; if the GPU is genuinely
    unavailable, it now fails here with a clear message instead of after the
    downloads.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("aucun GPU visible — inutile de télécharger les benchmarks")
    torch.zeros(1, device="cuda")
    print(f"GPU : {torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    claim_cuda()
    build_benchmarks()

    results = []
    for lang in ("fr", "en"):
        for text_only in (True, False):
            mode = "text_only" if text_only else "interleaved"
            done = OUT / f"{lang}_{mode}.json"
            if done.exists():
                results.append(json.loads(done.read_text(encoding="utf-8"))["summary"])
                print(f"{lang}/{mode} déjà fait", flush=True)
                continue
            results.append(transcribe(lang, text_only))

    (OUT / "asr_modes.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("===RESULT asr_modes.json===", flush=True)
    print(json.dumps(results, ensure_ascii=False), flush=True)
    print("===END===", flush=True)


if __name__ == "__main__":
    main()
