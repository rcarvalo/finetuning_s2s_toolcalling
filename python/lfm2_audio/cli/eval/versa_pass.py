"""Run the VERSA authority pass over a campaign's generated speech.

Entry point: ``lfm2-eval-versa``. Every gate of the bilingual plan is decided
on VERSA numbers; this makes that one command instead of a bespoke script per
gate, and it always scores the audio stored in the log rather than
re-generating it.

    lfm2-eval-versa --log-dir logs_0b/fr_s2s --out reports/versa_fr_s2s.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lfm2_audio.core.errors import Lfm2AudioError
from lfm2_audio.evaluation.eval_log_audio import extract_replies, latest_log
from lfm2_audio.evaluation.versa_gate import build_report
from lfm2_audio.evaluation.versa_runner import MOS_CONFIG, VersaRunner, nisqa_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", required=True, type=Path, help="dossier de logs Inspect d'une campagne")
    parser.add_argument("--out", type=Path, default=None, help="JSON de sortie (défaut : stdout seul)")
    parser.add_argument("--audio-out", type=Path, default=Path("data/versa_pass"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--versa-root", type=Path, default=None, help="racine de l'install VERSA")
    parser.add_argument("--skip-nisqa", action="store_true", help="pseudo-MOS seul (plus rapide)")
    args = parser.parse_args()

    runner = VersaRunner(args.versa_root) if args.versa_root else VersaRunner()
    try:
        log_path = latest_log(args.log_dir)
        replies = list(extract_replies(log_path, args.audio_out / args.log_dir.name, limit=args.limit))
        if not replies:
            print("aucune réponse audio dans ce log", file=sys.stderr)
            raise SystemExit(1)

        wavs = {reply.sample_id: reply.wav_path for reply in replies}
        scores: dict[str, dict[str, object]] = {}
        for key, values in runner.score(wavs, MOS_CONFIG).items():
            scores.setdefault(key, {}).update(values)
        if not args.skip_nisqa:
            for key, values in runner.score(wavs, nisqa_config(runner.root)).items():
                scores.setdefault(key, {}).update(values)
    except (Lfm2AudioError, FileNotFoundError) as error:
        print(f"❌ {error}", file=sys.stderr)
        raise SystemExit(1) from error

    report = build_report(replies, scores)
    print(f"{log_path.name} — {len(replies)} réponses")
    print(report.markdown())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.as_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
