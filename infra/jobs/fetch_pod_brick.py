"""Copy a brick's unpushed clips off a pod through its HTTP artifact server.

The pod entrypoint serves ``$OUT`` on port 8000 before the job even starts,
reachable as ``https://<pod>-8000.proxy.runpod.net``. When the pod's own
pushes fail (05/09: the Hub's 10 000-files-per-folder cap), this is the
only way out that needs nothing on the pod. Fetches the manifest and the
logs, then every clip the Hub lacks; re-runnable, skips what is local. ::

    python infra/jobs/fetch_pod_brick.py --url https://tpbcptosv3wmwk-8000.proxy.runpod.net/A_assistant_speech \\
        --out data/corpus/A_assistant_speech_pod --brick A_assistant_speech
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from lfm2_audio.data_prep.corpus_layout import CorpusEntry, read_manifest  # noqa: E402

USER_AGENT = "lfm2-audio-infra/1.0"


def missing_downloads(entries: list[CorpusEntry], on_hub: set[str], out: Path, brick: str) -> list[CorpusEntry]:
    """Entries whose audio is neither on the Hub (any path) nor already local."""
    hub_names = {Path(f).name for f in on_hub if f.startswith(f"{brick}/audio/")}
    return [e for e in entries if Path(e.audio).name not in hub_names and not (out / e.audio).exists()]


def fetch(url: str, target: Path, retries: int = 3) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(retries):
        try:
            # The proxy sits behind Cloudflare, which rejects urllib's default agent (403, code 1010).
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
            return True
        except OSError:
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="URL du dossier de la brique sur le pod")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--brick", required=True)
    parser.add_argument("--repo", default="Rcarvalo/lfm25-fr-corpus-v1")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    base = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.jsonl", "dropped.jsonl", "summary.json"):
        fetch(f"{base}/{name}", args.out / name)
    entries = list(read_manifest(args.out / "manifest.jsonl"))
    on_hub = set(HfApi().list_repo_files(args.repo, repo_type="dataset"))
    todo = missing_downloads(entries, on_hub, args.out, args.brick)
    print(f"{len(entries)} entrées sur le pod, {len(todo)} clips à rapatrier", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, ok in enumerate(pool.map(lambda e: fetch(f"{base}/{e.audio}", args.out / e.audio), todo), start=1):
            done += ok
            failed += not ok
            if index % 200 == 0 or index == len(todo):
                print(f"  {index}/{len(todo)} — rapatriés {done}, échecs {failed}", flush=True)
    print(f"===RESULT fetch_pod_brick=== fetched={done} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
