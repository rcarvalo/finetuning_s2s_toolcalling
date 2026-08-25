"""End-to-end conversation scenarios: audio in → tool call → REAL execution → spoken answer.

Runs on a GPU pod. Exercises the full orchestrator loop (`ReceptionAgent`) with
the v2 adapter, DuckDuckGo live for `web_search` and the demo DB for
`db_query`. Multi-turn: each scenario keeps ONE chat state across its turns,
so follow-ups measure conversational coherence, not just single-shot routing.

Writes a JSONL transcript (one record per user turn) with the tool activity,
timings and the spoken answer, plus a WAV of every spoken reply.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from lfm2_audio.orchestrator.agent import AgentConfig, ReceptionAgent
from lfm2_audio.orchestrator.events import (
    AudioChunk,
    TextDelta,
    ToolCallBegin,
    ToolCallResult,
    TurnComplete,
)
from lfm2_audio.serving.model import LFM2Audio
from lfm2_audio.tools.fake_db import FakeDbBackend
from lfm2_audio.tools.toolcalling_en import build_toolcalling_en_registry
from lfm2_audio.tools.web_search.duckduckgo import DuckDuckGoBackend

BASE = "LiquidAI/LFM2.5-Audio-1.5B"
ADAPTER = "Rcarvalo/lfm25-tc-en-v2-adapter"
SCENARIOS = Path("data/scenarios.json")
AUDIO_ROOT = Path("data/audio_scenarios")
OUT_DIR = Path("reports/scenarios")

SYSTEM = (
    "You are a voice assistant that can call tools. Routing rules: use web_search "
    "for anything PUBLIC or CURRENT — weather, news, prices, sports, facts about "
    "the world. Use db_query ONLY for our INTERNAL company data (customers, orders, "
    "products, employees, meetings). Call at most one tool, and only when needed; "
    "for greetings or chit-chat, reply directly without any tool."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=ADAPTER, help="Hub repo id or local adapter directory")
    parser.add_argument("--out-dir", default=str(OUT_DIR), type=Path)
    parser.add_argument(
        "--push-to",
        default=None,
        help="dataset repo id: upload the transcript when done. A reclaimed VM "
        "otherwise takes the only copy of the run with it.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the tested resolve/merge path, then hand the raw pieces to the agent.
    backend = LFM2Audio.from_pretrained(BASE, backend="liquid", adapter=args.adapter)
    registry = build_toolcalling_en_registry(
        web_backend=DuckDuckGoBackend(), db_backend=FakeDbBackend(), timeout_s=15.0
    )
    agent = ReceptionAgent(
        backend._model,  # the agent wants the raw model; the backend owns the merge
        backend._processor,
        registry,
        config=AgentConfig(max_new_tokens=512, system_instructions=SYSTEM),
    )

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    transcript = (out_dir / "transcript.jsonl").open("w", encoding="utf-8")

    for scenario in scenarios:
        chat = agent.new_session()
        print(f"\n=== SCENARIO {scenario['id']} ===", flush=True)
        for index, user_text in enumerate(scenario["turns"]):
            wav_path = AUDIO_ROOT / f"{scenario['id']}__t{index}_u0.wav"
            samples, rate = sf.read(str(wav_path), dtype="float32")
            record = {
                "scenario": scenario["id"],
                "turn": index,
                "user": user_text,
                "tools": [],
                "answer": "",
                "rounds": 0,
            }
            audio_out: list[torch.Tensor] = []
            start = time.monotonic()
            first_event_s = None
            for event in agent.respond(chat, torch.as_tensor(samples).reshape(1, -1), rate):
                if first_event_s is None:
                    first_event_s = time.monotonic() - start
                if isinstance(event, ToolCallBegin):
                    record["tools"].append({"name": event.name, "arguments": event.arguments})
                elif isinstance(event, ToolCallResult):
                    payload = json.dumps(event.payload, default=str)
                    record["tools"][-1] |= {
                        "ok": event.ok,
                        "elapsed_ms": round(event.elapsed_ms, 1),
                        "result_preview": payload[:300],
                    }
                elif isinstance(event, AudioChunk):
                    audio_out.append(torch.as_tensor(event.samples).reshape(-1))
                elif isinstance(event, TextDelta):
                    pass  # aggregated by TurnComplete
                elif isinstance(event, TurnComplete):
                    record["answer"] = event.text
                    record["rounds"] = event.tool_rounds
            record["total_s"] = round(time.monotonic() - start, 2)
            record["first_event_s"] = round(first_event_s or 0.0, 2)
            if audio_out:
                spoken = torch.cat(audio_out).float().cpu().numpy()
                out_wav = out_dir / f"{scenario['id']}__t{index}.wav"
                sf.write(str(out_wav), spoken, 24_000, subtype="PCM_16")
                record["spoken_s"] = round(len(spoken) / 24_000, 1)
            transcript.write(json.dumps(record, ensure_ascii=False) + "\n")
            transcript.flush()
            tools = ", ".join(t["name"] for t in record["tools"]) or "none"
            print(
                f"  [t{index}] tools={tools} rounds={record['rounds']} "
                f"total={record['total_s']}s answer={record['answer'][:90]!r}",
                flush=True,
            )

    transcript.close()

    if args.push_to:
        import os

        from huggingface_hub import HfApi

        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=str(out_dir / "transcript.jsonl"),
            path_in_repo=f"reports/scenarios_{args.adapter.rsplit('/', 1)[-1]}.jsonl",
            repo_id=args.push_to,
            repo_type="dataset",
        )
        print(f"transcript pushed to {args.push_to}", flush=True)
    print("SCENARIOS_DONE", flush=True)


if __name__ == "__main__":
    main()
