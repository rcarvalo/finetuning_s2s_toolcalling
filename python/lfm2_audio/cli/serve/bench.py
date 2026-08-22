"""Listening bench — talk to a model version, then rate it against a test set.

    lfm2-bench-app --endpoint $RUNPOD_ENDPOINT_ID            # no local GPU needed
    lfm2-bench-app --checkpoint LiquidAI/LFM2.5-Audio-1.5B --backend liquid
    lfm2-bench-app --checkpoint exports/tc_en --backend vllm --share

Two tabs over one loaded model:

- **Talk** — type or speak, hear the answer. For forming an impression.
- **Rate** — walk the test set, score each answer 1-5 on three axes, flag the
  clips that derail. Verdicts append to `reports/human_ratings.jsonl`.

Answers come either from a checkpoint loaded here (`--checkpoint`, needs a GPU)
or from a serverless endpoint (`--endpoint`, needs nothing but network). The
local path also lets you pick the backend: `liquid` is the reference PyTorch
one, `vllm` the low-latency one. Judging the same checkpoint through both is a
legitimate use — they are different code paths and can sound different.

A serverless worker is stateless, so the Talk tab is single-turn against an
endpoint. That is fine for judging a voice and useless for judging a
conversation; the UI says which one you are in rather than pretending.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import gradio as gr
import numpy as np

from lfm2_audio.bench.local_source import LocalModelSource
from lfm2_audio.bench.rating import SCALE_MAX, SCALE_MIN
from lfm2_audio.bench.remote_source import RemoteEndpointSource
from lfm2_audio.bench.session import BenchSession
from lfm2_audio.bench.source import AnswerSource
from lfm2_audio.bench.store import DEFAULT_PATH, RatingStore
from lfm2_audio.ds.audio import Waveform
from lfm2_audio.evaluation.question_set import QuestionSet
from lfm2_audio.remote.client import LiquidAudioClient
from lfm2_audio.serving.model import LFM2Audio

logger = logging.getLogger(__name__)

DEFAULT_QUESTIONS = "benchmark/baseline_en/questions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="local path, HF repo, or adapter (needs a GPU)")
    source.add_argument("--endpoint", help="RunPod serverless endpoint id (no GPU needed)")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--backend", choices=["auto", "vllm", "liquid"], default="auto")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument(
        "--version",
        default=None,
        help="label recorded with every verdict (default: <checkpoint>@<backend>)",
    )
    parser.add_argument("--ratings", default=str(DEFAULT_PATH))
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser


def to_gradio(waveform: Waveform | None) -> tuple[int, np.ndarray] | None:
    """Gradio wants (sample_rate, int16) — the value object keeps its own rate."""
    if waveform is None or waveform.is_empty:
        return None
    pcm = (np.clip(waveform.samples, -1.0, 1.0) * 32_767).astype(np.int16)
    return waveform.sample_rate, pcm


def build_app(session: BenchSession, *, max_tokens: int) -> gr.Blocks:
    """Two tabs sharing one loaded model."""
    with gr.Blocks(title=f"Listening bench — {session.version}") as demo:
        gr.Markdown(f"### {session.version}\nRatings append to `{session.store.path}`.")

        if not session.keeps_history:
            gr.Markdown(
                "⚠️ Stateless source: every turn starts fresh, so the Talk tab is "
                "single-turn. Fine for judging the voice, not a conversation."
            )

        with gr.Tab("Talk"):
            _build_talk_tab(session, max_tokens)
        with gr.Tab("Rate"):
            _build_rate_tab(session, max_tokens)

    return demo


def _build_talk_tab(session: BenchSession, max_tokens: int) -> None:
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Say something", lines=2)
            audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="…or speak")
            send = gr.Button("Send", variant="primary")
            reset = gr.Button("New conversation")
        with gr.Column():
            audio_out = gr.Audio(label="Answer", autoplay=True)
            text_out = gr.Textbox(label="Transcript", lines=4)
            stats = gr.Textbox(label="Timing", lines=1)

    def talk(text: str, mic: tuple[int, np.ndarray] | None) -> tuple[Any, str, str]:
        # History is kept between turns here, unlike the rating tab: this tab is
        # for conversation, where context is the point.
        wave = Waveform.from_pcm16(np.asarray(mic[1]), int(mic[0])) if mic is not None else None
        if wave is None and not (text or "").strip():
            return None, "", "nothing to send"

        reply = session.talk(text=None if wave is not None else text, audio=wave, max_tokens=max_tokens)
        ttfa = reply.metrics.ttfa_s
        timing = f"total {reply.metrics.total_s:.1f}s"
        if ttfa is not None:
            timing = f"TTFA {ttfa * 1000:.0f}ms · " + timing
        if reply.real_time_factor is not None:
            timing += f" · RTF {reply.real_time_factor:.2f}"
        return to_gradio(reply.audio), reply.text, timing

    def clear() -> tuple[Any, str, str]:
        session.reset_conversation()
        return None, "", "history cleared"

    send.click(talk, [text_in, audio_in], [audio_out, text_out, stats])
    reset.click(clear, outputs=[audio_out, text_out, stats])


def _build_rate_tab(session: BenchSession, max_tokens: int) -> None:
    cases = session.case_ids()

    with gr.Row():
        case = gr.Dropdown(cases, value=cases[0] if cases else None, label="Case")
        progress = gr.Textbox(session.progress(), label="Progress", interactive=False)
        generate = gr.Button("Generate", variant="primary")

    prompt = gr.Textbox(label="Question asked", lines=2, interactive=False)
    heard = gr.Audio(label="What the model said", autoplay=False)
    said = gr.Textbox(label="Text it produced", lines=3, interactive=False)

    gr.Markdown(
        "Rate what you *heard*. `derailed` is not a low score — tick it when the "
        "clip loops or babbles instead of speaking the text, so those cases can "
        "be counted separately."
    )
    with gr.Row():
        intelligibility = gr.Slider(SCALE_MIN, SCALE_MAX, value=3, step=1, label="Intelligibility")
        naturalness = gr.Slider(SCALE_MIN, SCALE_MAX, value=3, step=1, label="Naturalness")
        overall = gr.Slider(SCALE_MIN, SCALE_MAX, value=3, step=1, label="Overall")
    with gr.Row():
        derailed = gr.Checkbox(label="Derailed (loop / babble)")
        notes = gr.Textbox(label="Notes", lines=1)
    save = gr.Button("Save rating", variant="primary")
    saved = gr.Textbox(label="Saved", lines=1, interactive=False)

    def run_case(case_id: str) -> tuple[str, Any, str]:
        question = session.question(case_id)
        reply, path = session.generate(case_id, max_tokens=max_tokens)
        logger.info("generated %s -> %s", case_id, path)
        return question.text, to_gradio(reply.audio), reply.text

    def save_rating(case_id: str, intel: float, nat: float, over: float, bad: bool, note: str) -> tuple[str, str]:
        session.record(
            case_id,
            intelligibility=int(intel),
            naturalness=int(nat),
            overall=int(over),
            derailed=bool(bad),
            notes=note or "",
        )
        return f"{case_id} saved", session.progress()

    generate.click(run_case, [case], [prompt, heard, said])
    save.click(
        save_rating,
        [case, intelligibility, naturalness, overall, derailed, notes],
        [saved, progress],
    )


def build_source(args: argparse.Namespace) -> AnswerSource:
    """Local checkpoint or remote endpoint, per the mutually exclusive flags."""
    if args.endpoint:
        logger.info("using serverless endpoint %s", args.endpoint)
        return RemoteEndpointSource(
            LiquidAudioClient(args.endpoint),
            label=args.version or f"endpoint:{args.endpoint}",
        )

    logger.info("loading %s on backend %s", args.checkpoint, args.backend)
    model = LFM2Audio.from_pretrained(args.checkpoint, backend=args.backend, adapter=args.adapter)
    return LocalModelSource(model, label=args.version or f"{args.checkpoint}@{model.backend_name}")


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    questions = QuestionSet.from_jsonl(args.questions, audio_root=args.audio_root)
    session = BenchSession(build_source(args), questions, version=args.version, store=RatingStore(args.ratings))
    build_app(session, max_tokens=args.max_tokens).launch(server_port=args.port, share=args.share, quiet=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
