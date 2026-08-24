"""``lfm2-voice`` — hands-free voice assistant over the serverless endpoint.

ChatGPT-voice-style loop: the mic stays open, Silero VAD detects when you
stop speaking, the utterance goes to the endpoint and the reply audio plays
as it streams out (first sound ~1-2s after you pause, not after the full
answer). Requires the ``voice`` extra (``uv sync --extra voice``) and
``RUNPOD_API_KEY`` in the environment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastrtc import ReplyOnPause, Stream

from lfm2_audio.bench.voice_turn import VoiceTurnHandler
from lfm2_audio.remote.client import LiquidAudioClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Hands-free voice chat with a serverless endpoint")
    parser.add_argument("--endpoint", required=True, help="RunPod serverless endpoint id")
    parser.add_argument("--max-tokens", type=int, default=None, help="cap on generated tokens per turn")
    parser.add_argument("--share", action="store_true", help="expose a public Gradio link")
    parser.add_argument(
        "--save-turns",
        type=Path,
        default=Path("reports/voice_turns"),
        help="save each turn's user/reply WAVs here (listen to what the model heard); '' disables",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    handler = VoiceTurnHandler(
        LiquidAudioClient(args.endpoint),
        max_tokens=args.max_tokens,
        save_dir=args.save_turns if str(args.save_turns) else None,
    )
    stream = Stream(ReplyOnPause(handler.respond), modality="audio", mode="send-receive")
    stream.ui.launch(share=args.share)


if __name__ == "__main__":
    main()
