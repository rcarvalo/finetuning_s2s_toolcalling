#!/usr/bin/env python3
"""Build the tool-generalization probe: unseen tools, spoken cases, on the Hub.

The question this probe answers, and nothing else: **does the fine-tune call
tools it has never been trained on, declared only in the system prompt?**

v4 and v5 saw exactly two tools, `web_search` and `db_query`, and the trained
system instructions name them explicitly with per-name routing rules. Every
agentic plan — MCP, LangGraph, ADK, a `delegate` tool — rests on the model
reading the declared tool list rather than reciting two memorised names. That
has never been measured, so it gets measured before anything is built on it.

Four declaration sets over the SAME utterances, so the only variable is which
tools are declared:

    tools_2            web_search + db_query          control, must reproduce ~0.83
    tools_2plus1       + delegate                     does it route to one unseen tool
    tools_2plus4       + four unseen tools            does routing degrade with count
    tools_unseen_only  the unseen tools alone         does it hallucinate the trained names

Cases are voiced with Kokoro on the CPU — user-side audio, no GPU, no cost —
following `infra/voice_scenarios.py`.

    python infra/probe_tools_build.py --push
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from lfm2_audio.tools.schemas import DB_QUERY, WEB_SEARCH

REPO = "Rcarvalo/tc-en-voice-agent-v1"
OUT = Path("benchmark/tool_probe")
AUDIO = OUT / "audio"
ARCHIVE_IN_REPO = "tool_probe/audio.tar.gz"
CASES_IN_REPO = "tool_probe/cases"
SAMPLE_RATE = 16_000

# Kokoro voices, the same eight the corpus uses for user turns. Rotated by case
# index so a result cannot be an artefact of one voice.
VOICES = ["af_heart", "af_bella", "af_nicole", "am_adam", "am_michael", "am_eric", "bf_emma", "bm_george"]


# --------------------------------------------------------------------------- #
# Probe tools — deliberately outside the semantics of web_search and db_query.
# A tool the model could reach by analogy ("search the news") would measure
# nothing: these act on the world or on a device, which neither trained tool
# can do.
# --------------------------------------------------------------------------- #

SET_TIMER: dict[str, Any] = {
    "name": "set_timer",
    "description": "Start a countdown timer for a given number of minutes.",
    "parameters": {
        "type": "object",
        "properties": {
            "duration_minutes": {"type": "integer", "description": "How long the timer runs, in minutes."},
            "label": {"type": "string", "description": "Optional name for the timer."},
        },
        "required": ["duration_minutes"],
    },
}

SEND_MESSAGE: dict[str, Any] = {
    "name": "send_message",
    "description": "Send a short text message to one of the user's contacts.",
    "parameters": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string", "description": "Name of the contact to write to."},
            "body": {"type": "string", "description": "Text of the message."},
        },
        "required": ["recipient", "body"],
    },
}

CONTROL_LIGHTS: dict[str, Any] = {
    "name": "control_lights",
    "description": "Turn the lights of a room on or off.",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string", "description": "Room whose lights to change."},
            "state": {"type": "string", "enum": ["on", "off"], "description": "Desired state."},
        },
        "required": ["room", "state"],
    },
}

CALENDAR_LOOKUP: dict[str, Any] = {
    "name": "calendar_lookup",
    "description": "Read the user's own calendar for a given day.",
    "parameters": {
        "type": "object",
        "properties": {"day": {"type": "string", "description": "Day to read, e.g. 'today' or 'friday'."}},
        "required": ["day"],
    },
}

DELEGATE: dict[str, Any] = {
    "name": "delegate",
    "description": (
        "Hand a complex, multi-step task to a stronger assistant that can plan, use many tools and "
        "come back with a summary. Use it when the request needs several steps or research beyond a "
        "single lookup."
    ),
    "parameters": {
        "type": "object",
        "properties": {"task": {"type": "string", "description": "The task to hand over, in one sentence."}},
        "required": ["task"],
    },
}

TRAINED = [WEB_SEARCH, DB_QUERY]
UNSEEN_FOUR = [SET_TIMER, SEND_MESSAGE, CONTROL_LIGHTS, CALENDAR_LOOKUP]

CONDITIONS: dict[str, list[dict[str, Any]]] = {
    "2": TRAINED,
    "2plus1": [*TRAINED, DELEGATE],
    "2plus4": [*TRAINED, *UNSEEN_FOUR],
    "unseen_only": [*UNSEEN_FOUR, DELEGATE],
}


# --------------------------------------------------------------------------- #
# Cases — (id, utterance, expected tool or None, expected arguments)
# Hand-written rather than LLM-generated: the probe must not inherit a
# generator's idea of which tool fits, which is the very thing under test.
# --------------------------------------------------------------------------- #

Case = tuple[str, str, str | None, dict[str, Any]]

TRAINED_CASES: list[Case] = [
    ("ws01", "What's the weather like in Berlin today?", "web_search", {"query": "weather in Berlin today"}),
    (
        "ws02",
        "Who won the last Formula One world championship?",
        "web_search",
        {"query": "last Formula One world championship winner"},
    ),
    ("ws03", "Look up the current price of silver.", "web_search", {"query": "current price of silver"}),
    ("ws04", "What time does the sun set in Oslo tomorrow?", "web_search", {"query": "sunset time Oslo tomorrow"}),
    ("ws05", "Give me the latest news about the Mars rover.", "web_search", {"query": "latest news Mars rover"}),
    ("ws06", "How tall is the Eiffel Tower?", "web_search", {"query": "height of the Eiffel Tower"}),
    (
        "db01",
        "How many orders did we ship last month?",
        "db_query",
        {"question": "How many orders did we ship last month?"},
    ),
    (
        "db02",
        "Which of our customers is based in Germany?",
        "db_query",
        {"question": "Which of our customers is based in Germany?"},
    ),
    (
        "db03",
        "What's the average value of an order in our system?",
        "db_query",
        {"question": "What is the average value of an order?"},
    ),
    (
        "db04",
        "List the products we currently have in the catalogue.",
        "db_query",
        {"question": "List the products in the catalogue."},
    ),
    (
        "db05",
        "Who is the account manager for Innovate Solutions?",
        "db_query",
        {"question": "Who is the account manager for Innovate Solutions?"},
    ),
    (
        "db06",
        "How many employees work in our sales team?",
        "db_query",
        {"question": "How many employees work in the sales team?"},
    ),
]

TIMER_CASES: list[Case] = [
    ("tm01", "Set a timer for ten minutes.", "set_timer", {"duration_minutes": 10}),
    (
        "tm02",
        "Start a twenty five minute timer for my focus session.",
        "set_timer",
        {"duration_minutes": 25, "label": "focus session"},
    ),
    ("tm03", "Give me a five minute countdown, would you?", "set_timer", {"duration_minutes": 5}),
    ("tm04", "I need a timer running for forty five minutes.", "set_timer", {"duration_minutes": 45}),
]

MESSAGE_CASES: list[Case] = [
    (
        "ms01",
        "Send a message to Sarah saying I'll be late.",
        "send_message",
        {"recipient": "Sarah", "body": "I'll be late."},
    ),
    (
        "ms02",
        "Text Marc that the meeting moved to three.",
        "send_message",
        {"recipient": "Marc", "body": "The meeting moved to three."},
    ),
    (
        "ms03",
        "Let Julie know the documents are ready, please.",
        "send_message",
        {"recipient": "Julie", "body": "The documents are ready."},
    ),
    (
        "ms04",
        "Write to Tom and tell him I'm on my way.",
        "send_message",
        {"recipient": "Tom", "body": "I'm on my way."},
    ),
]

LIGHTS_CASES: list[Case] = [
    ("lt01", "Turn off the lights in the kitchen.", "control_lights", {"room": "kitchen", "state": "off"}),
    ("lt02", "Switch the bedroom lights on.", "control_lights", {"room": "bedroom", "state": "on"}),
    (
        "lt03",
        "Could you kill the lights in the living room?",
        "control_lights",
        {"room": "living room", "state": "off"},
    ),
    ("lt04", "Lights on in the office, please.", "control_lights", {"room": "office", "state": "on"}),
]

CALENDAR_CASES: list[Case] = [
    ("cl01", "What's on my calendar today?", "calendar_lookup", {"day": "today"}),
    ("cl02", "Do I have anything scheduled on Friday?", "calendar_lookup", {"day": "friday"}),
    ("cl03", "Check my agenda for tomorrow.", "calendar_lookup", {"day": "tomorrow"}),
    ("cl04", "Am I free on Monday morning?", "calendar_lookup", {"day": "monday"}),
]

DELEGATE_CASES: list[Case] = [
    (
        "dg01",
        "Plan me a three day trip to Lisbon with flights, a hotel and two restaurants.",
        "delegate",
        {"task": "Plan a three day trip to Lisbon with flights, a hotel and two restaurants."},
    ),
    (
        "dg02",
        "Research our three biggest competitors and write me a summary of their pricing.",
        "delegate",
        {"task": "Research our three biggest competitors and summarise their pricing."},
    ),
    (
        "dg03",
        "Go through last quarter's sales, find the weakest region and suggest what to do about it.",
        "delegate",
        {"task": "Analyse last quarter's sales, find the weakest region and suggest actions."},
    ),
    (
        "dg04",
        "Compare the top three electric cars on range, price and charging speed, then recommend one.",
        "delegate",
        {"task": "Compare the top three electric cars on range, price and charging speed and recommend one."},
    ),
]

NEGATIVE_CASES: list[Case] = [
    ("ng01", "Hey, how are you doing today?", None, {}),
    ("ng02", "Thanks a lot, that's really helpful.", None, {}),
    ("ng03", "Good morning!", None, {}),
    ("ng04", "What can you actually do for me?", None, {}),
    ("ng05", "Never mind, forget it.", None, {}),
    ("ng06", "You're great, thank you.", None, {}),
]

UNSEEN_CASES = TIMER_CASES + MESSAGE_CASES + LIGHTS_CASES + CALENDAR_CASES

# Which cases each condition may legitimately be asked. A case whose tool is
# not declared has no defensible answer, so asking it would measure confusion
# rather than generalization.
CASE_SETS: dict[str, list[Case]] = {
    "2": TRAINED_CASES + NEGATIVE_CASES,
    "2plus1": TRAINED_CASES + DELEGATE_CASES + NEGATIVE_CASES,
    "2plus4": TRAINED_CASES + UNSEEN_CASES + NEGATIVE_CASES,
    "unseen_only": UNSEEN_CASES + DELEGATE_CASES + NEGATIVE_CASES,
}

ALL_CASES: list[Case] = TRAINED_CASES + UNSEEN_CASES + DELEGATE_CASES + NEGATIVE_CASES


# --------------------------------------------------------------------------- #


def to_dialogue(case: Case, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """A case as the dialogue schema every other stage already speaks."""
    case_id, utterance, tool, arguments = case
    user = {"role": "user", "text": utterance, "audio": f"{case_id}.wav"}
    if tool is None:
        assistant: dict[str, Any] = {"role": "assistant", "text": "Sure — how can I help?"}
    else:
        assistant = {"role": "assistant", "tool_calls": [{"name": tool, "arguments": arguments}]}
    return {
        "id": case_id,
        "tools": [definition["name"] for definition in tools],
        "meta": {"target": tool or "none", "seen_in_training": tool in {"web_search", "db_query", None}},
        "turns": [user, assistant],
    }


def validate(dialogues: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
    """Fail here rather than on a pod: schema, then argument validity.

    The registry is the same one inference uses, so a case it rejects would be
    scored wrong for a reason that has nothing to do with the model.
    """
    from lfm2_audio.ds.dialogue import parse_dialogue
    from lfm2_audio.tools.registry import ToolRegistry

    async def noop(**_: object) -> dict[str, Any]:
        return {}

    registry = ToolRegistry()
    for definition in tools:
        registry.register(definition, noop)

    for dialogue in dialogues:
        parse_dialogue(dialogue)
        for turn in dialogue["turns"]:
            for call in turn.get("tool_calls") or []:
                error = registry.validate(call["name"], call["arguments"])
                if error:
                    raise ValueError(f"{dialogue['id']}: {error}")


def render_audio() -> int:
    """One WAV per case, Kokoro on the CPU, voices rotated across cases."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from kokoro import KPipeline

    AUDIO.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")
    written = 0
    for index, (case_id, utterance, _, _) in enumerate(ALL_CASES):
        target = AUDIO / f"{case_id}.wav"
        if target.exists():
            continue
        voice = VOICES[index % len(VOICES)]
        chunks = [audio for _, _, audio in pipeline(utterance, voice=voice)]
        wav24 = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
        # Kokoro is 24 kHz native; the mel encoder is calibrated at 16 kHz and a
        # mismatch degrades what the model hears without raising anything.
        wav16 = torchaudio.functional.resample(torch.from_numpy(wav24), 24_000, SAMPLE_RATE).numpy()
        sf.write(str(target), wav16, SAMPLE_RATE, subtype="PCM_16")
        written += 1
    return written


def push() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    archive = Path("tool_probe_audio.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for wav in sorted(AUDIO.glob("*.wav")):
            tar.add(wav, arcname=wav.name)
    api.upload_file(path_or_fileobj=str(archive), path_in_repo=ARCHIVE_IN_REPO, repo_id=REPO, repo_type="dataset")

    for path in sorted(OUT.glob("*.json")) + sorted(OUT.glob("*.jsonl")):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{CASES_IN_REPO}/{path.name}",
            repo_id=REPO,
            repo_type="dataset",
        )
    print(f"pushed audio + {len(list(OUT.glob('*.json'))) + len(list(OUT.glob('*.jsonl')))} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="upload cases and audio to the Hub")
    parser.add_argument("--skip-audio", action="store_true", help="write the case files only")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    for name, tools in CONDITIONS.items():
        (OUT / f"tools_{name}.json").write_text(json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8")
        dialogues = [to_dialogue(case, tools) for case in CASE_SETS[name]]
        validate(dialogues, tools)
        (OUT / f"cases_{name}.jsonl").write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dialogues), encoding="utf-8"
        )
        positives = sum(1 for case in CASE_SETS[name] if case[2] is not None)
        print(f"{name:12} {len(tools)} tools · {len(dialogues)} cases ({positives} with a call)", flush=True)

    if not args.skip_audio:
        print(f"audio: {render_audio()} rendered, {len(list(AUDIO.glob('*.wav')))} present", flush=True)
        # Every case must be voiced: a silent case would be scored as a model
        # failure when it is a build failure.
        missing = [case_id for case_id, *_ in ALL_CASES if not (AUDIO / f"{case_id}.wav").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} cases not voiced, e.g. {missing[:3]}")

    if args.push:
        push()
    print("PROBE_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
