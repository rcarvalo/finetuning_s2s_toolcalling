# End-to-end conversation scenarios — what works, what doesn't (2026-08-24)

Full loop exercised on a GPU pod: **spoken question → v2 routes → REAL tool
executes** (DuckDuckGo live, demo DB) **→ result reinjected → spoken answer**.
8 scenarios, 13 turns, multi-turn state kept per scenario. Harness:
`infra/pod_scenarios.py`; artifacts: `reports/scenarios/` (transcript, WAVs,
Whisper transcriptions of every spoken reply).

## What works — the plumbing and the decision

| Capability | Verdict |
|---|---|
| Routing (db vs web vs none) | **12/13 correct** — internal→db_query, public→web_search, chit-chat/thanks→no tool |
| Argument quality | strong: disfluent "Um, so, like… gold" → `query="current price of gold"`; inference case → `question="Which product is not selling well?"` |
| Real tool execution | DuckDuckGo live ~0.9–1.7 s, demo DB <1 ms — **round-trip under the 1.5 s target** |
| Malformed-call defense | registry rejected 2 bad calls (`body=`, `questionunciation=`) instead of crashing |
| Chit-chat turns | natural spoken replies ("I'm doing well, thank you for asking!") |
| Turn latency | 2–3 s with a tool, 0.2 s without |

The one routing miss is the known encyclopedic boundary (capital of Italy →
web_search), consistent with the eval's 3 false calls.

## What breaks the conversation — the spoken answer after a tool result

Every post-tool reply generates audio (1–12 s)… of **babble**. Whisper says:

> s1 (weather Paris): "It's not vision, it's us."
> s1 follow-up: "to you, to you, to you, to you…"
> s5 (gold price): "ooooooooooo…"
> s2 (order count): "later."

Zero grounding in the tool result, every time. Two scenarios also re-emitted a
*malformed second tool call* right after the result — the model literally does
not know what a `tool` turn is for.

**This is not a bug; it is the missing training phase.** v2 (like v1) was
trained single-turn: user audio → tool call, END. The corpus never contains a
tool response followed by a spoken answer. The model aces the decision it was
taught and babbles at the turn it never saw.

## Consequence: v3 = Phase B, and the data already exists

`Rcarvalo/tc-en-s2s-src` holds 2 197 four-turn dialogues with exactly the
missing structure (user → call → tool result → **spoken answer**), text-only.
The path: voice the user turns AND the answer turns (`lfm2-synthesize-audio
--assistant-voice`), pack with `assistant-audio-mode`, train v3 on
Phase A + Phase B mixed, re-run these scenarios as the acceptance test.

Until v3, a production stopgap exists: run the answer turn in TEXT mode and
speak it via TTS — the orchestrator's hybrid switch makes that a one-line
change — at the cost of the S2S latency advantage.
