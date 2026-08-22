# Dataset selection for EN tool calling (weekend step 3)

Full listing: `docs/dataset_inventory.md` (50 repos, 5 private, produced by
`lfm2-dataset-inventory`). This document is the decision: what we train on, what
we ignore, and what is missing.

## What the corpus actually contains

Of the 50 repos, the overwhelming majority is **French TTS/ASR material**
(`french-dialogue-tts-1000h`, `emilia-yodas-fr-filtered`, `kanitts2-fr-*`,
`audioFR*`, the `hugodecrypte_*` chunks, ~700 GB total). It is excellent data —
for the French stage of the project. **None of it serves EN tool calling**, so
none of it enters this pipeline.

Three repos are on-topic:

| Repo | Content | Verdict |
|---|---|---|
| `Rcarvalo/tc-en-audio-toolcalling` (private, 477 MB) | **2930 train / 12 test**, single-turn spoken EN → tool call, 16 kHz, held-out test voices | **core training set** |
| `Rcarvalo/tc-en-s2s-src` (private, 3.1 MB) | 3000 **conversational** dialogues, text only: user → tool_call → tool result → spoken reply (2197 with a tool, 803 negatives) | **to synthesize** |
| `Rcarvalo/tc-en-s2s-audio` (public) | **empty** — only `.gitattributes` | the TTS run was never pushed |

`lfm2-bilingual-pilot-125h` and `lfm2-audio-fr-data` are FR-adaptation assets;
they matter for the bilingual stage, not for this one.

## The two gaps, in order of importance

**1. The evaluation set is 12 cases.** The step-2 baseline (0.333, zero calls
emitted) was measured on twelve examples. That is enough to prove the model
never calls a tool, but far too small to tell a 5-point improvement from noise
at step 7. A held-out set of ~200 cases is the prerequisite for every decision
that follows.

**2. There is no conversational audio.** `tc-en-s2s-src` holds exactly the
4-turn structure a voice agent needs (tool result reinjected, spoken answer),
but it was never synthesized — hence the empty `tc-en-s2s-audio`. Phase A
(single-turn, emit the call) can train today; Phase B (speak the answer after
the tool round-trip) cannot.

Both gaps are addressed by tooling that already exists in this repo:
`lfm2-synthesize-audio` supports `--assistant-voice` precisely for the Phase B
spoken replies, and the TTS/push/pack chain is in place. What was missing is a
curation step that merges sources, removes duplicates and guarantees no
train/test leakage.

## Plan

| Step | Action | Target |
|---|---|---|
| 4 | Curate `Rcarvalo/tc-en-voice-agent` from the two on-topic sources + optional public function-calling corpora, with dedup and a contamination guard | one clean repo |
| 4b | Grow the held-out test split to ~200 cases, voices disjoint from train | reliable step-7 comparison |
| 5 | `lfm2-dataset-curate` → TTS → `lfm2-build-dataset` push | reproducible pipeline |
| 6 | LoRA fine-tune on the curated set, scoring callbacks on the same metrics | Phase A then Phase B |
| 7 | Re-run both baseline campaigns, `lfm2-eval-compare` | keep or discard |

**Enrichment from the internet** is kept as an option, not a default: the two
private sources already provide ~5900 dialogues. If more variety is needed
(more tools, harder routing), public function-calling corpora exist in text form
(BFCL-style suites, xlam/Glaive-style sets) and would go through the same TTS
step. That decision belongs after the first fine-tune shows where the model
still fails — buying more data before knowing the failure mode is guesswork.
