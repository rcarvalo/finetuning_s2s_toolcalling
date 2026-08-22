# EN baseline — LFM2.5-Audio-1.5B (vanilla)

Reference numbers every fine-tune is compared against (weekend steps 1-2).
Run on a Colab **L4 22 GB**, backend `liquid`, checkpoint
`LiquidAI/LFM2.5-Audio-1.5B`, 2026-08-22.

**Comparability**: a later campaign is only comparable if it uses the same
question set, the same `--max-tokens` and the same ASR
(`openai/whisper-large-v3-turbo`). `lfm2-eval-compare` warns when the report
contexts disagree. Reports are committed under `reports/`.

## Step 2 — tool calling (12 cases, 8 positive / 4 negative)

`benchmark/toolcalling_en/cases.sample.jsonl`, scorer `tool_call`, max 200 tokens.

| Figure | Baseline | Reading |
|---|---:|---|
| Aggregate score | **0.333** | mean over the 12 cases (median 0.0) |
| Cases expecting a call | 8/12 | — |
| Tool calls actually emitted | **0/12** | the vanilla model never emits `<\|tool_call_start\|>` |
| name / call accuracy | **0%** | nothing to match |
| Relevance on negatives | **100%** (4/4) | correctly abstains when no tool is needed |
| Parse rate | 100% | no malformed output |

The 0.333 comes entirely from the negatives. The model answers positives in
plain language instead of calling a tool — either from parametric memory
("Manchester City won…") or by refusing ("I don't have live weather data").

**What the fine-tune must achieve**: teach emission on the 8 positives *without
losing the 4 abstentions*. Relevance is the metric to watch for regression —
it is currently perfect and the easiest thing to break.

## Step 1 — audio quality (24 open EN questions)

`benchmark/baseline_en/questions.jsonl`, scorers `wer,dnsmos,nisqa`, max 400 tokens.

| Metric | Baseline | Direction | Notes |
|---|---:|---|---|
| WER | **0.086** | lower better | median 0.057, measured 24/24 |
| DNSMOS OVRL | **3.329** | higher better | median 3.358; SIG 3.34-3.72, BAK 4.03-4.21 |
| NISQA | unavailable | higher better | weights not provided (`NISQA_MODEL_PATH`) |

WER here is *self-consistency*: the generated speech is re-transcribed and
compared to the text the model itself produced. ~9% means the voice is
intelligible to an ASR — it does not measure whether the answer is correct.

DNSMOS OVRL 3.33 with BAK above 4 is a healthy profile for a neural-codec
voice: no background degradation, the ceiling comes from signal quality (SIG
~3.5). This is the number a fine-tune must not damage — audio heads are the
easiest thing to break while teaching tool calling.

### Calibration incident (fixed — and it mattered)

The first run reported OVRL ~1.61 with BAK pinned at ~1.90 across all 24
utterances. That near-zero variance was the tell: the calibration polynomials
were cubics, not the official non-personalized quadratics from
`microsoft/DNS-Challenge` (`dnsmos_local.py`, `get_polyfit_val`). Fixed in
`e2f567e`, coefficients fetched from source, 6 regression tests including
"the scale must reach the top".

Corrected figures: **OVRL 1.61 → 3.33, BAK 1.90 → ~4.1**. A first re-run still
printed 1.61 because the VM had not picked up the new commit; the campaign job
now asserts the live calibration before scoring and aborts rather than publish
stale numbers. Two independent checks confirm the current values: the ONNX
model scores white noise ~1.0 on all subscores and silence BAK 3.29, and
zero-padding was ruled out (tiling the audio instead of padding changes OVRL by
0.01).

**Any DNSMOS figure produced before `e2f567e` is invalid.**
