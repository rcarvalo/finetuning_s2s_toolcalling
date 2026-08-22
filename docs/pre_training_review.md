# Before launching the fine-tune: what is missing, and does LoRA suffice?

Written after the EN baselines (`docs/baseline_en.md`) and the dataset curation
(`docs/dataset_selection.md`), before any GPU training run.

## 1. Does LoRA suffice, or do we need a full fine-tune?

The honest answer is that the question is decidable in about an hour of GPU, and
guessing is more expensive than measuring. But the prior is strong, for three
reasons.

**The capability exists in the backbone.** `<|tool_call_start|>` is part of the
model's own chat format; LFM2 emits tool calls in text today. What is missing is
not the ability to produce the token sequence, it is the *routing* from an
audio-conditioned context to that sequence. Teaching a routing decision is what
low-rank adaptation is for; teaching a new representation is not.

**There is a documented precedent on this exact model and task.** The Liquid
cookbook `voice-assistant` (OHF-Voice) reaches 99% name accuracy and 90% argument
accuracy on audio→function-call with LoRA on LFM2.5-Audio. Our acceptance
thresholds (README) are name > 90%, relevance > 85%, arg > 75%, parse > 98% —
below what LoRA already achieved there.

**The baseline failure is binary, not gradual.** The model emits zero calls while
abstaining perfectly. That is the signature of an unexpressed behaviour, not of
an overloaded capacity. A model that emitted malformed or wrong-tool calls would
argue for more capacity; one that emits nothing argues for a decision boundary.

### The escalation ladder, with a trigger for each rung

Run rung 1, measure on `test_utterances` (200 cases), and climb only on evidence:

| Rung | Configuration | Climb when |
|---|---|---|
| 1 | LoRA r=16, backbone only | start here (~1 h on L4) |
| 2 | LoRA r=64, + attention *and* MLP projections | name accuracy < 90% but calls are being emitted |
| 3 | Unfreeze the last 4–6 backbone blocks alongside LoRA | arguments still wrong while names are right |
| 4 | Full fine-tune of the text backbone | rung 3 plateaus below threshold |

Rung 4 is affordable and should not be feared: 1.5 B parameters in bf16 with an
8-bit optimizer and gradient checkpointing fits a 22 GB L4. But it costs ~10x the
time of rung 1 and risks the audio heads, so it must be earned by a measurement,
not chosen upfront.

**What would make me skip to rung 3 immediately**: if training data were scarce.
It is not — 2729 examples today, ~5600 after enrichment.

## 2. What is missing before we can train at all

Blocking, in order:

1. **No baseline on the 200-case test set.** The published baseline (0.333, zero
   calls) was measured on the 12-case split. Comparing a fine-tune on 200 cases
   against a baseline on 12 is not a comparison. **This must run first.**
2. **The packed datasets do not exist.** The recipe points at
   `datasets/tc_en_v1_train` / `_val`; nothing has been packed from the curated
   repo yet.
3. **No validation split.** `test_utterances` is the *final* judge and must not
   be watched during training. A separate val split has to be carved from train.
4. **In-training scoring is off.** `EvaluationScheduleConfig.enabled` defaults to
   `false` and `question_set` is empty — the recipe as written would have trained
   blind for 1023 steps. Fixed below.

Non-blocking but worth knowing:

5. **Three voices only** in training (`casual_male`, `casual_female`,
   `cheerful_female`). The model may key on timbre rather than content;
   `test_voices` exists precisely to detect that, and it is 12 cases wide.
6. **Two tools only.** With a binary choice, a model can score well by learning
   "public → web_search, internal → db_query" without learning to abstain from
   calling at all. Relevance on negatives is the metric that catches this.
7. **No audio augmentation.** Training audio is clean TTS; the Reachy Mini will
   hear a room. Noise/reverb/gain augmentation at packing time is cheap insurance.

## 3. Enrichment: what to add, and why

| Source | Adds | Cost | Verdict |
|---|---|---|---|
| `tc-en-s2s-src` (3000 conversational dialogues, text) | ~2859 new Phase A examples *and* unlocks Phase B | TTS on L4 | **do it** |
| More TTS voices on a slice of the corpus | robustness to timbre | TTS on L4 | do it with the above |
| Audio augmentation at packing | robustness to rooms | free (CPU) | do it |
| Public function-calling corpora (BFCL/xlam-style) → TTS | more tools, harder routing | high | only if rung 1–2 plateaus |

The conversational set is the obvious first move: the text already exists, the
TTS tooling already supports assistant voices (`--assistant-voice`), and each
4-turn dialogue yields a Phase A example (user → tool call) *plus* a Phase B one
(tool result → spoken answer). It roughly doubles the corpus for one TTS run.

Buying more data from the internet before rung 1 has run would be guesswork: we
do not yet know whether the model fails on tool choice, on arguments, or on
abstention, and each failure mode calls for different data.

## 4. Order of operations

1. Baseline the vanilla model on `test_utterances` (200) and `test_voices` (12).
2. Synthesize the conversational corpus; push `tc-en-voice-agent-v2`.
3. Carve val from train; pack train/val.
4. Rung 1 LoRA, in-training scoring on a 32-case slice of *val*.
5. Re-run both campaigns, `lfm2-eval-compare`, then decide on the ladder above.
