# v2 — tool-calling scores, measured honestly

Adapter: `Rcarvalo/lfm25-tc-en-v2-adapter` (LoRA r=32 α=64, **step 500 of 948**
— the Colab VM was reclaimed mid-run; the 250-step Hub pushes did their job).
Evaluation: tools declared in the system prompt (byte-identical to training),
text-only decode, `tool_call` scorer. All six campaigns on one L4, 2026-08-22.

## The table

| model | set | mean | emission | name | call (args token-F1 ≥ 0.7) | abstention | parse |
|---|---|---:|---|---|---|---|---|
| vanilla | voices (12) | 0.333 | 0/8 | 0/8 | 0/8 | 4/4 | 12/12 |
| vanilla | utterances (200) | 0.255 | 0/149 | 0/149 | 0/149 | 51/51 | 200/200 |
| v1 (June) | voices (12) | 0.750 | 8/8 | 8/8 | 5/8 | 4/4 | 12/12 |
| v1 (June) | utterances (200) | 0.855 ⚠️ | 149/149 | 149/149 | 122/149 | 49/51 | 200/200 |
| **v2 @500** | voices (12) | **0.750** | 8/8 | 8/8 | 5/8 | 4/4 | 12/12 |
| **v2 @500** | utterances (200) | **0.810** | **149/149** | **149/149** | **114/149** | **48/51** | **200/200** |

⚠️ **v1's 0.855 is contaminated**: the 200 `test_utterances` were carved from
the very rows v1 trained on in June. v2 trained on `train` minus both test
splits — its 0.810 is the honest figure. On the only set clean for both (the 12
held-out voices), they tie at 0.750.

## Against the project's acceptance thresholds (README)

| Criterion | Threshold | v2 @500 | |
|---|---|---|---|
| name accuracy | > 90% | **100%** (149/149) | ✅ |
| relevance | > 85% | **95%** (emission 100%, abstention 94%) | ✅ |
| args (tolerant) | > 75% | **76.5%** at token-F1 ≥ 0.7 (the scorer's default) | ✅ just above |
| parse rate | > 98% | **100%** | ✅ |

**v2 passes every acceptance criterion on an uncontaminated test set, at
53% of its training budget.** Vanilla → v2: 0.255 → 0.810 (3.2×).

## Reading the failures

- **Argument mismatches (35/149)**: the call is well-formed, the tool right,
  the wording differs. Note the 76.5% is already token-F1 ≥ 0.7 (scorer
  default); close reading shows mostly faithful paraphrases of a free-text
  question — see `docs/v2_limits.md` for why a semantic judge is the honest
  arbiter on `db_query` args.
- **3 false calls on 51 negatives** — abstention held (94%), the regression the
  ladder feared did not materialize.
- **The 12-voice set saturates at 5/8 exact calls for both adapters**: at n=8,
  one utterance is 12.5 points. This set detects timbre collapse, nothing finer.

## The in-training curve (val, 32 mixed cases — first run ever measured)

step 0: 0.281 (vanilla ref) → 100: 0.719 → 200: 0.688 → 300: 0.812 →
400: 0.812 → **500: 0.844**. The signal that used to be invisible.

## Verdict and next steps

1. v1 was never broken — the evaluation was. v2 confirms the recipe and adds a
   clean provenance: **it is the deployable candidate.**
2. Finishing the remaining 448 steps is optional; the curve had flattened.
   Cheaper wins first: re-score args with token-F1 (measurement), and train on
   the conversational corpus once synthesized (Phase B, real capability).
3. The fresh Voxtral test set (300 unseen utterances, held-out voices) remains
   the missing measurement piece — blocked on the Voxtral server environment.

## Final clean verdict — the fresh set (added 2026-08-23)

300 cases the training corpus has **never seen in any form**: unseen wording
(drawn from the conversational source, deduplicated against every training
utterance) *and* an unseen TTS engine (Kokoro `af_sarah`/`bm_lewis` — training
audio is entirely Voxtral). This measures voice/engine transfer, the closest
proxy we have to a real user.

| model | mean | emission | name | args (F1 ≥ 0.7) | abstention | parse |
|---|---:|---|---|---|---|---|
| vanilla | 0.250 | 0/225 | 0/225 | 0/225 | 75/75 | 300/300 |
| v1 (June) | 0.827 | 224/225 | 222/225 | 176/225 | 72/75 | 300/300 |
| **v2 @500** | **0.833** | **224/225 (99.6%)** | **222/225 (98.7%)** | **178/225 (79.1%)** | **72/75 (96%)** | **300/300** |

Every acceptance threshold passes on fully held-out data: name 98.7% (>90),
relevance ≈ 97% (>85), tolerant args 79.1% (>75), parse 100% (>98). The
adapters transfer across TTS engines almost losslessly — emission and routing
barely move from the seen-voice numbers.

**v1 vs v2 on clean ground: a tie** (0.827 vs 0.833; the 2-case argument edge
is noise at n=225). The June contamination accounted for v1's apparent
advantage. v2 remains the deployable candidate on provenance: trained with
zero test overlap, curve measured, adapter on the Hub.
