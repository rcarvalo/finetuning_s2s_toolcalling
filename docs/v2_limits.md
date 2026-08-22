# v2 limits — what the 200-case dissection actually shows

Source: `reports/final/v2_test_utterances.json` joined with the split parquet.
No new GPU run — these are the failures v2 already produced, read closely.

## Score by slice (the real map)

| Slice | Score | Δ vs best |
|---|---:|---|
| negatives (abstention) | 0.941 | — |
| `explicit arguments` | 0.889 | — |
| **`requires inference`** | **0.733** | **−15.6 pts** |
| `polite question` | 0.857 | — |
| indirect / direct | 0.825 / 0.812 | — |
| **`with disfluency`** | **0.750** | **−10.7 pts** |
| web_search | 0.822 | — |
| **db_query** | **0.711** | −11.1 pts (mostly metric, see below) |

**Limit 1 — inference-depth routing.** When the tool choice must be *deduced*
("can you identify our slowest-moving products?" → db_query with a reformulated
question), v2 loses ~16 points versus explicit requests. This is the largest
real gap and the place more/harder training data would pay.

**Limit 2 — disfluent speech.** "uh", "like", restarts cost ~11 points. This is
speech-specific and will get worse with real users than with TTS; noise/reverb
robustness is untested entirely (training audio is clean synthesis).

## The "argument problem" is mostly a measurement problem

The scorer already matches args with **token-F1 ≥ 0.7** (its default) — so the
76.5% "call" figure is the *tolerant* number, not exact match. Reading the 35
failures: nearly all are **faithful paraphrases** of a free-text question —

> expected `Who is managing the new website project?`
> got `Who is managing the project for the new website?` *(failed at F1 0.7)*

`db_query` takes a natural-language question **by design** (NL→SQL happens
backend-side); any faithful paraphrase is production-correct. Token-F1 on short
questions punishes word order and synonyms ("meetings on my calendar" vs
"events scheduled"). Of the 35, only ~2 are real semantic drifts (e.g.
"slowest-moving products" → "longest average order duration"). A semantic
judge — or the NL→SQL backend itself — is the honest arbiter here; db_query's
0.711 is mostly this artifact, which also explains db_query < web_search.

## The 3 false calls share one shape

All three are **encyclopedic questions**: "capital of Canada" (×2), "primary
function of a voice assistant". The dataset labels these "answer from memory";
v2 prefers to search. That boundary is genuinely fuzzy — a human assistant
might search too. It is a labeling-philosophy limit, not a routing failure:
v2 never called a tool on chit-chat or greetings.

## What this dissection cannot see (untested dimensions)

1. **Noise/reverb/real microphones** — all audio is clean TTS.
2. **Voice diversity** — 3 training voices; the 12-case held-out-voice set can
   only detect total collapse (one utterance = 12.5 pts there).
3. **Phase B** — tool-result reinjection and the spoken answer: untrained,
   unmeasured.
4. **Multi-intent, code-switching FR/EN, interruptions** — absent from data.
5. **Generator monoculture** — test and train both come from the same Gemini
   pipeline; the fresh 300-case set shares that DNA. Real-user phrasing is the
   next distribution shift.

## Ordered consequences

1. Clean fresh test set (300 unseen, held-out voices) → the uncontaminated
   verdict for every version. *(in flight — RunPod Voxtral job)*
2. Score db_query args with a semantic judge before buying data for a
   mostly-metric artifact.
3. If training again: weight `requires inference` and disfluent cases; add
   noise augmentation at packing.
4. Encyclopedic false calls: decide the *product* policy (is searching wrong?)
   before treating it as a model bug.
